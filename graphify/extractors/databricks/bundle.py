"""Databricks Asset Bundles (DABs): ``databricks.yml`` and its included resource YAML.

Turns a bundle into graph structure instead of a bag of YAML keys::

    bundle --includes--> resources/jobs.yml
    bundle --targets--> target(dev)
    job --has_task--> task --runs_notebook--> src/ingest.py
                      task --depends_on--> task
                      task --runs_pipeline--> pipeline --pipeline_source--> src/dlt/*.py
    job/task/pipeline --uses_variable--> variable (with per-target values)

Design choices
--------------
* The generic YAML extractor is NOT run on bundle files: its one-node-per-key
  output (``tasks``, ``notebook_task``, ``base_parameters`` ...) is noise that
  drowns the bundle semantics. A bundle file yields its file node plus the
  entities below.
* Entity ids are corpus-wide (``names.MERGEABLE_NODE_TYPES``), so a job declared
  in ``resources/jobs.yml`` and overridden under ``targets.dev.resources`` in
  ``databricks.yml`` is one node.
* A reference to an entity this file does not declare (``${resources.jobs.x.id}``,
  ``${var.catalog}`` in an included file) is emitted as an **edge only**, never
  as a sourceless stub node: build() merges same-id nodes attribute-by-attribute
  and a stub's empty ``source_file`` would blank the real declaration's. An edge
  whose target is declared nowhere in the corpus is dropped by build().
* Secrets never leave the file: ``workspace`` (host, profile, auth) and
  ``run_as``/``permissions`` principals are not read; variable values and
  parameters are omitted when the name or value looks credential-shaped.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterator

from graphify.extractors.base import _make_id, active_scan_root
from graphify.extractors.databricks.names import normalize_placeholders, uc_node_id
from graphify.extractors.databricks.paths import (
    NOTEBOOK_EXTENSIONS,
    file_ref_id,
    resolve_workspace_path,
)
from graphify.ids import make_id

_MAX_BYTES = 1_048_576  # 1 MiB, same cap as the generic YAML extractor
_SNIFF_BYTES = 64 * 1024
_MAX_DEPTH = 64  # recursive YAML aliases cannot loop the converter forever

BUNDLE_FILENAMES = frozenset({"databricks.yml", "databricks.yaml"})

# Top-level keys a bundle (or an included bundle file) may contain. A file with
# any other top-level key (a workflow's `on:`, compose's `services:`, k8s
# `apiVersion:`) is not a bundle file.
_BUNDLE_TOP_KEYS = frozenset({
    "bundle", "include", "variables", "targets", "environments", "resources",
    "workspace", "artifacts", "sync", "permissions", "run_as", "presets",
    "experimental", "scripts", "python", "git",
})

# Resource kinds of the bundle schema (singular form used in labels).
RESOURCE_KINDS: dict[str, str] = {
    "jobs": "job", "pipelines": "pipeline", "schemas": "schema", "volumes": "volume",
    "registered_models": "registered_model", "models": "model",
    "model_serving_endpoints": "model_serving_endpoint", "experiments": "experiment",
    "clusters": "cluster", "dashboards": "dashboard", "apps": "app",
    "quality_monitors": "quality_monitor", "sql_warehouses": "sql_warehouse",
    "secret_scopes": "secret_scope", "database_instances": "database_instance",
    "database_catalogs": "database_catalog", "synced_database_tables": "synced_database_table",
    "alerts": "alert", "catalogs": "catalog", "external_locations": "external_location",
    "postgres_projects": "postgres_project",
}

_HEAD_RX = re.compile(r"^(?:resources|bundle|targets|environments)\s*:", re.MULTILINE)
_VAR_REF_RX = re.compile(r"\$\{\s*var\.([A-Za-z_][\w-]*)")
_RESOURCE_REF_RX = re.compile(r"\$\{\s*resources\.([A-Za-z_]+)\.([A-Za-z_][\w-]*)(?:\.([\w.]+))?\s*\}")
_SECRET_NAME_RX = re.compile(r"(?i)(secret|token|password|passwd|pwd|credential|api[_-]?key|private[_-]?key|client[_-]?id)")
_GLOB_CHARS = frozenset("*?[")


# ── located YAML ─────────────────────────────────────────────────────────────


class _Map(dict):
    """A YAML mapping that remembers its own line and each key's line."""

    line: int = 1
    key_lines: dict


class _Seq(list):
    line: int = 1


def _scalar(node: Any) -> Any:
    tag = node.tag or ""
    val = node.value
    if tag.endswith(":null"):
        return None
    if tag.endswith(":bool"):
        return str(val).lower() in ("true", "yes", "on", "y")
    if tag.endswith(":int"):
        try:
            return int(str(val).replace("_", ""), 0)
        except ValueError:
            return val
    if tag.endswith(":float"):
        try:
            return float(val)
        except ValueError:
            return val
    return val


def _locate(node: Any, depth: int = 0) -> Any:
    """Composed YAML node -> plain values, mappings/sequences carrying lines.

    Works on the composed tree so unknown ``!tags`` never raise and every key
    keeps its written spelling and line. ``<<`` merges are applied with the
    same precedence as the generic YAML extractor.
    """
    import yaml  # type: ignore

    from graphify.extractors.yaml_config import _mapping_pairs

    if depth > _MAX_DEPTH:
        return None
    if isinstance(node, yaml.MappingNode):
        out = _Map()
        out.line = node.start_mark.line + 1
        out.key_lines = {}
        for key, key_node, val_node in _mapping_pairs(node):
            out[key] = _locate(val_node, depth + 1)
            out.key_lines[key] = key_node.start_mark.line + 1
        return out
    if isinstance(node, yaml.SequenceNode):
        seq = _Seq(_locate(v, depth + 1) for v in node.value)
        seq.line = node.start_mark.line + 1
        return seq
    if isinstance(node, yaml.ScalarNode):
        return _scalar(node)
    return None


def _load(path: Path) -> tuple[Any, str | None]:
    """(located top-level mapping or None, error)."""
    try:
        import yaml  # type: ignore
    except ImportError:
        return None, "pyyaml not installed; install with `pip install pyyaml`"
    try:
        with path.open("rb") as fh:
            raw = fh.read(_MAX_BYTES + 1)
    except OSError as exc:
        return None, str(exc)
    if len(raw) > _MAX_BYTES:
        return None, "yaml file too large to index"
    try:
        trees = list(yaml.compose_all(raw.decode("utf-8", errors="ignore"), Loader=yaml.SafeLoader))
    except Exception as exc:  # noqa: BLE001 - any parse failure is reported, not raised
        return None, str(exc)
    for tree in trees:
        if tree is not None:
            return _locate(tree), None
    return None, None


# ── ownership ────────────────────────────────────────────────────────────────

_owns_cache: dict[tuple[str, float], bool] = {}


def _looks_like_bundle(data: Any) -> bool:
    if not isinstance(data, dict) or not data:
        return False
    keys = {str(k) for k in data}
    if not keys <= _BUNDLE_TOP_KEYS:
        return False
    if "bundle" in keys:
        return True
    resources = data.get("resources")
    if isinstance(resources, dict) and set(map(str, resources)) & set(RESOURCE_KINDS):
        return True
    for section in ("targets", "environments"):
        targets = data.get(section)
        if isinstance(targets, dict) and any(
            isinstance(t, dict) and ({"resources", "variables", "mode", "workspace", "default"} & set(map(str, t)))
            for t in targets.values()
        ):
            return True
    return False


def owns(path: Path) -> bool:
    """True for ``databricks.yml`` and YAML files whose top level is bundle-shaped."""
    if path.suffix.lower() not in (".yml", ".yaml"):
        return False
    if path.name.lower() in BUNDLE_FILENAMES:
        return True
    try:
        with path.open("rb") as fh:
            head = fh.read(_SNIFF_BYTES).decode("utf-8", errors="ignore")
    except OSError:
        return False
    if not _HEAD_RX.search(head):
        return False
    try:
        key = (str(path), path.stat().st_mtime)
    except OSError:
        return False
    hit = _owns_cache.get(key)
    if hit is None:
        data, _err = _load(path)
        hit = _looks_like_bundle(data)
        if len(_owns_cache) > 4096:
            _owns_cache.clear()
        _owns_cache[key] = hit
    return hit


# ── helpers ──────────────────────────────────────────────────────────────────


def _find_bundle_root(path: Path) -> Path:
    """The directory of the nearest ``databricks.yml`` at or above ``path``,
    no higher than the scan root; the file's own directory if there is none."""
    scan_root = active_scan_root()
    stop = scan_root.resolve() if scan_root is not None else None
    directory = path.parent
    while True:
        for name in BUNDLE_FILENAMES:
            if (directory / name).is_file():
                return directory
        try:
            if stop is not None and directory.resolve() == stop:
                break
        except OSError:
            break
        if directory.parent == directory:
            break
        directory = directory.parent
    return path.parent


def _within_scan_root(p: Path) -> bool:
    root = active_scan_root()
    if root is None:
        return True
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _glob(base: Path, pattern: str) -> list[Path]:
    """Files matching ``pattern`` relative to ``base`` (``..`` and ``**`` allowed),
    keeping ``base``'s absolute/relative form, sorted, inside the scan root."""
    pattern = re.sub(r"^\$\{\s*workspace\.(?:file_path|root_path)\s*\}(?:/files)?/?", "", pattern.strip())
    if not pattern or pattern.startswith("/") or "${" in pattern:
        return []
    full = os.path.normpath(os.path.join(str(base), pattern))
    parts = Path(full).parts
    idx = next((i for i, part in enumerate(parts) if _GLOB_CHARS & set(part)), None)
    if idx is None:
        candidates = [Path(full)]
    else:
        anchor = Path(*parts[:idx]) if idx else Path(".")
        rest = "/".join(parts[idx:])
        try:
            candidates = list(anchor.glob(rest))
        except (ValueError, OSError):
            return []
    out = []
    for c in candidates:
        try:
            if c.is_file() and _within_scan_root(c):
                out.append(c)
        except OSError:
            continue
    return sorted(out)


def _is_secretish(name: str, value: Any = None) -> bool:
    """Whether ``name`` or anything inside ``value`` looks like a credential.

    Complex variables (dicts/lists) are walked recursively: an innocuous outer
    name such as ``connection`` must not smuggle a nested ``client_secret`` key
    or a token-shaped string into the graph.
    """
    if _SECRET_NAME_RX.search(str(name)):
        return True
    if isinstance(value, str):
        from graphify.security import redact_secrets

        return redact_secrets(value) != value
    if isinstance(value, dict):
        return any(_is_secretish(k, v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return any(_is_secretish("", v) for v in value)
    return False


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def _line_of(container: Any, key: str, default: int) -> int:
    lines = getattr(container, "key_lines", None)
    if lines and key in lines:
        return lines[key]
    return getattr(container, "line", default)


def _resource_id(kind: str, key: str) -> str:
    if kind == "jobs":
        return make_id("dab_job", key)
    if kind == "pipelines":
        return make_id("dab_pipeline", key)
    return make_id("dab_resource", kind, key)


def _plain(value: Any) -> Any:
    """Located containers -> plain dict/list (graph attributes must be JSON)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


# ── extraction ───────────────────────────────────────────────────────────────


class _Emitter:
    def __init__(self, path: Path):
        self.path = path
        self.str_path = str(path)
        self.file_nid = _make_id(self.str_path)
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self.nodes[self.file_nid] = {
            "id": self.file_nid, "label": path.name, "file_type": "code",
            "source_file": self.str_path, "source_location": "L1", "type": "dab_file",
        }

    def node(self, nid: str, label: str, ntype: str, line: int, **attrs: Any) -> dict:
        """Declare (or extend) a node this file declares, contained by the file."""
        existing = self.nodes.get(nid)
        if existing is None:
            existing = {
                "id": nid, "label": label, "file_type": "code", "type": ntype,
                "source_file": self.str_path, "source_location": f"L{line}",
            }
            self.nodes[nid] = existing
            self.edge(self.file_nid, nid, "contains", line)
        for k, v in attrs.items():
            if v is None or v == [] or v == {}:
                continue
            if k == "targets":
                bucket = existing.setdefault("targets", [])
                for t in v:
                    if t not in bucket:
                        bucket.append(t)
            elif k not in existing or existing[k] in (None, "", [], {}):
                existing[k] = v
        return existing

    def edge(self, src: str, tgt: str, relation: str, line: int, **attrs: Any) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, relation)
        if key in self._edge_keys:
            return
        self._edge_keys.add(key)
        edge = {
            "source": src, "target": tgt, "relation": relation, "confidence": "EXTRACTED",
            "source_file": self.str_path, "source_location": f"L{line}", "weight": 1.0,
        }
        edge.update({k: v for k, v in attrs.items() if v is not None})
        self.edges.append(edge)

    def result(self) -> dict:
        return {"nodes": list(self.nodes.values()), "edges": self.edges}


class _BundleExtractor:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.yaml_dir = path.parent
        self.bundle_root = _find_bundle_root(path)
        self.out = _Emitter(path)
        # resources this file declares: (kind, key) -> node id
        self.declared: dict[tuple[str, str], str] = {}
        self.declared_vars: set[str] = set()
        # schema resource key -> normalized UC schema name (ids are keyed by name)
        self.schema_names: dict[str, str] = {}
        self.owner = self.out.file_nid  # bundle node once known

    # -- top level -------------------------------------------------------

    def run(self) -> dict:
        d = self.data
        bundle = d.get("bundle")
        if isinstance(bundle, dict) and bundle.get("name"):
            name = str(bundle["name"])
            nid = make_id("dab_bundle", normalize_placeholders(name))
            self.out.node(nid, name, "dab_bundle", _line_of(d, "bundle", 1),
                          dab_name=name, databricks_cli_version=bundle.get("databricks_cli_version"))
            self.owner = nid
        self._includes(d.get("include"), _line_of(d, "include", 1))
        self._variables(d.get("variables"))
        # Pre-register every resource key declared anywhere in this file so
        # references resolve regardless of order.
        self._collect_declared(d.get("resources"))
        for section in ("targets", "environments"):
            targets = d.get(section)
            if isinstance(targets, dict):
                for spec in targets.values():
                    if isinstance(spec, dict):
                        self._collect_declared(spec.get("resources"))
        self._resources(d.get("resources"), target=None)
        for section in ("targets", "environments"):
            self._targets(d.get(section), section)
        return self.out.result()

    def _includes(self, include: Any, line: int) -> None:
        if not isinstance(include, list):
            return
        globs = [str(g) for g in include if isinstance(g, str)]
        for pattern in globs:
            for match in _glob(self.yaml_dir, pattern):
                if match.suffix.lower() in (".yml", ".yaml") and match != self.path:
                    self.out.edge(self.owner, file_ref_id(match), "includes", line, pattern=pattern)
        if self.owner in self.out.nodes:
            self.out.nodes[self.owner]["dab_include"] = globs

    def _variables(self, variables: Any) -> None:
        if not isinstance(variables, dict):
            return
        for name, spec in variables.items():
            name = str(name)
            line = _line_of(variables, name, 1)
            attrs: dict[str, Any] = {"dab_name": name}
            if isinstance(spec, dict):
                attrs["description"] = spec.get("description") if isinstance(spec.get("description"), str) else None
                if isinstance(spec.get("lookup"), dict):
                    attrs["dab_lookup"] = next(iter(spec["lookup"]), None)
                if spec.get("type"):
                    attrs["dab_var_type"] = str(spec["type"])
                default = spec.get("default")
            else:
                default = spec
            if default is not None and not _is_secretish(name, default):
                attrs["dab_default"] = _plain(default) if isinstance(default, (dict, list)) else default
            self.out.node(make_id("dab_variable", name), name, "dab_variable", line, **attrs)
            self.declared_vars.add(name)
            self.out.edge(self.owner, make_id("dab_variable", name), "declares_variable", line)

    def _collect_declared(self, resources: Any) -> None:
        if not isinstance(resources, dict):
            return
        for kind, items in resources.items():
            if not isinstance(items, dict):
                continue
            for key, spec in items.items():
                kind, key = str(kind), str(key)
                if kind == "schemas":
                    name = spec.get("name") if isinstance(spec, dict) else None
                    if isinstance(name, str) and name:
                        norm = normalize_placeholders(name).lower()
                        self.schema_names[key] = norm
                        self.declared[(kind, key)] = uc_node_id("uc_schema", norm)
                elif (kind, key) not in self.declared:
                    self.declared[(kind, key)] = _resource_id(kind, key)

    def _targets(self, targets: Any, section: str) -> None:
        if not isinstance(targets, dict):
            return
        for name, spec in targets.items():
            name = str(name)
            line = _line_of(targets, name, 1)
            spec = spec if isinstance(spec, dict) else {}
            overrides: dict[str, Any] = {}
            variables = spec.get("variables")
            if isinstance(variables, dict):
                for var, val in variables.items():
                    var = str(var)
                    if isinstance(val, dict):
                        val = val.get("default", {"lookup": next(iter(val.get("lookup") or {}), None)}
                                      if isinstance(val.get("lookup"), dict) else None)
                    if val is None or _is_secretish(var, val):
                        continue
                    overrides[var] = _plain(val)
            tnid = make_id("dab_target", name)
            self.out.node(tnid, name, "dab_target", line, dab_name=name,
                          dab_mode=spec.get("mode"), dab_default=spec.get("default") is True,
                          dab_variables=overrides or None)
            self.out.edge(self.owner, tnid, "targets", line)
            for var, val in overrides.items():
                vline = _line_of(variables, var, line)
                self.out.edge(tnid, make_id("dab_variable", var), "uses_variable", vline, context="override")
                vnid = make_id("dab_variable", var)
                if vnid in self.out.nodes:
                    self.out.nodes[vnid].setdefault("dab_env_values", {})[name] = val
            self._resources(spec.get("resources"), target=name)

    # -- resources -------------------------------------------------------

    def _resources(self, resources: Any, *, target: str | None) -> None:
        if not isinstance(resources, dict):
            return
        for kind, items in resources.items():
            kind = str(kind)
            if not isinstance(items, dict):
                continue
            for key, spec in items.items():
                key = str(key)
                spec = spec if isinstance(spec, dict) else _Map()
                line = _line_of(items, key, getattr(items, "line", 1))
                targets = [target] if target else None
                if kind == "jobs":
                    self._job(key, spec, line, targets)
                elif kind == "pipelines":
                    self._pipeline(key, spec, line, targets)
                elif kind == "schemas":
                    self._schema(key, spec, line, targets)
                elif kind == "volumes":
                    self._volume(key, spec, line, targets)
                else:
                    nid = _resource_id(kind, key)
                    label = str(spec.get("name") or key)
                    self.out.node(nid, label, "dab_resource", line, dab_kind=RESOURCE_KINDS.get(kind, kind),
                                  dab_key=key, targets=targets)
                    self._refs(nid, spec, line)
                resource_nid = self.declared.get((kind, key))
                if target and resource_nid:
                    self.out.edge(make_id("dab_target", target), resource_nid, "overrides", line)

    def _refs(self, src: str, spec: Any, line: int, *, skip: tuple[str, ...] = ()) -> None:
        """uses_variable / references edges for every ${var.x} / ${resources...}
        string anywhere in ``spec`` (keys in ``skip`` handled by the caller)."""
        values = {k: v for k, v in spec.items() if k not in skip} if isinstance(spec, dict) else spec
        for text in _strings(values):
            for m in _VAR_REF_RX.finditer(text):
                self.out.edge(src, make_id("dab_variable", m.group(1)), "uses_variable", line)
            for m in _RESOURCE_REF_RX.finditer(text):
                tgt = self._resource_ref(m)
                if tgt:
                    self.out.edge(src, tgt, "references", line)

    def _resource_ref(self, m: re.Match) -> str | None:
        kind, key = m.group(1), m.group(2)
        if kind in ("schemas", "volumes"):
            # Keyed by UC name, known only where declared; no guess otherwise.
            return self.declared.get((kind, key))
        return self.declared.get((kind, key)) or _resource_id(kind, key)

    def _job(self, key: str, spec: dict, line: int, targets: list[str] | None) -> None:
        jnid = make_id("dab_job", key)
        attrs: dict[str, Any] = {"dab_key": key, "dab_name": spec.get("name"), "targets": targets}
        sched = spec.get("schedule")
        if isinstance(sched, dict):
            attrs["dab_schedule"] = sched.get("quartz_cron_expression")
        trig = spec.get("trigger")
        if isinstance(trig, dict):
            attrs["dab_trigger"] = sorted(k for k in trig if k != "pause_status")
        if spec.get("continuous"):
            attrs["dab_continuous"] = True
        params = spec.get("parameters")
        if isinstance(params, list):
            attrs["dab_parameters"] = [str(p.get("name")) for p in params if isinstance(p, dict) and p.get("name")]
        clusters = spec.get("job_clusters")
        if isinstance(clusters, list):
            attrs["dab_job_clusters"] = [str(c.get("job_cluster_key")) for c in clusters
                                         if isinstance(c, dict) and c.get("job_cluster_key")]
        envs = spec.get("environments")
        if isinstance(envs, list):
            attrs["dab_environments"] = [str(e.get("environment_key")) for e in envs
                                         if isinstance(e, dict) and e.get("environment_key")]
        label = str(spec.get("name") or key)
        self.out.node(jnid, label, "dab_job", line, **attrs)
        self._refs(jnid, spec, line, skip=("tasks", "permissions", "run_as"))
        tasks = spec.get("tasks")
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    self._task(key, jnid, task, getattr(task, "line", line), targets, git=bool(spec.get("git_source")))

    def _resolve(self, ref: Any) -> Path | None:
        if not isinstance(ref, str) or not ref:
            return None
        hit = resolve_workspace_path(ref, self.yaml_dir, bundle_root=self.bundle_root)
        if hit is None:
            scan_root = active_scan_root()
            # GIT-sourced paths are relative to the repository root.
            if scan_root is not None and not ref.startswith("/"):
                hit = resolve_workspace_path(ref, self.yaml_dir, bundle_root=self._rel_like(scan_root))
        return hit

    def _root_rel(self, ref: str) -> str | None:
        """``ref`` (relative to this YAML) as a scan-root-relative posix path or
        glob, for the corpus pass to bind against files present at build time
        (the per-file cache cannot know about notebooks added later)."""
        import re as _re
        scan_root = active_scan_root()
        if scan_root is None or not isinstance(ref, str) or not ref or ref.startswith("/"):
            return None
        ref = _re.sub(r"^\$\{\s*workspace\.(?:file_path|root_path)\s*\}(?:/files)?/?", "", ref.strip())
        if "${" in ref:
            return None
        try:
            full = os.path.normpath(os.path.join(str(self.yaml_dir.resolve()), ref))
            rel = os.path.relpath(full, str(Path(scan_root).resolve()))
        except ValueError:
            return None
        rel = rel.replace(os.sep, "/")
        return None if rel.startswith("..") else rel

    def _rel_like(self, root: Path) -> Path:
        """``root`` in the same absolute/relative form as this file's path."""
        if self.path.is_absolute():
            return root.resolve()
        try:
            return Path(os.path.relpath(root.resolve(), Path.cwd()))
        except ValueError:
            return root

    def _task(self, job_key: str, jnid: str, task: dict, line: int,
              targets: list[str] | None, *, git: bool = False, parent: str | None = None) -> None:
        task_key = task.get("task_key")
        if not task_key:
            return
        task_key = str(task_key)
        tnid = make_id("dab_task", job_key, task_key)
        kind = next((k for k in task if str(k).endswith("_task") and isinstance(task[k], dict)), None)
        attrs: dict[str, Any] = {"dab_job": job_key, "dab_task_key": task_key,
                                 "dab_task_type": kind, "targets": targets}
        for k in ("job_cluster_key", "environment_key", "existing_cluster_id", "run_if", "max_retries"):
            if task.get(k) is not None:
                attrs[f"dab_{k}"] = task[k]
        self.out.node(tnid, f"{job_key}.{task_key}", "dab_task", line, **attrs)
        self.out.edge(parent or jnid, tnid, "has_task", line, context="for_each" if parent else None)
        body = task.get(kind) if kind else None
        body_line = _line_of(task, kind, line) if kind else line
        if isinstance(body, dict):
            self._task_body(job_key, tnid, kind, body, body_line, targets, git=git)
        deps = task.get("depends_on")
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, dict) and dep.get("task_key"):
                    self.out.edge(tnid, make_id("dab_task", job_key, str(dep["task_key"])), "depends_on",
                                  getattr(dep, "line", line), outcome=dep.get("outcome"))
        self._libraries(tnid, task.get("libraries"), _line_of(task, "libraries", line))
        self._refs(tnid, task, line, skip=(kind or "", "libraries") if kind == "for_each_task" else ("libraries",))

    def _task_body(self, job_key: str, tnid: str, kind: str, body: dict, line: int,
                   targets: list[str] | None, *, git: bool) -> None:
        node = self.out.nodes[tnid]

        def link_file(ref: Any, relation: str, attr: str) -> None:
            if not isinstance(ref, str):
                return
            node[attr] = ref
            rel = self._root_rel(ref)
            if rel:
                node.setdefault("dab_path_refs", {})[relation] = rel
            hit = self._resolve(ref)
            if hit is not None:
                self.out.edge(tnid, file_ref_id(hit), relation, line)

        if kind == "notebook_task":
            node["dab_source"] = str(body.get("source") or ("GIT" if git else "WORKSPACE"))
            link_file(body.get("notebook_path"), "runs_notebook", "dab_notebook_path")
            params = body.get("base_parameters")
            if isinstance(params, dict):
                node["dab_parameters"] = [str(k) for k in params]
        elif kind == "spark_python_task":
            link_file(body.get("python_file"), "runs_file", "dab_python_file")
        elif kind == "python_wheel_task":
            pkg = body.get("package_name")
            if isinstance(pkg, str) and pkg:
                node["dab_package_name"] = pkg
                # Package nodes are keyed by name (manifest_ingest._pkg_id); an
                # external wheel has none and build() drops the edge.
                self.out.edge(tnid, make_id("pkg", pkg), "runs_package", line)
            if body.get("entry_point"):
                node["dab_entry_point"] = str(body["entry_point"])
            named = body.get("named_parameters")
            if isinstance(named, dict):
                node["dab_parameters"] = [str(k) for k in named]
        elif kind == "sql_task":
            f = body.get("file")
            if isinstance(f, dict):
                link_file(f.get("path"), "runs_file", "dab_sql_file")
            for sub, field in (("query", "query_id"), ("alert", "alert_id"), ("dashboard", "dashboard_id")):
                if isinstance(body.get(sub), dict) and body[sub].get(field):
                    node[f"dab_{field}"] = str(body[sub][field])
        elif kind == "pipeline_task":
            self._ref_edge(tnid, body.get("pipeline_id"), "runs_pipeline", "pipelines", line)
            if body.get("full_refresh"):
                node["dab_full_refresh"] = True
        elif kind == "run_job_task":
            self._ref_edge(tnid, body.get("job_id"), "runs_job", "jobs", line)
        elif kind == "dbt_task":
            if body.get("project_directory"):
                node["dab_project_directory"] = str(body["project_directory"])
            if isinstance(body.get("commands"), list):
                node["dab_commands"] = [str(c) for c in body["commands"]]
        elif kind == "for_each_task":
            if body.get("inputs") is not None:
                node["dab_inputs"] = str(body["inputs"]) if not isinstance(body["inputs"], (list, dict)) \
                    else _plain(body["inputs"])
            inner = body.get("task")
            if isinstance(inner, dict):
                self._task(job_key, tnid, inner, getattr(inner, "line", line),
                           targets, git=git, parent=tnid)
        elif kind == "condition_task":
            node["dab_condition"] = " ".join(str(body.get(k, "")) for k in ("left", "op", "right")).strip()
        elif kind == "spark_jar_task":
            if body.get("main_class_name"):
                node["dab_main_class"] = str(body["main_class_name"])
        elif kind == "spark_submit_task":
            pass
        # Every other *_task kind (dashboard_task, power_bi_task, clean_rooms_notebook_task,
        # gen_ai_compute_task, ...) is recorded by dab_task_type only.
        if kind != "for_each_task":
            self._refs(tnid, body, line, skip=("pipeline_id", "job_id"))

    def _ref_edge(self, src: str, value: Any, relation: str, kind: str, line: int) -> None:
        if not isinstance(value, str):
            return
        m = _RESOURCE_REF_RX.search(value)
        if m and m.group(1) == kind:
            self.out.edge(src, self.declared.get((kind, m.group(2)), _resource_id(kind, m.group(2))),
                          relation, line)
        else:
            self.out.nodes[src][f"dab_{relation}_id"] = value  # a literal id outside the bundle

    def _libraries(self, src: str, libraries: Any, line: int) -> None:
        if not isinstance(libraries, list):
            return
        pypi: list[str] = []
        for lib in libraries:
            if not isinstance(lib, dict):
                continue
            lline = getattr(lib, "line", line)
            for kind in ("whl", "jar", "requirements", "egg"):
                ref = lib.get(kind)
                if isinstance(ref, str):
                    for hit in _glob(self.yaml_dir, ref) if not ref.startswith("/") else []:
                        self.out.edge(src, file_ref_id(hit), "uses_library", lline, library_kind=kind)
            spec = lib.get("pypi")
            if isinstance(spec, dict) and isinstance(spec.get("package"), str):
                name = re.split(r"[\s<>=!~\[;]", spec["package"].strip(), maxsplit=1)[0]
                if name:
                    pypi.append(spec["package"])
                    self.out.edge(src, make_id("pkg", name), "uses_library", lline, library_kind="pypi")
        if pypi and src in self.out.nodes:
            self.out.nodes[src]["dab_pypi"] = pypi

    def _pipeline(self, key: str, spec: dict, line: int, targets: list[str] | None) -> None:
        pnid = make_id("dab_pipeline", key)
        catalog = spec.get("catalog")
        schema = spec.get("schema") if spec.get("schema") is not None else spec.get("target")
        attrs: dict[str, Any] = {"dab_key": key, "dab_name": spec.get("name"), "targets": targets}
        if isinstance(catalog, str):
            attrs["uc_catalog"] = catalog
            attrs["uc_catalog_norm"] = normalize_placeholders(catalog).lower()
        if isinstance(schema, str):
            attrs["uc_schema"] = schema
            resolved = self._schema_name_from_ref(schema)
            attrs["uc_schema_norm"] = (resolved or normalize_placeholders(schema)).lower()
        for k in ("serverless", "continuous", "channel", "edition", "photon", "development"):
            if spec.get(k) is not None:
                attrs[f"dab_{k}"] = spec[k]
        conf = spec.get("configuration")
        if isinstance(conf, dict):
            attrs["dab_configuration"] = sorted(str(k) for k in conf if not _is_secretish(k))
        label = str(spec.get("name") or key)
        self.out.node(pnid, label, "dab_pipeline", line, **attrs)
        libraries = spec.get("libraries")
        unresolved: list[str] = []
        library_refs: list[str] = []
        if isinstance(libraries, list):
            for lib in libraries:
                if not isinstance(lib, dict):
                    continue
                lline = getattr(lib, "line", line)
                for kind in ("notebook", "file"):
                    entry = lib.get(kind)
                    if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                        ref = entry["path"]
                        if (rel := self._root_rel(ref)):
                            library_refs.append(rel)
                        hit = self._resolve(ref)
                        if hit is None and _GLOB_CHARS & set(ref):
                            hits = _glob(self.yaml_dir, ref)
                        else:
                            hits = [hit] if hit is not None else []
                        if not hits:
                            unresolved.append(ref)
                        for h in hits:
                            self.out.edge(pnid, file_ref_id(h), "pipeline_source", lline, library_kind=kind)
                glob = lib.get("glob")
                if isinstance(glob, dict) and isinstance(glob.get("include"), str):
                    pattern = glob["include"]
                    if (rel := self._root_rel(pattern)):
                        library_refs.append(rel)
                    hits = [h for h in _glob(self.yaml_dir, pattern)
                            if h.suffix in NOTEBOOK_EXTENSIONS]
                    if not hits and not (_GLOB_CHARS & set(pattern)):
                        # A plain directory include means everything under it.
                        hits = [h for h in _glob(self.yaml_dir, pattern.rstrip("/") + "/**/*")
                                if h.suffix in NOTEBOOK_EXTENSIONS]
                    if not hits:
                        unresolved.append(pattern)
                    for h in hits:
                        self.out.edge(pnid, file_ref_id(h), "pipeline_source", lline, library_kind="glob")
        if spec.get("root_path"):
            self.out.nodes[pnid]["dab_root_path"] = str(spec["root_path"])
        if unresolved:
            self.out.nodes[pnid]["dab_unresolved_libraries"] = unresolved
        if library_refs:
            self.out.nodes[pnid]["dab_library_refs"] = library_refs
        self._refs(pnid, spec, line, skip=("permissions", "run_as"))

    def _schema_name_from_ref(self, value: str) -> str | None:
        m = _RESOURCE_REF_RX.fullmatch(value.strip())
        if m and m.group(1) == "schemas":
            return self.schema_names.get(m.group(2))
        return None

    def _schema(self, key: str, spec: dict, line: int, targets: list[str] | None) -> None:
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            return
        norm = normalize_placeholders(name).lower()
        nid = uc_node_id("uc_schema", norm)
        catalog = spec.get("catalog_name")
        self.out.node(nid, norm, "uc_schema", line, dab_key=key, targets=targets,
                      uc_catalogs=[normalize_placeholders(catalog).lower()] if isinstance(catalog, str) else None)
        self._refs(nid, spec, line, skip=("grants",))

    def _volume(self, key: str, spec: dict, line: int, targets: list[str] | None) -> None:
        name = spec.get("name")
        schema = spec.get("schema_name")
        if not isinstance(name, str) or not name:
            return
        schema_norm = None
        if isinstance(schema, str):
            schema_norm = self._schema_name_from_ref(schema) or normalize_placeholders(schema).lower()
        label = ".".join(p for p in (schema_norm, normalize_placeholders(name).lower()) if p)
        nid = uc_node_id("uc_volume", label)
        catalog = spec.get("catalog_name")
        self.out.node(nid, label, "uc_volume", line, dab_key=key, targets=targets,
                      dab_volume_type=spec.get("volume_type"),
                      uc_catalogs=[normalize_placeholders(catalog).lower()] if isinstance(catalog, str) else None)
        self.declared[("volumes", key)] = nid
        if schema_norm:
            self.out.edge(nid, uc_node_id("uc_schema", schema_norm), "in_schema", line)
        self._refs(nid, spec, line, skip=("grants",))


def extract(path: Path) -> dict:
    """Extract a Databricks Asset Bundle file (see module docstring)."""
    data, error = _load(path)
    if error:
        return {"nodes": [], "edges": [], "error": error}
    if not isinstance(data, dict):
        out = _Emitter(path)
        return out.result()
    return _BundleExtractor(path, data).run()
