"""Corpus-level Databricks pass, run on the combined extraction before graph build.

Per-file extractors cannot see the whole repo, so this pass finishes the job
once every file's nodes and edges are in one list. It is called from
``graphify.build.build`` (before dedup) and ``build_from_json``, the two
functions every full, incremental and watch rebuild goes through, and it is
idempotent and a no-op for a corpus with no Databricks nodes.

Steps, in order:

1. **Merge corpus-wide entities.** Every file that mentions a table (or a
   bundle job, task, pipeline, ...) emits its own copy under the same id;
   NetworkX would keep the last copy's attributes, blanking a definition's
   ``source_file`` when a reader came later. Copies are folded into one dict:
   the copy with a ``source_file`` wins position, list attributes are unioned,
   dict attributes merged, and a known ``uc_kind`` beats ``"unknown"``.
2. **Learn environment catalogs**: values a bundle variable whose name contains
   ``catalog`` takes (its default and every per-target value) plus
   ``databricks.env_catalogs``. Each table gets ``uc_env_catalogs``: the
   catalogs it was written with that are environment switches or placeholders.
   Under ``databricks.catalog_identity=keep`` a table written with an env
   catalog (``dev.silver.orders``) is re-keyed onto the placeholder form
   (``{catalog}.silver.orders``, named after the variable) so environments
   merge while genuinely different catalogs stay apart.
3. **Qualify DLT / Lakeflow datasets.** A one-part table (``orders_clean``, as
   ``@dlt.table`` or ``CREATE STREAMING TABLE`` names it) referenced from a
   file that a bundle pipeline lists in ``libraries`` lives in that pipeline's
   schema; those references move to ``<schema>.<name>`` (catalog too under
   ``keep``), merging with fully qualified references elsewhere.
4. **Wheel tasks.** A ``python_wheel_task`` is linked to its package node and,
   through ``[project.scripts]`` / entry-point tables in the matching
   ``pyproject.toml`` (or Databricks' fallback: function ``entry_point`` in
   package ``package_name``), to the function it runs (``runs_entry_point``).
5. **Edge cleanup.** Exact duplicate edges are dropped. When one symbol both
   reads and writes a table, the undirected graph can keep only one edge per
   pair, so the pair is folded into a single ``writes_to`` edge carrying
   ``also_reads=True`` instead of losing one fact at random.
"""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.databricks.names import (
    MERGEABLE_NODE_TYPES,
    TABLE_NODE_TYPE,
    parse_table_name,
    table_key,
    table_node_id,
)

_TABLE_RELATIONS = ("reads_from", "writes_to")


def _has_databricks(nodes: list) -> bool:
    for n in nodes:
        if isinstance(n, dict):
            t = n.get("type")
            if t in MERGEABLE_NODE_TYPES or (isinstance(t, str) and t.startswith("dab_")):
                return True
    return False


def run(nodes: list, edges: list, root: str | Path | None = None) -> None:
    """Resolve Databricks entities in place (see module docstring)."""
    if not _has_databricks(nodes):
        return
    from graphify.extractors.databricks.config import get_config, load_config

    root_path = Path(root).resolve() if root else None
    cfg = load_config(root_path) if root_path is not None else get_config()
    keep = cfg.catalog_identity == "keep"

    _merge_entities(nodes)
    env_catalogs = _learn_env_catalogs(nodes, cfg)
    if keep:
        _rekey_env_catalogs(nodes, edges, env_catalogs)
    _late_bind_paths(nodes, edges, root_path)
    _qualify_pipeline_datasets(nodes, edges, keep, root_path)
    _annotate_env_catalogs(nodes, env_catalogs)
    _link_wheel_tasks(nodes, edges, root_path)
    _clean_edges(nodes, edges)


# ── 1. entity merge ──────────────────────────────────────────────────────────

def _merge_into(into: dict, other: dict) -> None:
    if not into.get("source_file") and other.get("source_file"):
        for k in ("source_file", "source_location"):
            if other.get(k):
                into[k] = other[k]
    for k, v in other.items():
        if k in ("id", "source_file", "source_location"):
            continue
        cur = into.get(k)
        if cur is None or cur == "" or cur == []:
            into[k] = v
        elif isinstance(cur, list) and isinstance(v, list):
            for item in v:
                if item not in cur:
                    cur.append(item)
        elif isinstance(cur, dict) and isinstance(v, dict):
            for dk, dv in v.items():
                cur.setdefault(dk, dv)
        elif k == "uc_kind" and cur == "unknown" and v != "unknown":
            into[k] = v


def _merge_entities(nodes: list) -> None:
    first: dict[str, dict] = {}
    out: list = []
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") not in MERGEABLE_NODE_TYPES:
            out.append(n)
            continue
        nid = n.get("id")
        kept = first.get(nid)
        if kept is None:
            # Copy so callers' per-file result dicts are never mutated through us.
            kept = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                    for k, v in n.items()}
            first[nid] = kept
            out.append(kept)
        else:
            _merge_into(kept, n)
    nodes[:] = out


# ── 2. environment catalogs ──────────────────────────────────────────────────

def _learn_env_catalogs(nodes: list, cfg) -> dict[str, str]:
    """env catalog value -> placeholder name it stands for (``dev`` -> ``catalog``)."""
    env: dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != "dab_variable":
            continue
        name = str(n.get("label") or n.get("dab_name") or "").lower()
        if "catalog" not in name:
            continue
        values = [n.get("dab_default")]
        values += list((n.get("dab_env_values") or {}).values())
        for v in values:
            if isinstance(v, str) and v and "${" not in v:
                env.setdefault(v.strip().lower(), name)
    for c in cfg.env_catalogs:
        env.setdefault(c, "catalog")
    return env


def _annotate_env_catalogs(nodes: list, env: dict[str, str]) -> None:
    for n in nodes:
        if isinstance(n, dict) and n.get("type") == TABLE_NODE_TYPE:
            cats = n.get("uc_catalogs") or []
            switched = [c for c in cats if c in env or c.startswith("{")]
            if switched:
                n["uc_env_catalogs"] = switched


def _rekey_env_catalogs(nodes: list, edges: list, env: dict[str, str]) -> None:
    if not env:
        return
    remap: dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict) or n.get("type") != TABLE_NODE_TYPE:
            continue
        tn = parse_table_name(str(n.get("label", "")))
        if tn is None or not tn.catalog or tn.catalog not in env:
            continue
        new_label = "{" + env[tn.catalog] + "}." + table_key(tn)
        remap[n["id"]] = new_label
    _rename_tables(nodes, edges, remap)


# ── 2b. late path binding ────────────────────────────────────────────────────

_NOTEBOOK_EXTS = (".py", ".ipynb", ".sql", ".scala", ".r", ".R")


def _glob_rx(pattern: str) -> "re.Pattern[str]":
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out))


def _late_bind_paths(nodes: list, edges: list, root: Path | None) -> None:
    """Bind bundle task paths and pipeline libraries to files in the corpus.

    The bundle extractor resolves them against the disk when the YAML is
    extracted, but its cached result cannot see a notebook added afterwards.
    ``dab_path_refs`` / ``dab_library_refs`` hold the same references as
    scan-root-relative paths and globs, so every build re-binds them against
    the files actually in the graph."""
    tasks = [n for n in nodes if isinstance(n, dict) and n.get("dab_path_refs")]
    pipes = [n for n in nodes if isinstance(n, dict) and n.get("dab_library_refs")]
    if not tasks and not pipes:
        return
    files: dict[str, str] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        key = _sf_key(n.get("source_file"), root)
        if key and str(n.get("label", "")) == key.rsplit("/", 1)[-1]:
            files.setdefault(key, n["id"])
    if not files:
        return
    existing = {(e.get("source"), e.get("target"), e.get("relation")) for e in edges if isinstance(e, dict)}

    def add(src: dict, fid: str, relation: str, **extra) -> None:
        if (src["id"], fid, relation) in existing:
            return
        existing.add((src["id"], fid, relation))
        edges.append({"source": src["id"], "target": fid, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": src.get("source_file", ""),
                      "source_location": src.get("source_location", ""), "weight": 1.0, **extra})

    def lookup(path: str) -> str | None:
        for cand in (path, *(path + ext for ext in _NOTEBOOK_EXTS)):
            if cand in files:
                return files[cand]
        return None

    for task in tasks:
        for relation, path in (task.get("dab_path_refs") or {}).items():
            fid = lookup(path) if isinstance(path, str) else None
            if fid:
                add(task, fid, relation)
    for pipe in pipes:
        for pattern in pipe.get("dab_library_refs") or []:
            if not isinstance(pattern, str):
                continue
            if any(c in pattern for c in "*?["):
                rx = _glob_rx(pattern)
                hits = [fid for key, fid in files.items()
                        if key.endswith(_NOTEBOOK_EXTS) and rx.fullmatch(key)]
            else:
                fid = lookup(pattern)
                prefix = pattern.rstrip("/") + "/"
                hits = [fid] if fid else [f for key, f in files.items()
                                          if key.startswith(prefix) and key.endswith(_NOTEBOOK_EXTS)]
            for fid in hits:
                add(pipe, fid, "pipeline_source")


# ── 3. DLT / Lakeflow dataset qualification ─────────────────────────────────

def _sf_key(sf, root: Path | None) -> str:
    """Comparable form of a source_file: repo-relative posix when possible."""
    if not isinstance(sf, str) or not sf:
        return ""
    p = Path(sf)
    if p.is_absolute() and root is not None:
        try:
            return p.resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            pass
    return p.as_posix()


def _qualify_pipeline_datasets(nodes: list, edges: list, keep: bool, root: Path | None = None) -> None:
    by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
    # pipeline file (source_file) -> qualifying prefix ("silver" / "{catalog}.silver")
    prefix_by_file: dict[str, str] = {}
    for e in edges:
        if not isinstance(e, dict) or e.get("relation") != "pipeline_source":
            continue
        pipe, f = by_id.get(e.get("source")), by_id.get(e.get("target"))
        if not pipe or not f or pipe.get("type") != "dab_pipeline":
            continue
        schema = _norm_part(pipe.get("uc_schema_norm") or pipe.get("uc_schema"))
        sf = f.get("source_file")
        if not schema or not sf:
            continue
        prefix = schema
        catalog = _norm_part(pipe.get("uc_catalog_norm") or pipe.get("uc_catalog"))
        if keep and catalog:
            prefix = f"{catalog}.{schema}"
        prefix_by_file.setdefault(_sf_key(sf, root), prefix)
    if not prefix_by_file:
        return

    one_part = {nid for nid, n in by_id.items()
                if n.get("type") == TABLE_NODE_TYPE and "." not in str(n.get("label", ""))}
    if not one_part:
        return
    new_nodes: dict[str, dict] = {}
    moved_from: set[str] = set()
    for e in edges:
        if not isinstance(e, dict) or e.get("relation") not in _TABLE_RELATIONS:
            continue
        tid = e.get("target")
        if tid not in one_part:
            continue
        prefix = prefix_by_file.get(_sf_key(e.get("source_file", ""), root))
        if not prefix:
            continue
        old = by_id[tid]
        label = f"{prefix}.{old['label']}"
        nid = table_node_id(label)
        if nid not in by_id and nid not in new_nodes:
            node = {k: v for k, v in old.items()}
            node["id"], node["label"] = nid, label
            node["source_file"], node["source_location"] = "", ""
            node["uc_catalogs"] = list(old.get("uc_catalogs") or [])
            new_nodes[nid] = node
        target = by_id.get(nid) or new_nodes[nid]
        target["uc_pipeline_dataset"] = True
        if e.get("context") == "defines" and not target.get("source_file"):
            target["source_file"] = e.get("source_file", "")
            target["source_location"] = e.get("source_location", "")
            if old.get("uc_kind") not in (None, "unknown"):
                target["uc_kind"] = old["uc_kind"]
        e["target"] = nid
        moved_from.add(tid)
    nodes.extend(new_nodes.values())
    _drop_orphans(nodes, edges, moved_from)


def _norm_part(value) -> str | None:
    if not isinstance(value, str) or not value.strip() or "${" in value:
        return None
    return value.strip().strip("`").lower()


# ── 4. wheel tasks ───────────────────────────────────────────────────────────

_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".databricks",
              "graphify-out", ".tox", "build", "dist"}


def _find_pyprojects(root: Path) -> list[Path]:
    found: list[Path] = []
    stack = [root]
    while stack and len(found) < 50:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if p.is_dir():
                if p.name not in _SKIP_DIRS and not p.name.startswith("."):
                    stack.append(p)
            elif p.name == "pyproject.toml":
                found.append(p)
    return found


def _pkg_norm(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _read_entry_points(pyproject: Path) -> tuple[str | None, dict[str, str]]:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - py<3.11
        return None, {}
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None, {}
    project = data.get("project") or {}
    eps: dict[str, str] = {}
    for table in [project.get("scripts") or {}, project.get("gui-scripts") or {},
                  *((project.get("entry-points") or {}).values())]:
        if isinstance(table, dict):
            for k, v in table.items():
                if isinstance(v, str):
                    eps.setdefault(k, v)
    poetry = ((data.get("tool") or {}).get("poetry") or {})
    for k, v in (poetry.get("scripts") or {}).items():
        if isinstance(v, str):
            eps.setdefault(k, v)
    name = project.get("name") or poetry.get("name")
    return (name if isinstance(name, str) else None), eps


def _link_wheel_tasks(nodes: list, edges: list, root: Path | None) -> None:
    tasks = [n for n in nodes if isinstance(n, dict) and n.get("type") == "dab_task"
             and isinstance(n.get("dab_package_name"), str)]
    if not tasks:
        return
    if root is None:
        from graphify.extractors.base import active_scan_root

        root = active_scan_root() or Path.cwd()
    manifests: dict[str, tuple[Path, dict[str, str]]] = {}
    for pp in _find_pyprojects(root):
        name, eps = _read_entry_points(pp)
        if name:
            manifests.setdefault(_pkg_norm(name), (pp, eps))
    by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
    functions: dict[str, list[dict]] = {}
    for n in nodes:
        if isinstance(n, dict) and str(n.get("label", "")).endswith("()") and n.get("source_file", "").endswith(".py"):
            functions.setdefault(str(n["label"])[:-2].lstrip("."), []).append(n)
    from graphify.ids import make_id

    existing = {(e.get("source"), e.get("target"), e.get("relation")) for e in edges if isinstance(e, dict)}
    for task in tasks:
        pkg = task["dab_package_name"]
        pkg_id = make_id("pkg", pkg)
        hit = manifests.get(_pkg_norm(pkg))
        if pkg_id not in by_id and hit is not None:
            pp = hit[0]
            try:
                rel = pp.relative_to(root).as_posix()
            except ValueError:
                rel = pp.as_posix()
            node = {"id": pkg_id, "label": pkg, "type": "package", "file_type": "code",
                    "source_file": rel, "source_location": "L1"}
            nodes.append(node)
            by_id[pkg_id] = node
        ep = task.get("dab_entry_point")
        if not isinstance(ep, str) or not ep:
            continue
        target = (hit[1].get(ep) if hit else None) or f"{pkg}:{ep}"
        module, _, func = target.partition(":")
        func = func.split("[")[0].strip().split(".")[-1]
        module = module.strip()
        if not func:
            continue
        suffixes = (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py")
        candidates = [n for n in functions.get(func, [])
                      if any(("/" + n["source_file"]).endswith("/" + s) for s in suffixes)]
        if len(candidates) != 1:
            continue
        key = (task["id"], candidates[0]["id"], "runs_entry_point")
        if key in existing:
            continue
        existing.add(key)
        edges.append({"source": task["id"], "target": candidates[0]["id"],
                      "relation": "runs_entry_point", "confidence": "INFERRED",
                      "source_file": task.get("source_file", ""),
                      "source_location": task.get("source_location", ""), "weight": 1.0,
                      "entry_point": target})


# ── 5. edges ─────────────────────────────────────────────────────────────────

def _clean_edges(nodes: list, edges: list) -> None:
    table_ids = {n["id"] for n in nodes if isinstance(n, dict) and n.get("type") == TABLE_NODE_TYPE}
    seen: set = set()
    rw: dict[tuple, dict] = {}
    out: list = []
    for e in edges:
        if not isinstance(e, dict):
            out.append(e)
            continue
        key = (e.get("source"), e.get("target"), e.get("relation"), e.get("context"))
        if key in seen:
            continue
        seen.add(key)
        rel = e.get("relation")
        if rel in _TABLE_RELATIONS and e.get("target") in table_ids:
            pair = (e.get("source"), e.get("target"))
            other = rw.get(pair)
            if other is not None:
                if other.get("relation") != rel:
                    writer = other if other.get("relation") == "writes_to" else e
                    writer["also_reads"] = True
                    if writer is e:
                        out[out.index(other)] = e
                        rw[pair] = e
                continue
            rw[pair] = e
        out.append(e)
    edges[:] = out


# ── helpers ──────────────────────────────────────────────────────────────────

def _rename_tables(nodes: list, edges: list, remap: dict[str, str]) -> None:
    """Re-key table nodes ``old id -> new label``, merging into existing nodes."""
    if not remap:
        return
    by_id = {n["id"]: n for n in nodes if isinstance(n, dict) and "id" in n}
    id_map: dict[str, str] = {}
    for old_id, label in remap.items():
        new_id = table_node_id(label)
        id_map[old_id] = new_id
        old = by_id[old_id]
        if new_id in by_id:
            _merge_into(by_id[new_id], old)
        else:
            node = dict(old)
            node["id"], node["label"] = new_id, label
            nodes.append(node)
            by_id[new_id] = node
    nodes[:] = [n for n in nodes if not (isinstance(n, dict) and n.get("id") in id_map)]
    for e in edges:
        if isinstance(e, dict):
            if e.get("source") in id_map:
                e["source"] = id_map[e["source"]]
            if e.get("target") in id_map:
                e["target"] = id_map[e["target"]]


def _drop_orphans(nodes: list, edges: list, candidates: set[str]) -> None:
    if not candidates:
        return
    used = {e.get("source") for e in edges if isinstance(e, dict)} | {
        e.get("target") for e in edges if isinstance(e, dict)}
    nodes[:] = [n for n in nodes if not (isinstance(n, dict) and n.get("id") in candidates
                                          and n.get("id") not in used)]
