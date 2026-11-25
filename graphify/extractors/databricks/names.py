"""Unity Catalog table names: parsing, normalization, and the shared node contract.

Every Databricks extractor (notebooks, PySpark/DLT lineage, Databricks SQL,
Asset Bundles) emits tables through :func:`emit_table_refs` so that the same
table seen from any file lands on ONE node.

Identity rule
-------------
A table's *key* is ``schema.table`` (lower-cased, quotes/backticks removed),
with the catalog dropped by default: in a bundle-driven repo the catalog is
almost always an environment switch (``${var.catalog}``, a ``catalog`` widget,
``dev``/``prod``), so ``{catalog}.silver.orders``, ``dev.silver.orders`` and
``silver.orders`` are the same logical table. The catalogs actually written
are kept on the node as ``uc_catalogs``. ``databricks.catalog_identity=keep``
(see :mod:`.config`) keeps the catalog in the key for multi-catalog lakehouses.

Placeholders are normalized to ``{name}``: ``${var.catalog}`` -> ``{catalog}``,
``${bundle.target}`` -> ``{target}``, f-string ``{cfg.catalog}`` -> ``{catalog}``.
A one-part name (``orders``, a DLT dataset) keys as just ``orders``; the corpus
pass may qualify it with its pipeline's schema.

Node contract
-------------
``id``     ``make_id("uc_table", key)``
``type``   ``"uc_table"`` (exempt from per-file id disambiguation, so copies
           from different files merge into one node)
``label``  the key as displayed (``silver.orders``)
``source_file`` the file that *defines* it (CREATE / saveAsTable / @dlt.table)
           when this file defines it, else ``""``; ``origin_file`` always
           names the file that mentioned it.
Edges: ``reads_from`` / ``writes_to`` from the reading/writing symbol (or the
file node) to the table; ``defines`` is emitted as ``writes_to`` with
``context="defines"``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from graphify.ids import make_id

Relation = Literal["reads_from", "writes_to", "defines"]
TableKind = Literal["table", "view", "streaming_table", "materialized_view", "unknown"]

TABLE_NODE_TYPE = "uc_table"

# Node types that name one corpus-wide entity and therefore merge across files
# instead of being split by per-file id disambiguation (see
# extractors/resolution.py). Bundle entities are minted by ``bundle.py``.
MERGEABLE_NODE_TYPES = frozenset({
    TABLE_NODE_TYPE, "uc_schema", "uc_volume",
    "dab_bundle", "dab_target", "dab_variable", "dab_job", "dab_task", "dab_pipeline",
    "dab_resource",
})

# ${var.catalog} / ${bundle.target} / ${workspace.current_user.short_name}
_BUNDLE_VAR_RX = re.compile(r"\$\{\s*(?:var\.|bundle\.|workspace\.)?([\w.]+?)\s*\}")
# f-string / str.format placeholders: {catalog}, {cfg.catalog}, {self.schema}, {catalog!r}
_FMT_VAR_RX = re.compile(r"\{\s*([A-Za-z_][\w.\[\]'\"]*)\s*(?:![rsa])?\s*(?::[^{}]*)?\}")
# A Spark/DLT "live" virtual schema prefix: LIVE.orders, live.orders
_LIVE_PREFIX_RX = re.compile(r"^\s*live\s*\.\s*", re.IGNORECASE)
# One name part: `quoted`, or identifier chars with optional {placeholder}s
# embedded (orders_{env}). A full name is one to three parts joined by dots.
_PART = r"(?:`(?:[^`]|``)+`|(?:[\w$-]|\{[a-z_][\w]*\})+)"
_PART_RX = re.compile(_PART)
_FULL_NAME_RX = re.compile(_PART + r"(?:\s*\.\s*" + _PART + r"){0,2}")

# Schemas that are catalog metadata, not data, and never become nodes.
IGNORED_SCHEMAS = frozenset({"information_schema"})


@dataclass(frozen=True)
class TableName:
    """A parsed table reference. Parts are already normalized (lower-cased,
    unquoted, placeholders as ``{name}``); ``catalog``/``schema`` may be None."""

    name: str
    schema: str | None = None
    catalog: str | None = None
    live: bool = False  # written as LIVE.x / live.x (DLT pipeline-local dataset)
    raw: str = ""

    @property
    def has_placeholder(self) -> bool:
        return any(p and p.startswith("{") for p in (self.catalog, self.schema, self.name))


@dataclass
class TableRef:
    """One table mention found by a scanner. ``line`` is 1-based in the file."""

    raw: str
    relation: Relation
    line: int
    kind: TableKind = "unknown"
    via: str = ""  # e.g. "spark.table", "saveAsTable", "spark.sql", "dlt.table", "%sql"
    # The node id of the symbol doing the read/write (function, DLT dataset fn,
    # task); None means the file node.
    source_nid: str | None = None
    extra: dict = field(default_factory=dict)


def normalize_placeholders(text: str) -> str:
    """Rewrite bundle ``${...}`` and f-string ``{...}`` placeholders to ``{leaf}``.

    The leaf is the last dotted/attribute segment, lower-cased, so
    ``${var.catalog}``, ``{catalog}``, ``{cfg.catalog}`` and ``{CATALOG}`` agree.
    """

    def _leaf(expr: str) -> str:
        expr = re.sub(r"\[\s*['\"]?([\w]+)['\"]?\s*\]", r".\1", expr)  # cfg["catalog"] -> cfg.catalog
        return "{" + expr.strip().split(".")[-1].lower() + "}"

    text = _BUNDLE_VAR_RX.sub(lambda m: _leaf(m.group(1)), text)
    return _FMT_VAR_RX.sub(lambda m: _leaf(m.group(1)), text)


def _unquote(part: str) -> str:
    part = part.strip()
    if len(part) >= 2 and part[0] == part[-1] == "`":
        part = part[1:-1].replace("``", "`")
    elif len(part) >= 2 and part[0] == part[-1] and part[0] in "\"'":
        part = part[1:-1]
    return part.strip().lower()


def parse_table_name(raw: str) -> TableName | None:
    """Parse ``catalog.schema.table`` (1-3 parts) or return None if it is not a
    plausible table name (empty, a path, a SQL fragment, more than 3 parts)."""
    if not raw:
        return None
    text = normalize_placeholders(raw.strip().strip("\"'"))
    live = bool(_LIVE_PREFIX_RX.match(text))
    if live:
        text = _LIVE_PREFIX_RX.sub("", text, count=1)
    text = text.strip()
    if not _FULL_NAME_RX.fullmatch(text):
        return None  # a path, a SQL fragment, more than three parts, ...
    parts = [m.group(0) for m in _PART_RX.finditer(text)]
    norm = [_unquote(p) for p in parts]
    if any(not p for p in norm):
        return None
    # delta.`/mnt/raw/x`, parquet.`s3://...`: a path read through a format
    # prefix, not a catalog object.
    if any("/" in p or ":" in p for p in norm):
        return None
    while len(norm) < 3:
        norm.insert(0, None)  # type: ignore[arg-type]
    catalog, schema, name = norm
    if schema in IGNORED_SCHEMAS:
        return None
    return TableName(name=name, schema=schema, catalog=catalog, live=live, raw=raw)


def table_key(tn: TableName, *, keep_catalog: bool = False) -> str:
    """The canonical identity string for a table (see module docstring)."""
    parts = [tn.schema, tn.name]
    if keep_catalog and tn.catalog:
        parts.insert(0, tn.catalog)
    return ".".join(p for p in parts if p)


def uc_node_id(kind: str, key: str) -> str:
    """Collision-safe id for a Unity Catalog entity (table, schema, volume).

    ``make_id`` folds ``.`` and ``_`` together, so ``a_b.c`` and ``a.b_c`` (or
    ``silver.orders`` and ``silver._orders``) would share an id. These nodes are
    exempt from per-file disambiguation, so a collision would silently merge
    unrelated tables. A short digest of the exact canonical key keeps distinct
    keys apart while the readable prefix stays greppable.
    """
    import hashlib

    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
    return f"{make_id(kind, key)}_{digest}"


def table_node_id(key: str) -> str:
    return uc_node_id(TABLE_NODE_TYPE, key)


def emit_table_refs(
    refs: Iterable[TableRef],
    *,
    file_nid: str,
    file_path: str,
    nodes: list[dict],
    edges: list[dict],
    keep_catalog: bool | None = None,
) -> list[str]:
    """Append ``uc_table`` nodes and reads/writes edges for ``refs``.

    Idempotent per node id within ``nodes`` (merges ``uc_catalogs`` and
    upgrades ``source_file`` when this file defines the table). Returns the
    table node ids touched, in order. ``keep_catalog=None`` reads the active
    project config.
    """
    from graphify.extractors.databricks.config import get_config

    cfg = get_config()
    if keep_catalog is None:
        keep_catalog = cfg.catalog_identity == "keep"
    by_id = {n["id"]: n for n in nodes if n.get("type") == TABLE_NODE_TYPE}
    seen_edges = {(e["source"], e["target"], e["relation"]) for e in edges}
    touched: list[str] = []
    for ref in refs:
        tn = parse_table_name(ref.raw)
        if tn is None or (tn.schema and tn.schema in cfg.ignore_schemas):
            continue
        key = table_key(tn, keep_catalog=keep_catalog)
        nid = table_node_id(key)
        if not nid:
            continue
        node = by_id.get(nid)
        if node is None:
            node = {
                "id": nid, "label": key, "type": TABLE_NODE_TYPE, "file_type": "code",
                "source_file": "", "source_location": "", "origin_file": file_path,
                "uc_catalogs": [], "uc_kind": ref.kind,
            }
            nodes.append(node)
            by_id[nid] = node
        if tn.catalog and tn.catalog not in node["uc_catalogs"]:
            node["uc_catalogs"].append(tn.catalog)
        if tn.live:
            node["uc_live"] = True
        if ref.kind != "unknown" and node.get("uc_kind") in (None, "unknown"):
            node["uc_kind"] = ref.kind
        if ref.relation == "defines" and not node.get("source_file"):
            node["source_file"] = file_path
            node["source_location"] = f"L{ref.line}"
        relation = "writes_to" if ref.relation == "defines" else ref.relation
        src = ref.source_nid or file_nid
        if (src, nid, relation) not in seen_edges:
            seen_edges.add((src, nid, relation))
            edge = {
                "source": src, "target": nid, "relation": relation,
                "confidence": "EXTRACTED", "source_file": file_path,
                "source_location": f"L{ref.line}", "weight": 1.0,
            }
            if ref.relation == "defines":
                edge["context"] = "defines"
            if ref.via:
                edge["via"] = ref.via
            edges.append(edge)
        touched.append(nid)
    return touched
