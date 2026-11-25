"""Unity Catalog table lineage from Python (PySpark, DLT/Lakeflow) and Databricks SQL.

Two pure scanners produce :class:`~graphify.extractors.databricks.names.TableRef`
lists, and :func:`attach` turns them into ``uc_table`` nodes and
``reads_from`` / ``writes_to`` edges on an extractor result:

``scan_python(source)``
    tree-sitter-python walk over every call and decorator. Recognized:

    * reads  - ``spark.table``, ``spark.read[...].table``, ``spark.readStream.table``,
      ``DeltaTable.forName`` (a write when the handle is later ``.merge``/``.update``/
      ``.delete``-d), ``dlt.read``/``dlt.read_stream`` and their Lakeflow
      ``pyspark.pipelines`` equivalents under any import alias.
    * writes - ``saveAsTable``, ``insertInto``, ``toTable``, ``writeTo``.
    * defines - ``@dlt.table`` / ``@dlt.view`` / ``@dp.materialized_view`` / ...
      (``name=`` or the function name), ``create_streaming_table``; flows
      (``apply_changes``, ``create_auto_cdc_flow``, ``@append_flow``) write
      their ``target`` and read their ``source``.
    * ``spark.sql(...)`` strings, handed to :func:`scan_sql`.
    * Team wrappers named in ``databricks.readers`` / ``databricks.writers``.

    Table-name arguments are evaluated statically: literals, f-strings,
    ``.format``, ``%`` and ``+``, and same-file string constants. A variable bound
    to ``dbutils.widgets.get("catalog")`` (or a task value, ``spark.conf.get``,
    ``os.environ``/``getenv``) becomes the placeholder ``{catalog}``; an
    interpolated expression that cannot be evaluated becomes ``{leaf}`` (its last
    attribute/key). A name whose final (table) part is only placeholders, or
    any expression that cannot be evaluated at all, is skipped - never guessed.

``scan_sql(text)``
    Regex scan of comment-masked Databricks SQL: CREATE [OR REPLACE|OR REFRESH]
    [TEMPORARY] [LIVE|STREAMING] TABLE / [MATERIALIZED] VIEW, INSERT INTO/OVERWRITE,
    MERGE INTO ... USING, UPDATE, DELETE FROM, COPY INTO, TRUNCATE, FROM/JOIN
    (incl. ``STREAM(x)``, ``LIVE.x``, comma joins), ``IDENTIFIER(...)``, with
    ``USE CATALOG``/``USE SCHEMA`` applied to later short names and CTE names
    excluded. Placeholders ``${x}``, ``{{x}}``, ``:x`` and ``$x`` become ``{x}``.

``.sql`` files (decision)
    For a Databricks-flavored ``.sql`` file the core SQL extractor still runs (it
    owns routines and the file node), then every core *table* node whose name
    :func:`scan_sql` also found is folded onto the matching ``uc_table`` node:
    its edges are retargeted and the per-file node (or sourceless stub) is
    dropped. Without the fold the same table would exist twice - once as the
    core's per-file ``<stem>_<name>`` node and once as the corpus-wide
    ``uc_table`` - and SQL lineage would never meet Python lineage. Core table
    nodes the scanner did not see are left untouched.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from graphify.extractors.databricks.names import (
    TABLE_NODE_TYPE,
    TableRef,
    emit_table_refs,
    parse_table_name,
    table_key,
    table_node_id,
)

_HEAD_BYTES = 256 * 1024
_NOTEBOOK_HEADERS = ("# databricks notebook source", "-- databricks notebook source")

# ── ownership ────────────────────────────────────────────────────────────────

_PY_MARKER_RX = re.compile(r"\b(?:spark|pyspark|dlt|pipelines|dbutils|DeltaTable)\b")

# Databricks-only SQL constructs. Bare three-part names count only when the file
# carries no T-SQL marker (T-SQL also writes database.schema.table).
_SQL_DATABRICKS_RX = re.compile(
    r"\bSTREAMING\s+TABLE\b|\bMATERIALIZED\s+VIEW\b|\bLIVE\s*\.|\bSTREAM\s*\(|\bUSE\s+CATALOG\b"
    r"|\bIDENTIFIER\s*\(|\$\{|\{\{|\bCOPY\s+INTO\b|\bread_files\s*\(|\bUSING\s+DELTA\b"
    r"|\bCREATE\s+OR\s+REFRESH\b|`[^`\n]+`\s*\.\s*`",
    re.IGNORECASE,
)
_SQL_THREE_PART_RX = re.compile(
    r"\b(?:FROM|JOIN|INTO|TABLE|VIEW|UPDATE)\s+[\w`{}$]+\s*\.\s*[\w`{}$]+\s*\.\s*[\w`{}$]+",
    re.IGNORECASE,
)
_SQL_TSQL_RX = re.compile(
    r"\bdbo\b|\[\w+\]|^\s*GO\s*$|@\w+|\bNVARCHAR\b|\bBEGIN\s+TRAN", re.IGNORECASE | re.MULTILINE
)


def _head(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            return fh.read(_HEAD_BYTES).decode("utf-8", errors="replace")
    except OSError:
        return ""


def owns(path: Path) -> bool:
    """Whether this module should extract ``path`` (see module docstring)."""
    from graphify.extractors.databricks.config import get_config

    cfg = get_config()
    if not cfg.enabled:
        return False
    suffix = path.suffix.lower()
    if suffix not in (".py", ".sql"):
        return False
    head = _head(path)
    if head.lstrip("﻿").lstrip().lower().startswith(_NOTEBOOK_HEADERS):
        return False  # a source-format notebook: the notebook module's job
    if suffix == ".py":
        if cfg.mode == "on":
            return True
        if _PY_MARKER_RX.search(head):
            return True
        names = cfg.readers | cfg.writers
        return bool(names) and any(re.search(r"\b" + re.escape(n) + r"\b", head, re.IGNORECASE) for n in names)
    masked = _mask_sql(head)
    if _SQL_DATABRICKS_RX.search(masked):
        return True
    return bool(_SQL_THREE_PART_RX.search(masked)) and not _SQL_TSQL_RX.search(masked)


# ── SQL ──────────────────────────────────────────────────────────────────────


def _mask_sql(text: str) -> str:
    """``text`` with comments and string-literal *contents* blanked to spaces.

    Length and newlines are preserved, so offsets and line numbers hold. Quote
    characters stay, so ``'...'`` remains visibly a string; backticks are
    identifiers and are kept verbatim.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "-" and text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif c == "/" and text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "'\"":
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == c:
                    if j + 1 < n and text[j + 1] == c:  # '' escape
                        j += 2
                        continue
                    break
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        elif c == "`":
            j = text.find("`", i + 1)
            i = n if j < 0 else j + 1
        else:
            i += 1
    return "".join(out)


_PLACEHOLDER_SUBS = (
    (re.compile(r"\$\{\s*(?:var\.|bundle\.|workspace\.)?([\w.]+?)\s*\}"), lambda m: "{" + m.group(1).split(".")[-1].lower() + "}"),
    (re.compile(r"\{\{\s*([\w.]+)\s*\}\}"), lambda m: "{" + m.group(1).split(".")[-1].lower() + "}"),
    (re.compile(r"(?<![:\w]):([A-Za-z_]\w*)\b"), lambda m: "{" + m.group(1).lower() + "}"),
    (re.compile(r"(?<![\w$])\$([A-Za-z_]\w*)\b"), lambda m: "{" + m.group(1).lower() + "}"),
)

# IDENTIFIER('cat.sch.t'), IDENTIFIER(:cat || '.silver.t'), IDENTIFIER({cat} || '.x')
_IDENTIFIER_RX = re.compile(r"\bIDENTIFIER\s*\(([^()]*)\)", re.IGNORECASE)


def _identifier_value(arg: str) -> str | None:
    pieces = []
    for piece in arg.split("||"):
        piece = piece.strip()
        if len(piece) >= 2 and piece[0] == piece[-1] and piece[0] in "'\"":
            pieces.append(piece[1:-1])
        elif re.fullmatch(r"\{[\w]+\}", piece):
            pieces.append(piece)
        else:
            return None
    return "".join(pieces) or None


def _prepare_sql(text: str) -> str:
    """Placeholders -> ``{x}`` and ``IDENTIFIER(...)`` -> its name, keeping
    every newline in place so line numbers survive."""
    for rx, repl in _PLACEHOLDER_SUBS:
        text = rx.sub(repl, text)

    def _ident(m: re.Match) -> str:
        value = _identifier_value(m.group(1))
        newlines = m.group(0).count("\n")
        if value is None:
            return "IDENTIFIER_UNRESOLVED" + "\n" * newlines
        return "`" + "`.`".join(value.split(".")) + "`" + "\n" * newlines

    return _IDENTIFIER_RX.sub(_ident, text)


_SQL_PART = r"(?:`(?:[^`]|``)+`|(?:[\w$]|\{[a-z_]\w*\})+)"
_SQL_NAME = _SQL_PART + r"(?:\s*\.\s*" + _SQL_PART + r"){0,2}"
_NAME_END = r"(?![\w$`{]|\s*\.)"

_CREATE_RX = re.compile(
    r"\bCREATE\s+(?:OR\s+(?:REPLACE|REFRESH)\s+)?"
    r"(?P<temp>(?:GLOBAL\s+)?(?:TEMPORARY|TEMP)\s+)?"
    r"(?P<mod>LIVE\s+|STREAMING\s+)?"
    r"(?P<obj>TABLE|MATERIALIZED\s+VIEW|VIEW)\s+"
    r"(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>" + _SQL_NAME + r")" + _NAME_END,
    re.IGNORECASE,
)
_REPLACE_TABLE_RX = re.compile(
    r"(?<!OR\s)\bREPLACE\s+TABLE\s+(?P<name>" + _SQL_NAME + r")" + _NAME_END, re.IGNORECASE
)
_WRITE_RX = re.compile(
    r"\b(?P<verb>INSERT\s+(?:INTO|OVERWRITE)(?:\s+TABLE)?|MERGE\s+INTO|UPDATE|DELETE\s+FROM"
    r"|COPY\s+INTO|TRUNCATE\s+TABLE)\s+(?P<name>" + _SQL_NAME + r")" + _NAME_END,
    re.IGNORECASE,
)
_READ_RX = re.compile(
    r"\b(?P<kw>FROM|JOIN)\s+(?:STREAM\s*\(\s*(?P<sname>" + _SQL_NAME + r")\s*\)"
    r"|STREAM\s+(?P<sname2>" + _SQL_NAME + r")" + _NAME_END +
    r"|(?P<name>" + _SQL_NAME + r")" + _NAME_END + r")",
    re.IGNORECASE,
)
_MERGE_USING_RX = re.compile(
    r"\bUSING\s+(?:STREAM\s*\(\s*(?P<sname>" + _SQL_NAME + r")\s*\)|(?P<name>" + _SQL_NAME + r")" + _NAME_END + r")",
    re.IGNORECASE,
)
_COMMA_RX = re.compile(r"\s*,\s*(?P<name>" + _SQL_NAME + r")" + _NAME_END)
_ALIAS_RX = re.compile(r"\s+(?:AS\s+)?([A-Za-z_]\w*)", re.IGNORECASE)
_CTE_RX = re.compile(
    r"(?:\bWITH\s+(?:RECURSIVE\s+)?|,\s*)(`[^`]+`|[\w$]+)\s*(?:\([^()]*\))?\s+AS\s*(?:NOT\s+)?(?:MATERIALIZED\s+)?\(",
    re.IGNORECASE,
)
_USE_RX = re.compile(
    r"\bUSE\s+(?:(?P<what>CATALOG|SCHEMA|DATABASE)\s+)?(?P<name>" + _SQL_NAME + r")" + _NAME_END,
    re.IGNORECASE,
)
_NOT_TABLE_PRECEDING_RX = re.compile(
    r"(?:\b(?:EXTRACT|TRIM|SUBSTRING|OVERLAY|POSITION)\s*\([^()]*|\bDISTINCT\s+)$", re.IGNORECASE
)
_SQL_NON_TABLES = frozenset({
    "select", "where", "set", "values", "lateral", "null", "true", "false", "with",
    "as", "on", "using", "not", "exists", "inner", "left", "right", "full", "cross",
    "outer", "join", "group", "order", "having", "union", "except", "intersect",
    "from", "table", "stream", "identifier_unresolved", "unnest", "explode", "dual",
    "only", "first", "last", "the",
})


def _short(name: str) -> str:
    return re.sub(r"[`\s]", "", name).lower()


def _qualify(name: str, catalog: str | None, schema: str | None) -> str:
    """Apply USE CATALOG/SCHEMA to a one- or two-part name."""
    if re.match(r"\s*live\s*\.", name, re.IGNORECASE):
        return name
    parts = re.findall(_SQL_PART, name)
    if len(parts) == 1 and schema:
        return f"{schema}.{name}" if not catalog else f"{catalog}.{schema}.{name}"
    if len(parts) == 2 and catalog:
        return f"{catalog}.{name}"
    return name


def _line_at(text: str, pos: int, base: int) -> int:
    return base + text.count("\n", 0, pos)


def scan_sql(
    text: str,
    *,
    default_schema: str | None = None,
    default_catalog: str | None = None,
    line_offset: int = 0,
    via: str = "sql",
) -> list[TableRef]:
    """Table refs in Databricks SQL ``text`` (lines are ``line_offset`` + 1-based)."""
    prepared = _prepare_sql(text)
    masked = _mask_sql(prepared)
    base = line_offset + 1
    refs: list[TableRef] = []
    catalog, schema = default_catalog, default_schema

    # Statements, split on top-level `;` of the masked text.
    starts = [0] + [m.end() for m in re.finditer(";", masked)]
    for idx, start in enumerate(starts):
        end = starts[idx + 1] - 1 if idx + 1 < len(starts) else len(masked)
        stmt = masked[start:end]
        if not stmt.strip():
            continue
        use = _USE_RX.match(stmt.lstrip())
        if use:
            what = (use.group("what") or "").upper()
            parts = [p.strip("`").lower() for p in re.findall(_SQL_PART, use.group("name"))]
            if what == "CATALOG":
                catalog = parts[-1]
            elif len(parts) >= 2:
                catalog, schema = parts[-2], parts[-1]
            else:
                schema = parts[-1]
            continue
        ctes = {_short(m.group(1)) for m in _CTE_RX.finditer(stmt)}
        aliases: set[str] = set()
        claimed: list[tuple[int, int]] = []

        def add(name: str, relation: str, pos: int, kind: str = "unknown", v: str = via) -> None:
            short = _short(name)
            if not short or short in _SQL_NON_TABLES:
                return
            if "." not in short and (short in ctes or short in aliases):
                return
            refs.append(TableRef(
                raw=_qualify(name.strip(), catalog, schema), relation=relation,  # type: ignore[arg-type]
                line=_line_at(masked, start + pos, base), kind=kind, via=v,  # type: ignore[arg-type]
            ))

        for m in _CREATE_RX.finditer(stmt):
            obj = re.sub(r"\s+", " ", m.group("obj").upper())
            mod = (m.group("mod") or "").strip().upper()
            if obj == "MATERIALIZED VIEW":
                kind = "materialized_view"
            elif obj == "VIEW":
                kind = "view"
            elif mod == "STREAMING":
                kind = "streaming_table"
            else:
                kind = "table"
            add(m.group("name"), "defines", m.start("name"), kind)
            claimed.append(m.span("name"))
        for m in _REPLACE_TABLE_RX.finditer(stmt):
            if not any(s <= m.start("name") < e for s, e in claimed):
                add(m.group("name"), "defines", m.start("name"), "table")
                claimed.append(m.span("name"))
        is_merge = False
        for m in _WRITE_RX.finditer(stmt):
            verb = m.group("verb").upper()
            if verb.startswith("UPDATE") and re.search(r"(?:\bON|\bFOR|,)\s*$", stmt[:m.start()], re.IGNORECASE):
                continue  # ON UPDATE / FOR UPDATE: a clause, not a statement
            if verb.startswith("DELETE") and re.search(r"\bON\s*$", stmt[:m.start()], re.IGNORECASE):
                continue
            is_merge = is_merge or verb.startswith("MERGE")
            add(m.group("name"), "writes_to", m.start("name"))
            claimed.append(m.span("name"))
            am = _ALIAS_RX.match(stmt, m.end("name"))
            if am and am.group(1).lower() not in _SQL_NON_TABLES:
                aliases.add(am.group(1).lower())
        if is_merge:
            for m in _MERGE_USING_RX.finditer(stmt):
                name = m.group("sname") or m.group("name")
                if name and not stmt[m.end():].lstrip().startswith("("):
                    add(name, "reads_from", m.start("sname") if m.group("sname") else m.start("name"))
        for m in _READ_RX.finditer(stmt):
            if any(s <= m.start() < e for s, e in claimed):
                continue
            if m.group("kw").upper() == "FROM" and re.search(r"\bDELETE\s*$", stmt[:m.start()], re.IGNORECASE):
                continue
            if _NOT_TABLE_PRECEDING_RX.search(stmt[max(0, m.start() - 200):m.start()]):
                continue
            name = m.group("sname") or m.group("sname2") or m.group("name")
            name_start = m.start("sname") if m.group("sname") else (
                m.start("sname2") if m.group("sname2") else m.start("name"))
            after = stmt[m.end():].lstrip()
            if (m.group("name") or m.group("sname2")) and after.startswith("("):
                continue  # table-valued function: read_files(...), range(...)
            add(name, "reads_from", name_start)
            pos = m.end()
            am = _ALIAS_RX.match(stmt, pos)
            if am and am.group(1).lower() not in _SQL_NON_TABLES:
                aliases.add(am.group(1).lower())
                pos = am.end()
            if m.group("kw").upper() == "FROM":
                while True:
                    cm = _COMMA_RX.match(stmt, pos)
                    if not cm or stmt[cm.end():].lstrip().startswith("("):
                        break
                    add(cm.group("name"), "reads_from", cm.start("name"))
                    pos = cm.end()
                    am = _ALIAS_RX.match(stmt, pos)
                    if am and am.group(1).lower() not in _SQL_NON_TABLES:
                        aliases.add(am.group(1).lower())
                        pos = am.end()
    refs.sort(key=lambda r: r.line)
    return refs


# ── Python ───────────────────────────────────────────────────────────────────

_PY_PARSER = None


def _parser():
    global _PY_PARSER
    if _PY_PARSER is None:
        import tree_sitter_python as tsp
        from tree_sitter import Language, Parser

        _PY_PARSER = Parser(Language(tsp.language()))
    return _PY_PARSER


_DLT_MODULES = {"dlt", "pyspark.pipelines"}
_DLT_DECORATOR_KINDS = {
    "table": "table", "create_table": "table", "streaming_table": "streaming_table",
    "materialized_view": "materialized_view", "view": "view", "temporary_view": "view",
    "create_view": "view",
}
_DLT_READS = {"read", "read_stream"}
_DLT_STREAMING_DEFS = {"create_streaming_table", "create_target_table", "create_streaming_live_table"}
_DLT_FLOWS = {
    "apply_changes", "create_auto_cdc_flow", "apply_changes_from_snapshot",
    "create_auto_cdc_from_snapshot_flow",
}
_TABLE_KWARGS = ("table", "table_name", "name", "target", "full_name", "tablename", "table_fqn")
_PLACEHOLDER_CALLS = re.compile(
    r"^(?:dbutils\.widgets\.(?:get|getArgument)|getArgument|dbutils\.jobs\.taskValues\.get"
    r"|(?:\w+\.)*spark\.conf\.get|spark\.conf\.get|os\.environ\.get|os\.getenv)$"
)


class _Unresolved(Exception):
    pass


def _txt(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _dotted(node, src: bytes) -> str:
    """``a.b.c`` for an identifier/attribute chain, with call parens dropped
    (``spark.read.format("x").table`` -> ``spark.read.format.table``)."""
    parts: list[str] = []
    while node is not None:
        if node.type == "identifier":
            parts.append(_txt(node, src))
            break
        if node.type == "attribute":
            attr = node.child_by_field_name("attribute")
            parts.append(_txt(attr, src) if attr is not None else "")
            node = node.child_by_field_name("object")
        elif node.type == "call":
            node = node.child_by_field_name("function")
        else:
            parts.append("?")
            break
    return ".".join(reversed(parts))


def _leaf_of(node, src: bytes) -> str | None:
    """Placeholder leaf for an interpolated expression: last attribute name,
    subscript key, or identifier."""
    if node.type == "identifier":
        return _txt(node, src).lower()
    if node.type == "attribute":
        attr = node.child_by_field_name("attribute")
        return _txt(attr, src).lower() if attr is not None else None
    if node.type == "subscript":
        sub = node.child_by_field_name("subscript")
        if sub is not None and sub.type == "string":
            return _plain_string(sub, src).lower() if _plain_string(sub, src) else None
        if sub is not None and sub.type == "integer":
            base = _leaf_of(node.child_by_field_name("value"), src) or "arg"
            return f"{base}_{_txt(sub, src)}"
        return None
    if node.type == "call":
        return None
    return None


def _plain_string(node, src: bytes) -> str | None:
    """Content of a non-interpolated string literal, else None."""
    if node.type != "string":
        return None
    out = []
    for c in node.children:
        if c.type == "interpolation":
            return None
        if c.type == "string_content":
            out.append(_txt(c, src))
        if c.type == "string_start" and "f" in _txt(c, src).lower():
            pass
    return "".join(out)


def _call_key_arg(call, src: bytes) -> str | None:
    """First string argument of a call like ``dbutils.widgets.get("catalog")``
    (or its ``key=``/``name=`` keyword)."""
    args = call.child_by_field_name("arguments")
    if args is None:
        return None
    for a in args.named_children:
        if a.type == "keyword_argument":
            k = a.child_by_field_name("name")
            v = a.child_by_field_name("value")
            if k is not None and _txt(k, src) in ("key", "name") and v is not None:
                s = _plain_string(v, src)
                if s:
                    return s
    for a in args.named_children:
        if a.type == "string":
            return _plain_string(a, src)
    return None


class _Scope:
    def __init__(self, parent: "_Scope | None" = None):
        self.parent = parent
        self.values: dict[str, list] = {}  # name -> [value nodes]

    def lookup(self, name: str):
        s: _Scope | None = self
        while s is not None:
            if name in s.values:
                return s, s.values[name]
            s = s.parent
        return None, None


class _PyScanner:
    def __init__(self, source: str, cfg) -> None:
        self.src = source.encode("utf-8")
        self.cfg = cfg
        self.tree = _parser().parse(self.src)
        self.refs: list[TableRef] = []
        self.dlt_aliases: set[str] = set()
        self.dlt_names: dict[str, str] = {}  # local name -> dlt function name
        self.scopes: dict[int, _Scope] = {}
        self.module_scope = _Scope()
        self._resolving: set[int] = set()

    # -- setup ---------------------------------------------------------------
    def _collect_imports(self) -> None:
        for node in self._iter(self.tree.root_node):
            if node.type == "import_statement":
                for c in node.named_children:
                    if c.type == "dotted_name" and _txt(c, self.src) in _DLT_MODULES:
                        self.dlt_aliases.add(_txt(c, self.src))
                    elif c.type == "aliased_import":
                        name = c.child_by_field_name("name")
                        alias = c.child_by_field_name("alias")
                        if name is not None and alias is not None and _txt(name, self.src) in _DLT_MODULES:
                            self.dlt_aliases.add(_txt(alias, self.src))
            elif node.type == "import_from_statement":
                mod = node.child_by_field_name("module_name")
                modname = _txt(mod, self.src) if mod is not None else ""
                for c in node.named_children:
                    if c is mod:
                        continue
                    if c.type == "dotted_name":
                        local = orig = _txt(c, self.src)
                    elif c.type == "aliased_import":
                        n, a = c.child_by_field_name("name"), c.child_by_field_name("alias")
                        if n is None or a is None:
                            continue
                        orig, local = _txt(n, self.src), _txt(a, self.src)
                    else:
                        continue
                    if modname == "pyspark" and orig == "pipelines":
                        self.dlt_aliases.add(local)
                    elif modname in _DLT_MODULES:
                        self.dlt_names[local] = orig
        if not self.dlt_aliases and re.search(rb"^\s*import\s+dlt\b", self.src, re.MULTILINE):
            self.dlt_aliases.add("dlt")

    def _iter(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))

    def _scope_for(self, node) -> _Scope:
        """The scope of the innermost function containing ``node``."""
        p = node.parent
        while p is not None:
            if p.type == "function_definition":
                return self.scopes.setdefault(p.id, _Scope(self.module_scope))
            p = p.parent
        return self.module_scope

    def _collect_bindings(self) -> None:
        for node in self._iter(self.tree.root_node):
            if node.type != "assignment":
                continue
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is None or right is None or left.type != "identifier":
                continue
            # `x: str = ...` keeps right; `x = y = ...` handled by nested assignment.
            scope = self._scope_for(node)
            scope.values.setdefault(_txt(left, self.src), []).append(right)

    # -- evaluation ------------------------------------------------------------
    def _eval_identifier(self, node) -> str:
        name = _txt(node, self.src)
        _scope, values = self._scope_for(node).lookup(name)
        if not values:
            raise _Unresolved(name)
        results = set()
        for v in values:
            if v.id in self._resolving:
                raise _Unresolved(name)
            self._resolving.add(v.id)
            try:
                results.add(self._eval(v))
            finally:
                self._resolving.discard(v.id)
        if len(results) != 1:
            raise _Unresolved(name)
        return results.pop()

    def _eval(self, node) -> str:
        t = node.type
        if t == "string":
            out = []
            for c in node.children:
                if c.type == "string_content":
                    out.append(_txt(c, self.src))
                elif c.type == "interpolation":
                    expr = next((x for x in c.named_children if x.type not in ("type_conversion", "format_specifier")), None)
                    if expr is None:
                        raise _Unresolved("empty interpolation")
                    out.append(self._eval_interp(expr))
            return "".join(out)
        if t == "concatenated_string":
            return "".join(self._eval(c) for c in node.named_children)
        if t == "parenthesized_expression" and node.named_children:
            return self._eval(node.named_children[0])
        if t == "identifier":
            return self._eval_identifier(node)
        if t == "binary_operator":
            op = node.child_by_field_name("operator")
            left, right = node.child_by_field_name("left"), node.child_by_field_name("right")
            opt = _txt(op, self.src) if op is not None else ""
            if opt == "+":
                return self._eval(left) + self._eval(right)
            if opt == "%":
                fmt = self._eval(left)
                items = right.named_children if right.type == "tuple" else [right]
                vals = [self._eval_interp(i) for i in items]
                it = iter(vals)
                try:
                    return re.sub(r"%[sdr]", lambda _m: next(it), fmt)
                except StopIteration:
                    raise _Unresolved("% arity")
            raise _Unresolved(opt)
        if t == "call":
            fn = node.child_by_field_name("function")
            dotted = _dotted(fn, self.src) if fn is not None else ""
            if _PLACEHOLDER_CALLS.match(dotted):
                key = _call_key_arg(node, self.src)
                if key:
                    return "{" + re.split(r"[.\s]", key.strip())[-1].lower() + "}"
                raise _Unresolved(dotted)
            if fn is not None and fn.type == "attribute":
                attr = fn.child_by_field_name("attribute")
                obj = fn.child_by_field_name("object")
                if attr is not None and _txt(attr, self.src) == "format" and obj is not None:
                    return self._eval_format(self._eval(obj), node)
                if attr is not None and _txt(attr, self.src) in ("lower", "upper", "strip") and obj is not None:
                    val = self._eval(obj)
                    return getattr(val, _txt(attr, self.src))() if "{" not in val else val
            raise _Unresolved(dotted)
        if t == "subscript":
            value = node.child_by_field_name("value")
            if value is not None and _dotted(value, self.src) == "sys.argv":
                leaf = _leaf_of(node, self.src)
                return "{" + (leaf or "argv") + "}"
            raise _Unresolved("subscript")
        raise _Unresolved(t)

    def _eval_interp(self, expr) -> str:
        """An f-string/format/% slot: evaluated when possible, else ``{leaf}``."""
        try:
            return self._eval(expr)
        except _Unresolved:
            leaf = _leaf_of(expr, self.src)
            if leaf is None:
                raise
            return "{" + leaf + "}"

    def _eval_format(self, fmt: str, call) -> str:
        args = call.child_by_field_name("arguments")
        positional: list[str] = []
        named: dict[str, str] = {}
        if args is not None:
            for a in args.named_children:
                if a.type == "keyword_argument":
                    k, v = a.child_by_field_name("name"), a.child_by_field_name("value")
                    if k is not None and v is not None:
                        named[_txt(k, self.src)] = self._eval_interp(v)
                elif a.type in ("list_splat", "dictionary_splat"):
                    continue
                else:
                    positional.append(self._eval_interp(a))
        it = iter(positional)

        def repl(m: re.Match) -> str:
            key = m.group(1).split("!")[0].split(":")[0].strip()
            if key == "":
                try:
                    return next(it)
                except StopIteration:
                    raise _Unresolved("format arity")
            if key.isdigit():
                idx = int(key)
                if idx < len(positional):
                    return positional[idx]
                raise _Unresolved("format index")
            return named.get(key, "{" + key.split(".")[-1].lower() + "}")

        return re.sub(r"\{([^{}]*)\}", repl, fmt)

    def _name_arg(self, call, *, index: int = 0, kwargs: Iterable[str] = _TABLE_KWARGS) -> tuple[str | None, int]:
        """(evaluated table-name argument, its line) or (None, line)."""
        args = call.child_by_field_name("arguments")
        line = call.start_point[0] + 1
        if args is None:
            return None, line
        kwset = set(kwargs)
        positional = []
        for a in args.named_children:
            if a.type == "keyword_argument":
                k, v = a.child_by_field_name("name"), a.child_by_field_name("value")
                if k is not None and v is not None and _txt(k, self.src) in kwset:
                    try:
                        return self._eval(v), v.start_point[0] + 1
                    except _Unresolved:
                        return None, line
            elif a.type not in ("list_splat", "dictionary_splat", "comment"):
                positional.append(a)
        if len(positional) > index:
            a = positional[index]
            try:
                return self._eval(a), a.start_point[0] + 1
            except _Unresolved:
                return None, line
        return None, line

    @staticmethod
    def _usable(name: str | None) -> bool:
        if not name:
            return False
        last = re.split(r"\.", name.strip())[-1]
        return bool(re.sub(r"\{[^{}]*\}|[`\"'\s]", "", last)) and parse_table_name(name) is not None

    # -- enclosing symbol ---------------------------------------------------
    @staticmethod
    def _enclosing(node, src: bytes):
        p = node.parent
        while p is not None:
            if p.type == "function_definition":
                name = p.child_by_field_name("name")
                return (_txt(name, src) if name is not None else "", p.start_point[0] + 1)
            p = p.parent
        return None

    def _add(self, node, name: str | None, relation: str, line: int, *, kind: str = "unknown",
             via: str = "", enclosing=None) -> None:
        if not self._usable(name):
            return
        ref = TableRef(raw=name, relation=relation, line=line, kind=kind, via=via)  # type: ignore[arg-type]
        enc = enclosing if enclosing is not None else self._enclosing(node, self.src)
        if enc is not None:
            ref.extra["enclosing"] = enc
        self.refs.append(ref)

    # -- main walk -------------------------------------------------------------
    def scan(self) -> list[TableRef]:
        self._collect_imports()
        self._collect_bindings()
        handles: list[tuple[str, object, object, TableRef | None]] = []
        for node in self._iter(self.tree.root_node):
            if node.type == "decorated_definition":
                self._decorators(node)
            elif node.type == "call":
                self._call(node, handles)
        self._resolve_delta_handles(handles)
        self.refs.sort(key=lambda r: r.line)
        return self.refs

    def _dlt_func(self, fn_node) -> str | None:
        """The dlt/pipelines function a call/decorator target names, if any."""
        if fn_node is None:
            return None
        if fn_node.type == "identifier":
            return self.dlt_names.get(_txt(fn_node, self.src))
        if fn_node.type == "attribute":
            obj = fn_node.child_by_field_name("object")
            attr = fn_node.child_by_field_name("attribute")
            if obj is not None and attr is not None and _dotted(obj, self.src) in self.dlt_aliases:
                return _txt(attr, self.src)
        return None

    def _decorators(self, node) -> None:
        func = node.child_by_field_name("definition")
        if func is None or func.type != "function_definition":
            return
        fname_node = func.child_by_field_name("name")
        fname = _txt(fname_node, self.src) if fname_node is not None else ""
        enc = (fname, func.start_point[0] + 1)
        for dec in node.children:
            if dec.type != "decorator":
                continue
            expr = next((c for c in dec.named_children), None)
            if expr is None:
                continue
            call = expr if expr.type == "call" else None
            target = expr.child_by_field_name("function") if call is not None else expr
            dfn = self._dlt_func(target)
            if dfn is None:
                continue
            line = dec.start_point[0] + 1
            if dfn in _DLT_DECORATOR_KINDS:
                name = None
                if call is not None:
                    name, _ = self._name_arg(call, index=99, kwargs=("name",))
                self._add(dec, name or fname, "defines", line, kind=_DLT_DECORATOR_KINDS[dfn],
                          via=f"dlt.{dfn}", enclosing=enc)
            elif dfn == "append_flow" and call is not None:
                name, _ = self._name_arg(call, index=99, kwargs=("target",))
                self._add(dec, name, "writes_to", line, via="dlt.append_flow", enclosing=enc)

    def _call(self, call, handles) -> None:
        fn = call.child_by_field_name("function")
        if fn is None:
            return
        # Decorator calls are handled by _decorators.
        if call.parent is not None and call.parent.type == "decorator":
            return
        line = call.start_point[0] + 1
        dfn = self._dlt_func(fn)
        if dfn is not None:
            if dfn in _DLT_READS:
                name, ln = self._name_arg(call)
                self._add(call, name, "reads_from", ln, via=f"dlt.{dfn}")
            elif dfn in _DLT_STREAMING_DEFS:
                name, ln = self._name_arg(call, kwargs=("name",))
                self._add(call, name, "defines", ln, kind="streaming_table", via=f"dlt.{dfn}")
            elif dfn in _DLT_FLOWS:
                tgt, ln = self._name_arg(call, kwargs=("target",))
                self._add(call, tgt, "writes_to", ln, via=f"dlt.{dfn}")
                src_name, ln2 = self._name_arg(call, index=1, kwargs=("source",))
                self._add(call, src_name, "reads_from", ln2, via=f"dlt.{dfn}")
            return
        dotted = _dotted(fn, self.src)
        segs = dotted.split(".")
        method = segs[-1]
        recv = segs[:-1]
        low_method = method.lower()
        cfg = self.cfg
        if low_method in cfg.readers or low_method in cfg.writers:
            name, ln = self._name_arg(call)
            rel = "writes_to" if low_method in cfg.writers else "reads_from"
            self._add(call, name, rel, ln, via=method)
            return
        if not recv:
            return
        if method == "table" and (
            recv[-1] in ("spark", "read", "readStream") or "read" in recv or "readStream" in recv
        ):
            name, ln = self._name_arg(call, kwargs=("tableName",))
            self._add(call, name, "reads_from", ln, via="spark.table" if recv[-1] == "spark" else "read.table")
        elif method in ("saveAsTable", "insertInto", "toTable", "writeTo"):
            name, ln = self._name_arg(call, kwargs=("name", "tableName", "table"))
            # saveAsTable / toTable create the table when it is missing, so they
            # define it (the first definer supplies the node's source_file);
            # insertInto needs an existing table and writeTo defers to .create().
            relation = "defines" if method in ("saveAsTable", "toTable") else "writes_to"
            self._add(call, name, relation, ln, via=method)
        elif method == "sql" and recv[-1] in ("spark", "sqlContext", "sql_context", "session"):
            self._spark_sql(call)
        elif method == "forName" and recv[-1] == "DeltaTable":
            name, ln = self._name_arg(call, index=1, kwargs=("tableOrViewName",))
            if not self._usable(name):
                return
            ref_index = len(self.refs)
            self._add(call, name, "reads_from", ln, via="DeltaTable.forName")
            ref = self.refs[ref_index] if len(self.refs) > ref_index else None
            handles.append((self._handle_var(call), call, self._func_of(call), ref))

    def _spark_sql(self, call) -> None:
        args = call.child_by_field_name("arguments")
        if args is None or not args.named_children:
            return
        arg = args.named_children[0]
        if arg.type == "keyword_argument":
            v = arg.child_by_field_name("value")
            if v is None:
                return
            arg = v
        try:
            text = self._eval(arg)
        except _Unresolved:
            return
        # A single literal keeps its own line map; anything assembled maps every
        # ref to the call's line. Triple-quoted literals start after the quote.
        if arg.type == "string":
            start = next((c for c in arg.children if c.type == "string_start"), None)
            offset = (start.end_point[0] if start is not None else arg.start_point[0])
            single_literal = True
        else:
            offset, single_literal = call.start_point[0], False
        enc = self._enclosing(call, self.src)
        for ref in scan_sql(text, line_offset=offset, via="spark.sql"):
            if not single_literal:
                ref.line = call.start_point[0] + 1
            if enc is not None:
                ref.extra["enclosing"] = enc
            if self._usable(ref.raw):
                self.refs.append(ref)

    def _handle_var(self, call) -> str | None:
        p = call.parent
        if p is not None and p.type == "assignment":
            left = p.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                return _txt(left, self.src)
        return None

    @staticmethod
    def _func_of(node):
        p = node.parent
        while p is not None:
            if p.type == "function_definition":
                return p.id
            p = p.parent
        return None

    def _resolve_delta_handles(self, handles) -> None:
        """A DeltaTable handle that is merged/updated/deleted is a write."""
        mutators = {"merge", "update", "delete", "updateExpr", "upsert"}
        for var, call, func_id, ref in handles:
            if ref is None:
                continue
            written = False
            # Chained: DeltaTable.forName(...).alias("t").merge(...)
            p = call.parent
            while p is not None and p.type in ("attribute", "call"):
                if p.type == "attribute":
                    attr = p.child_by_field_name("attribute")
                    if attr is not None and _txt(attr, self.src) in mutators:
                        written = True
                        break
                p = p.parent
            if not written and var:
                for node in self._iter(self.tree.root_node):
                    if node.type != "call" or self._func_of(node) != func_id:
                        continue
                    fn = node.child_by_field_name("function")
                    if fn is None or fn.type != "attribute":
                        continue
                    segs = _dotted(fn, self.src).split(".")
                    if segs[0] == var and any(s in mutators for s in segs[1:]):
                        written = True
                        break
            if written:
                ref.relation = "writes_to"


def scan_python(source: str, *, cfg=None) -> list[TableRef]:
    """Table refs in Python ``source`` (see module docstring). ``cfg`` defaults
    to the active project's :class:`DatabricksConfig`."""
    if cfg is None:
        from graphify.extractors.databricks.config import get_config

        cfg = get_config()
    try:
        return _PyScanner(source, cfg).scan()
    except RecursionError:
        return []


# ── attaching to an extractor result ───────────────────────────────────────


def _symbol_index(result: dict, path: Path) -> dict[tuple[str, int], str]:
    """(function name, def line) -> node id for function/method nodes of ``path``."""
    index: dict[tuple[str, int], str] = {}
    sp = str(path)
    for n in result.get("nodes", []):
        if n.get("source_file") != sp:
            continue
        loc = str(n.get("source_location") or "")
        m = re.match(r"L(\d+)", loc)
        label = str(n.get("label") or "")
        lm = re.fullmatch(r"\.?([A-Za-z_]\w*)\(\)", label)
        if m and lm:
            index.setdefault((lm.group(1), int(m.group(1))), n["id"])
    return index


def attach(refs: list[TableRef], result: dict, path: Path) -> list[str]:
    """Attribute ``refs`` to their enclosing function nodes in ``result`` and
    emit ``uc_table`` nodes/edges into it. Returns the touched table ids."""
    from graphify.extractors.base import _make_id

    if not refs:
        return []
    nodes = result.setdefault("nodes", [])
    edges = result.setdefault("edges", [])
    file_nid = _make_id(str(path))
    if not any(n.get("id") == file_nid for n in nodes):
        file_nid = next(
            (n["id"] for n in nodes if n.get("source_file") == str(path) and n.get("label") == path.name),
            file_nid,
        )
    index = _symbol_index(result, path)
    for ref in refs:
        enc = ref.extra.get("enclosing")
        if enc is not None and ref.source_nid is None:
            ref.source_nid = index.get((enc[0], enc[1]))
    return emit_table_refs(refs, file_nid=file_nid, file_path=str(path), nodes=nodes, edges=edges)


def _drop_dlt_decorator_stubs(result: dict, refs: list[TableRef]) -> None:
    """Remove the sourceless ``table``/``view`` stubs the Python extractor mints
    for ``@dlt.table``-style decorators: the decorator is now a real
    ``defines`` edge, and the stub would be a god node linking every dataset."""
    if not any(r.via.startswith("dlt.") for r in refs):
        return
    stub_labels = set(_DLT_DECORATOR_KINDS) | {"append_flow", "expect", "expect_or_drop", "expect_or_fail",
                                               "expect_all", "expect_all_or_drop", "expect_all_or_fail"}
    stubs = {
        n["id"] for n in result.get("nodes", [])
        if not n.get("source_file") and n.get("label") in stub_labels and n.get("type") != TABLE_NODE_TYPE
    }
    if not stubs:
        return
    # Only drop a stub nothing else in the file points at for another reason.
    result["nodes"] = [n for n in result["nodes"] if n["id"] not in stubs]
    result["edges"] = [e for e in result.get("edges", []) if e.get("source") not in stubs and e.get("target") not in stubs]


# ── extractor entry point ─────────────────────────────────────────────────────


def _fold_core_sql_tables(result: dict, refs: list[TableRef], path: Path) -> None:
    """Fold the core SQL extractor's table nodes onto ``uc_table`` ids (see the
    module docstring, ".sql files")."""
    from graphify.extractors.databricks.config import get_config

    keep = get_config().catalog_identity == "keep"
    wanted: set[str] = set()
    for r in refs:
        tn = parse_table_name(r.raw)
        if tn is not None:
            wanted.add(table_key(tn, keep_catalog=keep))
    file_nid = None
    from graphify.extractors.base import _make_id

    file_nid = _make_id(str(path))
    remap: dict[str, str] = {}
    for n in result.get("nodes", []):
        if n.get("id") == file_nid or n.get("type") == TABLE_NODE_TYPE:
            continue
        label = str(n.get("label") or "")
        if label.endswith(")"):
            continue  # a routine
        tn = parse_table_name(label)
        if tn is None:
            continue
        key = table_key(tn, keep_catalog=keep)
        if key in wanted:
            remap[n["id"]] = table_node_id(key)
    if not remap:
        return
    result["nodes"] = [n for n in result["nodes"] if n["id"] not in remap]
    kept: list[dict] = []
    seen: set[tuple] = set()
    for e in result.get("edges", []):
        if e.get("relation") == "contains" and e.get("target") in remap:
            continue  # the uc_table is linked by reads/writes, not containment
        e = dict(e)
        e["source"] = remap.get(e["source"], e["source"])
        e["target"] = remap.get(e["target"], e["target"])
        if e["source"] == e["target"]:
            continue
        sig = (e["source"], e["target"], e.get("relation"))
        if sig in seen:
            continue
        seen.add(sig)
        kept.append(e)
    result["edges"] = kept


def extract(path: Path) -> dict:
    """Core Python/SQL extraction of ``path`` plus Unity Catalog lineage."""
    from graphify.extractors.base import _read_source_bytes, active_scan_root

    suffix = path.suffix.lower()
    if suffix == ".py":
        from graphify.extract import extract_python

        result = extract_python(path, root=active_scan_root())
        if "error" in result:
            return result
        try:
            source = _read_source_bytes(path, warn=False).decode("utf-8", errors="replace")
        except OSError:
            return result
        refs = scan_python(source)
        _drop_dlt_decorator_stubs(result, refs)
        attach(refs, result, path)
        return result
    from graphify.extract import extract_sql

    result = extract_sql(path)
    try:
        text = _read_source_bytes(path, warn=False).decode("utf-8", errors="replace")
    except OSError:
        return result
    refs = scan_sql(text)
    if "error" in result and not result.get("nodes"):
        # No grammar: lineage still stands on its own with a file node.
        from graphify.extractors.base import _make_id

        result = {"nodes": [{"id": _make_id(str(path)), "label": path.name, "file_type": "code",
                             "source_file": str(path), "source_location": None}], "edges": []}
    _fold_core_sql_tables(result, refs, path)
    attach(refs, result, path)
    return result
