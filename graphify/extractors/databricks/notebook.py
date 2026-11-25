"""Databricks / Jupyter notebooks.

Handles

* **source-format notebooks**: ``.py`` / ``.r`` files whose first line is
  ``# Databricks notebook source``, ``.sql`` with ``-- Databricks notebook source``
  and ``.scala`` with ``// Databricks notebook source``. Cells are separated by
  ``<comment> COMMAND ----------``; a cell whose lines are ``<comment> MAGIC ...``
  and whose first line is ``%sql`` / ``%md`` / ``%run`` / ... is in that language.
* **``.ipynb``**: any Jupyter notebook (Databricks exports included). Outputs are
  never read beyond JSON parsing.

Line mapping
------------
Every cell is described by real 1-based line numbers. For source notebooks those
are the file's own lines. A ``.ipynb`` has no useful line numbers, so the cells
are laid out in a *virtual source*: each cell's source lines in order, followed by
one separator line. All ``source_location`` values of a ``.ipynb`` refer to that
virtual source, and nodes/edges also carry ``cell`` (the 0-based cell index in the
notebook, counting markdown cells).

Per-language *views* keep that line layout: the Python view holds the Python
cells' text on their own lines and blanks everything else, so the core Python
extractor (``extract_python(path, source=...)``) and the lineage scanner report
positions that are valid in the notebook. The SQL view does the same for SQL
cells, with ``;`` on each separator line so statements never run across cells.

``%run ./x`` becomes a ``runs_notebook`` edge to the target file node. When the
target is an importable ``.py`` module under the scan root, the ``%run`` line of
the Python view is replaced with ``from <module> import *`` (same line), which is
what ``%run`` means at runtime: the caller sees the target's definitions, so
calls into them resolve cross-file exactly like a star import.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from graphify.extractors.base import _make_id, active_scan_root
from graphify.extractors.databricks.paths import file_ref_id, resolve_workspace_path

_HEADER_TOKENS = {
    ".py": "#", ".r": "#", ".sql": "--", ".scala": "//",
}
_SUFFIX_LANGUAGE = {".py": "python", ".r": "r", ".sql": "sql", ".scala": "scala"}
_HEAD_BYTES = 256

# Magic commands that switch a whole cell's language, mapped to that language.
_CELL_MAGICS = {
    "%python": "python", "%py": "python", "%pyspark": "python",
    "%sql": "sql", "%scala": "scala", "%r": "r",
    "%md": "md", "%md-sandbox": "md", "%markdown": "md",
    "%run": "run", "%pip": "pip", "%conda": "pip", "%sh": "sh", "%fs": "fs",
}
# Jupyter cell magics (`%%x` on the first line) that keep the cell Python.
_PY_PASSTHROUGH_CELL_MAGICS = {"%%capture", "%%time", "%%timeit", "%%prun", "%%debug"}
_JUPYTER_CELL_MAGICS = {
    "%%sql": "sql", "%%sparksql": "sql", "%%bash": "sh", "%%sh": "sh", "%%script": "sh",
    "%%markdown": "md", "%%md": "md", "%%html": "html", "%%javascript": "js",
    "%%js": "js", "%%writefile": "file", "%%scala": "scala", "%%R": "r", "%%r": "r",
    "%%python": "python",
}
_WIDGET_KINDS = ("text", "dropdown", "combobox", "multiselect")
_SECRETISH_NAME_RX = re.compile(r"secret|token|passw|pwd|credential|api[_-]?key|private", re.IGNORECASE)
_SQL_WIDGET_RX = re.compile(
    r"\bCREATE\s+WIDGET\s+(TEXT|DROPDOWN|COMBOBOX|MULTISELECT)\s+`?(\w+)`?"
    r"(?:\s+DEFAULT\s+(?:'([^'\n]*)'|\"([^\"\n]*)\"))?",
    re.IGNORECASE,
)
_MD_HEADING_RX = re.compile(r"^\s{0,3}(#{1,3})\s+(.+?)\s*#*\s*$")
_MAX_MD_HEADINGS = 200


@dataclass
class Cell:
    """One notebook cell. ``text`` maps 1:1 onto lines ``start_line..end_line``
    (MAGIC prefixes removed, the magic command itself blanked, its arguments kept
    in ``magic_args``)."""

    index: int
    language: str
    start_line: int
    end_line: int
    text: str
    magic: str | None = None
    magic_args: str = ""
    magic_line: int | None = None  # real/virtual line holding the magic command


class NotebookError(ValueError):
    pass


# ── ownership ────────────────────────────────────────────────────────────────

def _source_header_token(path: Path) -> str | None:
    token = _HEADER_TOKENS.get(path.suffix.lower())
    if token is None:
        return None
    try:
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES)
    except OSError:
        return None
    first = head.lstrip(b"\xef\xbb\xbf").split(b"\n", 1)[0].strip()
    want = f"{token} Databricks notebook source".encode()
    return token if first == want else None


def owns(path: Path) -> bool:
    if path.suffix.lower() == ".ipynb":
        return True
    return _source_header_token(path) is not None


# ── cell splitting ───────────────────────────────────────────────────────────

def _classify_magic(first_line: str) -> tuple[str | None, str | None, str]:
    """(magic, language, args) for a cell's first line, or (None, None, "")."""
    stripped = first_line.strip()
    if not stripped.startswith("%"):
        return None, None, ""
    token, _, args = stripped.partition(" ")
    if token.startswith("%%"):
        if token in _PY_PASSTHROUGH_CELL_MAGICS:
            return token, "python", args.strip()
        return token, _JUPYTER_CELL_MAGICS.get(token, "magic"), args.strip()
    lang = _CELL_MAGICS.get(token.lower())
    if lang is None:
        return None, None, ""
    return token.lower(), lang, args.strip()


def _split_source(path: Path, token: str, text: str) -> tuple[list[Cell], str]:
    default = _SUFFIX_LANGUAGE[path.suffix.lower()]
    lines = text.split("\n")
    sep_rx = re.compile(r"^\s*" + re.escape(token) + r"\s*COMMAND\s*-{5,}\s*$")
    magic_rx = re.compile(r"^" + re.escape(token) + r" MAGIC(?: (.*)|\s*)$")
    cells: list[Cell] = []
    block: list[tuple[int, str]] = []

    def flush() -> None:
        if not block:
            return
        start = block[0][0]
        end = block[-1][0]
        content = [ln for _, ln in block]
        first_idx = next((i for i, ln in enumerate(content) if ln.strip()), None)
        magic = lang = None
        args = ""
        out = list(content)
        if first_idx is not None:
            m = magic_rx.match(content[first_idx])
            if m is not None:
                # A MAGIC cell: strip the prefix from every MAGIC line.
                out = []
                for ln in content:
                    mm = magic_rx.match(ln)
                    out.append((mm.group(1) or "") if mm else ln)
                magic, lang, args = _classify_magic(out[first_idx])
                if magic is not None and not magic.startswith("%%"):
                    # Blank the magic token, keep inline arguments in place.
                    lead = out[first_idx][: len(out[first_idx]) - len(out[first_idx].lstrip())]
                    rest = out[first_idx].strip()[len(magic):]
                    out[first_idx] = (lead + " " * len(magic) + rest) if lang in ("sql", "python", "scala", "r", "md") else ""
        cells.append(Cell(index=len(cells), language=lang or default, start_line=start,
                          end_line=end, text="\n".join(out), magic=magic, magic_args=args,
                          magic_line=start + first_idx if magic is not None else None))
        block.clear()

    for lineno, line in enumerate(lines, start=1):
        if lineno == 1:
            continue  # the "Databricks notebook source" header
        if sep_rx.match(line):
            flush()
            continue
        block.append((lineno, line))
    flush()
    return cells, default


def _ipynb_load(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise NotebookError(f"unreadable notebook: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("cells", []), list):
        raise NotebookError("not a Jupyter notebook (no cells list)")
    return data


def _ipynb_language(meta: dict) -> str:
    def norm(v: object) -> str | None:
        if not isinstance(v, str) or not v.strip():
            return None
        v = v.strip().lower()
        return {"pyspark": "python", "python3": "python", "spark": "scala", "ir": "r"}.get(v, v)

    db = meta.get("application/vnd.databricks.v1+notebook")
    for cand in (
        db.get("language") if isinstance(db, dict) else None,
        (meta.get("kernelspec") or {}).get("language") if isinstance(meta.get("kernelspec"), dict) else None,
        (meta.get("language_info") or {}).get("name") if isinstance(meta.get("language_info"), dict) else None,
    ):
        lang = norm(cand)
        if lang:
            return lang
    return "python"


def _cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(s for s in src if isinstance(s, str))
    return src if isinstance(src, str) else ""


def _split_ipynb(path: Path) -> tuple[list[Cell], str, int]:
    data = _ipynb_load(path)
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    default = _ipynb_language(meta)
    cells: list[Cell] = []
    line = 1
    for idx, raw in enumerate(data.get("cells") or []):
        if not isinstance(raw, dict):
            continue
        ctype = raw.get("cell_type")
        src_lines = _cell_source(raw).replace("\r\n", "\n").split("\n")
        if src_lines and src_lines[-1] == "" and len(src_lines) > 1:
            src_lines.pop()
        start, end = line, line + len(src_lines) - 1
        line = end + 2  # one virtual separator line after each cell
        if ctype == "markdown":
            cells.append(Cell(idx, "md", start, end, "\n".join(src_lines)))
            continue
        if ctype != "code":
            cells.append(Cell(idx, "raw", start, end, "\n".join(src_lines)))
            continue
        lang = default
        cmeta = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        dbc = cmeta.get("application/vnd.databricks.v1+cell")
        if isinstance(dbc, dict) and isinstance(dbc.get("language"), str) and dbc["language"].strip():
            lang = _ipynb_language({"kernelspec": {"language": dbc["language"]}})
        out = list(src_lines)
        first_idx = next((i for i, ln in enumerate(out) if ln.strip()), None)
        magic = None
        args = ""
        if first_idx is not None:
            magic, mlang, args = _classify_magic(out[first_idx])
            if magic is not None:
                lang = mlang or lang
                rest = out[first_idx].strip()[len(magic):]
                out[first_idx] = (" " * len(magic) + rest) if (
                    not magic.startswith("%%") and lang in ("sql", "md", "scala", "r")
                ) else ""
        cells.append(Cell(idx, lang, start, end, "\n".join(out), magic, args,
                          start + first_idx if magic is not None else None))
    return cells, default, max(line - 1, 1)


def split_cells(path: Path) -> list[Cell]:
    """The notebook's cells (see :class:`Cell`). Raises :class:`NotebookError`
    for an unreadable ``.ipynb``; returns ``[]`` for a non-notebook."""
    path = Path(path)
    if path.suffix.lower() == ".ipynb":
        return _split_ipynb(path)[0]
    token = _source_header_token(path)
    if token is None:
        return []
    text = path.read_bytes().decode("utf-8", errors="replace").lstrip("﻿")
    return _split_source(path, token, text.replace("\r\n", "\n"))[0]


# ── views ────────────────────────────────────────────────────────────────────

def _view(cells: list[Cell], total_lines: int, language: str, *, separator: str = "") -> list[str]:
    """``total_lines`` lines with ``language`` cells' text on their own lines."""
    lines = [""] * total_lines
    starts = set()
    for cell in cells:
        starts.add(cell.start_line)
        if cell.language != language:
            continue
        for offset, text in enumerate(cell.text.split("\n")):
            n = cell.start_line + offset
            if 1 <= n <= total_lines:
                lines[n - 1] = text
    if separator:
        # Put the separator on the blank line before each cell (the COMMAND line
        # or the virtual separator), so statements never span cells.
        for s in starts:
            if 2 <= s <= total_lines + 1 and not lines[s - 2].strip():
                lines[s - 2] = separator
    return lines


def _python_line_ok(line: str) -> str:
    """Blank IPython line magics and shell escapes, which are not Python."""
    stripped = line.lstrip()
    if stripped.startswith(("%", "!")):
        return ""
    return line


# ── helpers ──────────────────────────────────────────────────────────────────

def _bundle_root(path: Path) -> Path | None:
    scan_root = active_scan_root()
    directory = path.parent
    for _ in range(40):
        for name in ("databricks.yml", "databricks.yaml"):
            if (directory / name).is_file():
                return directory
        if scan_root is not None and directory.resolve() == scan_root.resolve():
            break
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


def _run_target(args: str) -> str:
    token = args.strip().split(None, 1)[0] if args.strip() else ""
    return token.strip("\"'")


def _module_name(target: Path) -> str | None:
    root = active_scan_root()
    if root is None or target.suffix != ".py":
        return None
    try:
        rel = target.resolve().relative_to(Path(root).resolve()).with_suffix("")
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if not parts or parts[-1] == "__init__" or not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def _lineage():
    try:
        from graphify.extractors.databricks import lineage  # type: ignore
    except ImportError:
        return None
    return lineage


def _file_node(path: Path, language: str, fmt: str) -> dict:
    return {"id": _make_id(str(path)), "label": path.name, "file_type": "code",
            "source_file": str(path), "source_location": "L1",
            "notebook": True, "notebook_language": language, "notebook_format": fmt}


def _merge(into: dict, other: dict) -> None:
    ids = {n.get("id") for n in into["nodes"]}
    for node in other.get("nodes", []):
        if node.get("id") not in ids:
            ids.add(node.get("id"))
            into["nodes"].append(node)
    into["edges"].extend(other.get("edges", []))
    if other.get("raw_calls"):
        into.setdefault("raw_calls", []).extend(other["raw_calls"])


def _edge(src: str, tgt: str, relation: str, path: Path, line: int, **extra) -> dict:
    edge = {"source": src, "target": tgt, "relation": relation, "confidence": "EXTRACTED",
            "source_file": str(path), "source_location": f"L{line}", "weight": 1.0}
    edge.update(extra)
    return edge


def _string_literal(node, source: bytes) -> str | None:
    """Text of a plain (non-interpolated) Python string literal node."""
    if node is None or node.type != "string":
        return None
    parts = []
    for child in node.children:
        if child.type == "interpolation":
            return None
        if child.type == "string_content":
            parts.append(source[child.start_byte:child.end_byte].decode("utf-8", "replace"))
    return "".join(parts)


def _enclosing_function_nid(node, source: bytes, nodes: list[dict], path: Path) -> str | None:
    cur = node.parent
    while cur is not None:
        if cur.type == "function_definition":
            name_node = cur.child_by_field_name("name")
            if name_node is None:
                return None
            name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
            loc = f"L{cur.start_point[0] + 1}"
            for n in nodes:
                label = str(n.get("label", ""))
                if (n.get("source_location") == loc and n.get("source_file") == str(path)
                        and label.endswith("()") and label[:-2].split(".")[-1] == name):
                    return n["id"]
            return None
        cur = cur.parent
    return None


def _scan_python_calls(path: Path, view: str, result: dict, file_nid: str) -> list[dict]:
    """``dbutils.notebook.run`` edges and widget declarations from the Python view."""
    widgets: list[dict] = []
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        parser = Parser(Language(tspython.language()))
        source = view.encode("utf-8")
        tree = parser.parse(source)
    except Exception:
        return widgets
    from graphify.security import redact_secrets

    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node.type != "call":
            continue
        fn = node.child_by_field_name("function")
        if fn is None:
            continue
        fname = re.sub(r"\s+", "", source[fn.start_byte:fn.end_byte].decode("utf-8", "replace"))
        args_node = node.child_by_field_name("arguments")
        args = [a for a in (args_node.named_children if args_node else []) if a.type != "comment"]
        line = node.start_point[0] + 1
        if fname == "dbutils.notebook.run" and args:
            ref = _string_literal(args[0], source)
            if not ref:
                continue
            target = resolve_workspace_path(ref, path.parent, bundle_root=_bundle_root(path))
            if target is None:
                continue
            extra = {"context": "dbutils.notebook.run"}
            params = next((a for a in args[1:] if a.type == "dictionary"), None)
            if params is not None:
                keys = []
                for pair in params.named_children:
                    k = _string_literal(pair.child_by_field_name("key"), source) if pair.type == "pair" else None
                    if k:
                        keys.append(k)
                if keys:
                    extra["params"] = keys
            src = _enclosing_function_nid(node, source, result["nodes"], path) or file_nid
            result["edges"].append(_edge(src, file_ref_id(target), "runs_notebook", path, line, **extra))
        elif fname.startswith("dbutils.widgets.") and fname.rsplit(".", 1)[1] in _WIDGET_KINDS and args:
            name = _string_literal(args[0], source)
            if not name:
                continue
            default = _string_literal(args[1], source) if len(args) > 1 else None
            if default is not None and (_SECRETISH_NAME_RX.search(name) or redact_secrets(default) != default):
                default = None
            widgets.append({"name": name, "default": default, "kind": fname.rsplit(".", 1)[1]})
    return widgets


def _sql_widgets(sql_view: str) -> list[dict]:
    from graphify.security import redact_secrets

    out = []
    for m in _SQL_WIDGET_RX.finditer(sql_view):
        default = m.group(3) if m.group(3) is not None else m.group(4)
        if default is not None and (_SECRETISH_NAME_RX.search(m.group(2)) or redact_secrets(default) != default):
            default = None
        out.append({"name": m.group(2), "default": default, "kind": m.group(1).lower()})
    return out


def _md_headings(path: Path, cells: list[Cell], file_nid: str, result: dict) -> None:
    count = 0
    for cell in cells:
        if cell.language != "md":
            continue
        in_fence = False
        for offset, text in enumerate(cell.text.split("\n")):
            if text.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            m = None if in_fence else _MD_HEADING_RX.match(text)
            if not m or count >= _MAX_MD_HEADINGS:
                continue
            title = m.group(2).strip()
            if not title:
                continue
            count += 1
            line = cell.start_line + offset
            nid = _make_id(str(path), "section", title)
            if any(n.get("id") == nid for n in result["nodes"]):
                continue
            result["nodes"].append({"id": nid, "label": title, "file_type": "code",
                                    "source_file": str(path), "source_location": f"L{line}",
                                    "notebook_section": True, "heading_level": len(m.group(1))})
            result["edges"].append(_edge(file_nid, nid, "contains", path, line))


def _remap_core_sql_tables(result: dict, sql_nodes: list[dict], sql_edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fold the core SQL extractor's per-file table nodes onto ``uc_table`` nodes
    already in ``result`` (minted by the lineage scan), redirecting their edges."""
    from graphify.extractors.databricks.config import get_config
    from graphify.extractors.databricks.names import (
        TABLE_NODE_TYPE, parse_table_name, table_key, table_node_id,
    )

    uc_ids = {n["id"] for n in result["nodes"] if n.get("type") == TABLE_NODE_TYPE}
    if not uc_ids:
        return sql_nodes, sql_edges
    keep_catalog = get_config().catalog_identity == "keep"
    remap: dict[str, str] = {}
    for n in sql_nodes:
        label = str(n.get("label", ""))
        if label.endswith("()"):
            continue  # a routine, not a table
        tn = parse_table_name(label)
        if tn is None:
            continue
        uc = table_node_id(table_key(tn, keep_catalog=keep_catalog))
        if uc in uc_ids:
            remap[n["id"]] = uc
    nodes = [n for n in sql_nodes if n["id"] not in remap]
    edges = []
    for e in sql_edges:
        if e["relation"] == "contains" and e["target"] in remap:
            continue
        e = dict(e)
        e["source"] = remap.get(e["source"], e["source"])
        e["target"] = remap.get(e["target"], e["target"])
        edges.append(e)
    return nodes, edges


# ── extract ──────────────────────────────────────────────────────────────────

def extract(path: Path) -> dict:
    path = Path(path)
    is_ipynb = path.suffix.lower() == ".ipynb"
    try:
        if is_ipynb:
            cells, default, total = _split_ipynb(path)
            raw_text = None
        else:
            token = _source_header_token(path)
            if token is None:
                return {"nodes": [], "edges": [], "error": "not a Databricks source notebook"}
            raw_text = path.read_bytes().decode("utf-8", errors="replace").lstrip("﻿").replace("\r\n", "\n")
            cells, default = _split_source(path, token, raw_text)
            total = len(raw_text.split("\n"))
    except NotebookError as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": f"{type(exc).__name__}: {exc}"}

    root = active_scan_root()
    file_nid = _make_id(str(path))
    result: dict = {"nodes": [], "edges": []}
    bundle_root = _bundle_root(path)

    # %run targets (resolved once, used for edges and the Python view).
    runs: list[tuple[Cell, Path]] = []
    for cell in cells:
        if cell.language == "run":
            target = resolve_workspace_path(_run_target(cell.magic_args), path.parent, bundle_root=bundle_root)
            if target is not None:
                runs.append((cell, target))

    # ── Python ──
    py_lines = [_python_line_ok(ln) for ln in _view(cells, total, "python")]
    for cell, target in runs:
        module = _module_name(target)
        if module and cell.magic_line and 1 <= cell.magic_line <= total:
            py_lines[cell.magic_line - 1] = f"from {module} import *"
    py_view = "\n".join(py_lines)
    has_python = any(c.language == "python" and c.text.strip() for c in cells) or any(
        _module_name(t) for _, t in runs)
    if has_python:
        from graphify.extract import extract_python

        injected = any(_module_name(t) for _, t in runs)
        magic_python = any(c.language == "python" and c.magic for c in cells)
        if raw_text is not None and default == "python" and not injected and not magic_python:
            # A .py source notebook is valid Python as written (every non-Python
            # cell is `# MAGIC` comments), so parse the file itself.
            py_result = extract_python(path, root=root)
        else:
            py_result = extract_python(path, root=root, source=py_view.encode("utf-8"))
        if "error" not in py_result:
            _merge(result, py_result)
            for key, value in py_result.items():
                if key not in ("nodes", "edges", "raw_calls") and key not in result:
                    result[key] = value
    elif raw_text is not None and default in ("scala", "r"):
        from graphify import extract as core

        base = (core.extract_scala if default == "scala" else core.extract_r)(path)
        if "error" not in base:
            _merge(result, base)

    if not any(n.get("id") == file_nid for n in result["nodes"]):
        result["nodes"].insert(0, _file_node(path, default, "ipynb" if is_ipynb else "source"))
    file_node = next(n for n in result["nodes"] if n.get("id") == file_nid)
    file_node.update({"notebook": True, "notebook_language": default,
                      "notebook_format": "ipynb" if is_ipynb else "source"})

    # ── %run / dbutils.notebook.run / widgets ──
    for cell, target in runs:
        result["edges"].append(_edge(file_nid, file_ref_id(target), "runs_notebook", path,
                                     cell.magic_line or cell.start_line, context="%run"))
    widgets = _scan_python_calls(path, py_view, result, file_nid) if has_python else []

    # ── SQL ──
    has_sql = any(c.language == "sql" and c.text.strip() for c in cells)
    sql_view = "\n".join(_view(cells, total, "sql", separator=";")) if has_sql else ""
    if has_sql:
        widgets.extend(_sql_widgets(sql_view))
    if widgets:
        file_node["db_widgets"] = widgets

    # ── lineage (tables) ──
    lineage = _lineage()
    if lineage is not None:
        refs = []
        if has_python and hasattr(lineage, "scan_python"):
            refs.extend(lineage.scan_python(py_view))
        if has_sql and hasattr(lineage, "scan_sql"):
            refs.extend(lineage.scan_sql(sql_view))
        if refs:
            if hasattr(lineage, "attach"):
                lineage.attach(refs, result, path)
            else:
                from graphify.extractors.databricks.names import emit_table_refs
                emit_table_refs(refs, file_nid=file_nid, file_path=str(path),
                                nodes=result["nodes"], edges=result["edges"])

    # Routines/tables the core SQL grammar finds in SQL cells; table nodes are
    # folded onto the lineage scan's uc_table nodes so each table is one node.
    if has_sql:
        from graphify.extract import extract_sql

        sql_result = extract_sql(path, content=sql_view)
        if "error" not in sql_result:
            nodes, edges = _remap_core_sql_tables(
                result,
                [n for n in sql_result.get("nodes", []) if n.get("id") != file_nid],
                sql_result.get("edges", []),
            )
            _merge(result, {"nodes": nodes, "edges": edges})

    _md_headings(path, cells, file_nid, result)

    if is_ipynb:
        _annotate_cells(result, cells, path)
    return result


def _annotate_cells(result: dict, cells: list[Cell], path: Path) -> None:
    import bisect

    starts = [c.start_line for c in cells]

    def cell_of(loc: object) -> int | None:
        if not isinstance(loc, str) or not loc.startswith("L"):
            return None
        try:
            line = int(loc[1:].split("-")[0])
        except ValueError:
            return None
        i = bisect.bisect_right(starts, line) - 1
        if i < 0 or line > cells[i].end_line:
            return None
        return cells[i].index

    for item in list(result["nodes"]) + list(result["edges"]):
        if item.get("source_file") != str(path):
            continue
        idx = cell_of(item.get("source_location"))
        if idx is not None:
            item["cell"] = idx
