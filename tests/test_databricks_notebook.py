"""Databricks / Jupyter notebook extraction (graphify/extractors/databricks/notebook.py)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from graphify.extractors.base import set_active_scan_root
from graphify.extractors.databricks import notebook as nb
from graphify.validate import validate_extraction

HAS_LINEAGE = importlib.util.find_spec("graphify.extractors.databricks.lineage") is not None
needs_lineage = pytest.mark.skipif(not HAS_LINEAGE, reason="lineage.py not available")


@pytest.fixture
def root(tmp_path):
    previous = set_active_scan_root(tmp_path.resolve())
    try:
        yield tmp_path.resolve()
    finally:
        set_active_scan_root(previous)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


PY_NOTEBOOK = """\
# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest orders
# MAGIC Loads raw orders.

# COMMAND ----------

# MAGIC %run ./utils/helpers $catalog="dev"

# COMMAND ----------

catalog = dbutils.widgets.get("catalog")
dbutils.widgets.text("catalog", "dev")
dbutils.widgets.text("api_token", "not-a-real-token-value")

def load():
    df = spark.table(f"{catalog}.bronze.orders")
    return clean(df)

def fan_out():
    dbutils.notebook.run("./utils/helpers", 60, {"catalog": catalog, "day": "1"})
    dbutils.notebook.run(f"./{catalog}/x", 60)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE silver.orders AS
# MAGIC SELECT * FROM bronze.orders
"""

HELPERS = """\
# Databricks notebook source
def clean(df):
    return df.dropna()
"""


def _ipynb(cells, metadata=None) -> str:
    return json.dumps({"cells": cells, "metadata": metadata or {}, "nbformat": 4, "nbformat_minor": 5})


def _code(source, **meta):
    return {"cell_type": "code", "source": source, "metadata": meta, "outputs": [], "execution_count": None}


def _schema_ok(result) -> bool:
    """Schema-valid, ignoring per-file dangling targets (import/table stubs the
    corpus pass resolves), which every per-file Python result has."""
    errors = [e for e in validate_extraction(result) if "does not match any node id" not in e]
    assert errors == []
    return True


def _by_label(result, label):
    return next(n for n in result["nodes"] if n["label"] == label)


def _edges(result, relation):
    return [e for e in result["edges"] if e["relation"] == relation]


# ── ownership ────────────────────────────────────────────────────────────────

def test_owns_source_notebooks_by_header(tmp_path):
    assert nb.owns(_write(tmp_path / "a.py", "# Databricks notebook source\nx = 1\n"))
    assert nb.owns(_write(tmp_path / "b.sql", "-- Databricks notebook source\nSELECT 1\n"))
    assert nb.owns(_write(tmp_path / "c.scala", "// Databricks notebook source\nval x = 1\n"))
    assert nb.owns(_write(tmp_path / "d.r", "# Databricks notebook source\nx <- 1\n"))
    assert nb.owns(tmp_path / "anything.ipynb")  # no read needed
    assert not nb.owns(_write(tmp_path / "plain.py", "x = 1\n# Databricks notebook source\n"))
    assert not nb.owns(_write(tmp_path / "wrong.sql", "# Databricks notebook source\n"))


# ── cell splitting ───────────────────────────────────────────────────────────

def test_split_source_py_notebook(root):
    path = _write(root / "ingest.py", PY_NOTEBOOK)
    cells = nb.split_cells(path)
    assert [c.language for c in cells] == ["md", "run", "python", "sql"]
    md, run, py, sql = cells
    assert md.start_line == 2 and "# Ingest orders" in md.text
    assert run.magic == "%run" and run.magic_args.startswith("./utils/helpers")
    assert run.magic_line == 8
    assert py.magic is None and "def load():" in py.text
    assert sql.magic == "%sql"
    # text lines map 1:1 onto file lines
    lines = PY_NOTEBOOK.split("\n")
    for offset, text in enumerate(sql.text.split("\n")):
        if "CREATE" in text:
            assert "CREATE OR REPLACE" in lines[sql.start_line - 1 + offset]


# ── .py source notebooks ─────────────────────────────────────────────────────

def test_py_notebook_extract(root):
    _write(root / "utils" / "helpers.py", HELPERS)
    path = _write(root / "ingest.py", PY_NOTEBOOK)
    result = nb.extract(path)
    assert _schema_ok(result)

    file_node = _by_label(result, "ingest.py")
    assert file_node["notebook"] is True
    assert file_node["notebook_language"] == "python"
    assert file_node["notebook_format"] == "source"
    widgets = {w["name"]: w["default"] for w in file_node["db_widgets"]}
    assert widgets == {"catalog": "dev", "api_token": None}  # secret-shaped default dropped

    helpers_id = nb.file_ref_id(root / "utils" / "helpers.py")
    runs = _edges(result, "runs_notebook")
    run_edge = next(e for e in runs if e.get("context") == "%run")
    assert run_edge["source"] == file_node["id"] and run_edge["target"] == helpers_id
    assert run_edge["source_location"] == "L8"

    dbr = [e for e in runs if e.get("context") == "dbutils.notebook.run"]
    assert len(dbr) == 1  # the f-string path is skipped
    assert dbr[0]["source"] == _by_label(result, "fan_out()")["id"]
    assert dbr[0]["target"] == helpers_id
    assert dbr[0]["params"] == ["catalog", "day"]

    # %run of an importable .py module behaves like a star import.
    imports = [e for e in result["edges"] if e["relation"] in ("imports_from", "imports")]
    assert any(e["target"] == helpers_id or "helpers" in e["target"] for e in imports)

    # %md headings surface as nodes on their real lines
    heading = _by_label(result, "Ingest orders")
    assert heading["source_location"] == "L3"

    # Python symbols keep real line numbers
    assert _by_label(result, "load()")["source_location"] == "L16"


def test_unresolved_run_target_mints_nothing(root):
    path = _write(root / "nb.py", "# Databricks notebook source\n# MAGIC %run ./missing\n")
    result = nb.extract(path)
    assert _edges(result, "runs_notebook") == []
    assert _schema_ok(result)


@needs_lineage
def test_py_notebook_tables(root):
    path = _write(root / "ingest.py", PY_NOTEBOOK)
    result = nb.extract(path)
    tables = {n["label"]: n for n in result["nodes"] if n.get("type") == "uc_table"}
    assert {"bronze.orders", "silver.orders"} <= set(tables)
    assert tables["silver.orders"]["source_location"] == "L27"
    load = _by_label(result, "load()")["id"]
    assert any(e["source"] == load and e["target"] == tables["bronze.orders"]["id"]
               for e in _edges(result, "reads_from"))
    # one node per table even though the core SQL grammar also saw the CREATE
    assert sum(1 for n in result["nodes"] if n["label"] == "silver.orders") == 1


# ── .sql source notebooks ────────────────────────────────────────────────────

SQL_NOTEBOOK = """\
-- Databricks notebook source
CREATE WIDGET TEXT catalog DEFAULT 'dev'

-- COMMAND ----------

CREATE OR REPLACE TABLE gold.daily AS SELECT * FROM silver.orders

-- COMMAND ----------

-- MAGIC %python
-- MAGIC def summarize(df):
-- MAGIC     return df.count()
"""


def test_sql_notebook_with_python_cell(root):
    path = _write(root / "daily.sql", SQL_NOTEBOOK)
    cells = nb.split_cells(path)
    assert [c.language for c in cells] == ["sql", "sql", "python"]
    result = nb.extract(path)
    assert _schema_ok(result)
    fn = _by_label(result, "summarize()")
    assert fn["source_location"] == "L11"
    file_node = _by_label(result, "daily.sql")
    assert file_node["notebook_language"] == "sql"
    assert file_node["db_widgets"] == [{"name": "catalog", "default": "dev", "kind": "text"}]


@needs_lineage
def test_sql_notebook_tables(root):
    result = nb.extract(_write(root / "daily.sql", SQL_NOTEBOOK))
    labels = {n["label"] for n in result["nodes"] if n.get("type") == "uc_table"}
    assert {"gold.daily", "silver.orders"} <= labels


# ── .ipynb ───────────────────────────────────────────────────────────────────

def test_ipynb_databricks_flavored(root):
    _write(root / "lib" / "common.py", "def helper():\n    return 1\n")
    path = _write(root / "job.ipynb", _ipynb(
        [
            {"cell_type": "markdown", "source": ["# Job\n", "Text\n"], "metadata": {}},
            _code(["%run ./lib/common\n"]),
            _code(["import os\n", "\n", "def main():\n", "    return helper()\n"]),
            _code("%sql\nSELECT * FROM silver.orders"),
        ],
        metadata={"application/vnd.databricks.v1+notebook": {"language": "python", "notebookName": "job"}},
    ))
    cells = nb.split_cells(path)
    assert [c.language for c in cells] == ["md", "run", "python", "sql"]
    assert [c.index for c in cells] == [0, 1, 2, 3]
    # virtual layout: each cell, then one separator line
    assert [(c.start_line, c.end_line) for c in cells] == [(1, 2), (4, 4), (6, 9), (11, 12)]

    result = nb.extract(path)
    assert _schema_ok(result)
    main = _by_label(result, "main()")
    assert main["source_location"] == "L8" and main["cell"] == 2
    file_node = _by_label(result, "job.ipynb")
    assert file_node["notebook_format"] == "ipynb"
    run = _edges(result, "runs_notebook")
    assert run and run[0]["target"] == nb.file_ref_id(root / "lib" / "common.py")
    assert run[0]["cell"] == 1
    assert _by_label(result, "Job")["cell"] == 0


def test_ipynb_plain_jupyter_magics(root):
    path = _write(root / "explore.ipynb", _ipynb(
        [
            _code("!pip install pandas\n%matplotlib inline\nimport pandas as pd\n"),
            _code("%%capture\ndef quiet():\n    pass\n"),
            _code("%%sql\nSELECT * FROM raw.events"),
            _code("%%bash\nls -la"),
            {"cell_type": "code", "source": "def tail():\n    pass", "metadata": {},
             "outputs": [{"output_type": "display_data", "data": {"image/png": "A" * 200_000}}]},
        ],
        metadata={"kernelspec": {"language": "python", "name": "python3"}},
    ))
    assert [c.language for c in nb.split_cells(path)] == ["python", "python", "sql", "sh", "python"]
    result = nb.extract(path)
    assert _schema_ok(result)
    labels = {n["label"] for n in result["nodes"]}
    assert {"quiet()", "tail()"} <= labels


def test_ipynb_cell_language_metadata_and_scala_kernel(root):
    path = _write(root / "s.ipynb", _ipynb(
        [_code("val x = 1"), _code("def py_fn():\n    pass", **{"application/vnd.databricks.v1+cell": {"language": "python"}})],
        metadata={"kernelspec": {"language": "scala"}},
    ))
    cells = nb.split_cells(path)
    assert [c.language for c in cells] == ["scala", "python"]
    result = nb.extract(path)
    assert _by_label(result, "s.ipynb")["notebook_language"] == "scala"
    assert "py_fn()" in {n["label"] for n in result["nodes"]}


def test_ipynb_empty_and_non_python(root):
    result = nb.extract(_write(root / "e.ipynb", _ipynb([], metadata={"kernelspec": {"language": "r"}})))
    assert [n["label"] for n in result["nodes"]] == ["e.ipynb"]
    assert _schema_ok(result)


@pytest.mark.parametrize("body", ["{not json", "[]", '{"cells": 3}'])
def test_ipynb_malformed_returns_error(root, body):
    result = nb.extract(_write(root / "bad.ipynb", body))
    assert result["nodes"] == [] and "error" in result


# ── pipeline integration ─────────────────────────────────────────────────────

def test_detect_and_full_pipeline(root):
    from graphify.detect import count_words, detect
    from graphify.extract import collect_files, extract

    _write(root / "utils" / "helpers.py", HELPERS)
    _write(root / "ingest.py", PY_NOTEBOOK)
    _write(root / "pipeline.ipynb", _ipynb(
        [_code("%run ./utils/helpers"), _code("def go(df):\n    return clean(df)\n")],
        metadata={"kernelspec": {"language": "python"}},
    ))
    _write(root / ".ipynb_checkpoints" / "pipeline-checkpoint.ipynb", _ipynb([_code("x = 1")]))

    found = detect(root)
    code = {Path(p).name for p in found["files"]["code"]}
    assert "pipeline.ipynb" in code
    assert "pipeline-checkpoint.ipynb" not in code
    assert count_words(root / "pipeline.ipynb") > 0

    files = collect_files(root)
    names = {p.name for p in files}
    assert "pipeline.ipynb" in names and "pipeline-checkpoint.ipynb" not in names

    result = extract(files, root=root)
    nodes = {n["id"]: n for n in result["nodes"]}
    helpers = next(i for i, n in nodes.items() if n.get("label") == "helpers.py")
    pipeline = next(i for i, n in nodes.items() if n.get("label") == "pipeline.ipynb")
    ingest = next(i for i, n in nodes.items() if n.get("label") == "ingest.py")
    runs = {(e["source"], e["target"]) for e in result["edges"] if e["relation"] == "runs_notebook"}
    assert (pipeline, helpers) in runs and (ingest, helpers) in runs
    # %run'd definitions are callable cross-file, like a star import
    calls = {(nodes[e["source"]]["label"], nodes[e["target"]]["label"])
             for e in result["edges"] if e["relation"] == "calls" and e["target"] in nodes}
    assert ("go()", "clean()") in calls
    assert ("load()", "clean()") in calls


def test_ipynb_word_count_ignores_outputs(tmp_path):
    from graphify.detect import count_words

    path = _write(tmp_path / "n.ipynb", _ipynb([
        {"cell_type": "code", "source": ["a b c"], "metadata": {},
         "outputs": [{"output_type": "stream", "text": "x " * 1000}]},
    ]))
    assert count_words(path) == 3
