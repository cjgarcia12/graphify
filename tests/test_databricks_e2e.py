"""End-to-end: tests/fixtures/databricks_repo graphs as one connected system.

The fixture is a small sales lakehouse: a bundle (databricks.yml + included
job and pipeline YAML, catalog switched by ${var.catalog}), source-format and
.ipynb notebooks reading/writing Unity Catalog tables through widgets and
f-strings, a DLT pipeline with one-part dataset names, a wheel package run by a
python_wheel_task, and a Databricks SQL view.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from graphify.build import build_from_json
from graphify.extract import collect_files, extract

FIXTURE = Path(__file__).parent / "fixtures" / "databricks_repo"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    dst = tmp_path / "repo"
    shutil.copytree(FIXTURE, dst)
    return dst


def _graph(root: Path):
    result = extract(collect_files(root), root=root, cache_root=root / "graphify-out")
    return build_from_json(result, root=root)


def _by_label(G) -> dict:
    return {d.get("label"): n for n, d in G.nodes(data=True)}


def _directed(G, relation: str) -> set[tuple[str, str]]:
    out = set()
    for u, v, d in G.edges(data=True):
        if d.get("relation") == relation:
            src, tgt = d.get("_src", u), d.get("_tgt", v)
            out.add((G.nodes[src].get("label"), G.nodes[tgt].get("label")))
    return out


def _table_edges(G, table_label: str) -> tuple[set[str], set[str]]:
    t = _by_label(G)[table_label]
    writers, readers = set(), set()
    for u in G.neighbors(t):
        d = G.edges[t, u]
        src = G.nodes[d.get("_src", u)].get("label")
        if d["relation"] == "writes_to":
            writers.add(src)
            if d.get("also_reads"):
                readers.add(src)
        elif d["relation"] == "reads_from":
            readers.add(src)
    return writers, readers


def test_table_lineage_connects_every_layer(repo: Path) -> None:
    G = _graph(repo)
    tables = {d["label"]: d for _, d in G.nodes(data=True) if d.get("type") == "uc_table"}

    assert _table_edges(G, "bronze.orders_raw") == ({"ingest_orders.py"}, {"orders_clean()"})
    assert tables["bronze.orders_raw"]["source_file"].endswith("src/bronze/ingest_orders.py")

    # Defined by the %sql cell of the source notebook, read by a DLT dataset.
    assert _table_edges(G, "bronze.customers_raw") == ({"ingest_orders.py"}, {"customers()"})
    assert tables["bronze.customers_raw"]["source_file"].endswith("src/bronze/ingest_orders.py")

    # DLT one-part names are qualified with the pipeline's schema, and the
    # ipynb's three-part read lands on the same node as the DLT definition.
    for name in ("orders_clean", "customers", "orders_enriched"):
        assert f"silver.{name}" in tables and name not in tables
        assert tables[f"silver.{name}"]["source_file"].endswith("src/pipelines/silver_orders.py")
    assert _table_edges(G, "silver.orders_enriched") == ({"orders_enriched()"}, {"compute()"})
    assert _table_edges(G, "silver.orders_clean")[1] == {"orders_enriched()"}

    # Written by the ipynb (and read back by its %sql cell), read by the wheel
    # function and the SQL view.
    writers, readers = _table_edges(G, "gold.daily_revenue")
    assert writers == {"daily_revenue.ipynb"}
    assert {"publish()", "daily_revenue.ipynb"} <= readers
    assert tables["gold.daily_revenue"]["source_file"].endswith("src/gold/daily_revenue.ipynb")

    catalogs = {c for d in tables.values() for c in d.get("uc_catalogs", [])}
    assert {"dev", "prod", "{catalog}"} <= catalogs
    # dev/prod are learned from the bundle's catalog variable.
    assert "prod" in tables["gold.daily_revenue"]["uc_env_catalogs"]


def test_sql_view_reads_gold(repo: Path) -> None:
    # Databricks SQL lineage comes from lineage.scan_sql, so it holds with or
    # without the optional tree-sitter-sql grammar.
    G = _graph(repo)
    assert "revenue_view.sql" in _table_edges(G, "gold.daily_revenue")[1]
    assert _table_edges(G, "reporting.revenue_by_month")[0] == {"revenue_view.sql"}


def test_bundle_job_dag_and_task_targets(repo: Path) -> None:
    G = _graph(repo)
    deps = _directed(G, "depends_on")
    assert deps == {
        ("sales_daily.refresh_silver", "sales_daily.ingest_orders"),
        ("sales_daily.build_gold", "sales_daily.refresh_silver"),
        ("sales_daily.publish_metrics", "sales_daily.build_gold"),
    }
    runs = _directed(G, "runs_notebook")
    assert ("sales_daily.ingest_orders", "ingest_orders.py") in runs
    assert ("sales_daily.build_gold", "daily_revenue.ipynb") in runs
    assert ("ingest_orders.py", "config.py") in runs  # %run ../common/config
    assert ("sales_daily.refresh_silver", "sales-silver-${bundle.target}") in _directed(G, "runs_pipeline")
    assert ("sales-silver-${bundle.target}", "silver_orders.py") in _directed(G, "pipeline_source")
    assert ("sales_daily.publish_metrics", "sales_lib") in _directed(G, "runs_package")
    assert ("sales_daily.publish_metrics", "publish()") in _directed(G, "runs_entry_point")
    assert len(_directed(G, "has_task")) == 4


def test_cache_rerun_is_identical(repo: Path) -> None:
    def key(G):
        return (sorted(G.nodes()), sorted(
            (d.get("_src", u), d.get("_tgt", v), d.get("relation")) for u, v, d in G.edges(data=True)))

    assert key(_graph(repo)) == key(_graph(repo))


def test_late_added_notebook_binds_without_re_extracting_bundle(repo: Path) -> None:
    """A notebook the bundle names but that did not exist when the YAML was
    cached is linked on the next build (corpus-pass late binding)."""
    jobs = repo / "resources" / "sales_job.yml"
    jobs.write_text(jobs.read_text() + (
        "        - task_key: audit\n"
        "          notebook_task:\n"
        "            notebook_path: ../src/audit/check\n"
    ))
    G = _graph(repo)
    assert not any(t == "sales_daily.audit" for t, _ in _directed(G, "runs_notebook"))
    (repo / "src" / "audit").mkdir()
    (repo / "src" / "audit" / "check.py").write_text(
        "# Databricks notebook source\nspark.table('gold.daily_revenue').count()\n")
    G = _graph(repo)
    assert ("sales_daily.audit", "check.py") in _directed(G, "runs_notebook")


def test_run_target_added_later_invalidates_notebook_cache(repo: Path) -> None:
    nb = repo / "src" / "bronze" / "late.py"
    nb.write_text("# Databricks notebook source\n# MAGIC %run ./helpers\n\n"
                  "# COMMAND ----------\n\ndef go():\n    return clean()\n")
    G = _graph(repo)
    assert ("late.py", "helpers.py") not in _directed(G, "runs_notebook")
    (repo / "src" / "bronze" / "helpers.py").write_text("def clean():\n    return 1\n")
    G = _graph(repo)
    assert ("late.py", "helpers.py") in _directed(G, "runs_notebook")
    assert ("go()", "clean()") in _directed(G, "calls")


def test_cli_update_writes_lineage_report(repo: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "graphify", "update", "."], cwd=repo,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = (repo / "graphify-out" / "GRAPH_REPORT.md").read_text()
    assert "## Data lineage (Databricks)" in report
    assert "`silver.orders_enriched`" in report
    assert "publish_metrics <- build_gold" in report
    graph = json.loads((repo / "graphify-out" / "graph.json").read_text())
    links = graph.get("links", graph.get("edges"))
    assert {"source": "dab_task_sales_daily_build_gold", "target": "dab_task_sales_daily_refresh_silver"}.items() <= next(
        e for e in links if e.get("relation") == "depends_on"
        and e["source"] == "dab_task_sales_daily_build_gold").items()
