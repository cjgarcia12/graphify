"""Unity Catalog lineage from PySpark / DLT / Lakeflow / Databricks SQL."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from graphify.extractors.databricks import lineage
from graphify.extractors.databricks.config import DatabricksConfig, set_config_override
from graphify.extractors.databricks.lineage import scan_python, scan_sql
from graphify.extractors.databricks.names import table_node_id

DEFAULT = DatabricksConfig()


@pytest.fixture(autouse=True)
def _reset_config():
    set_config_override(None)
    yield
    set_config_override(None)


def _refs(src: str, cfg: DatabricksConfig = DEFAULT):
    return [(r.relation, r.raw, r.kind) for r in scan_python(textwrap.dedent(src), cfg=cfg)]


def _sql(text: str, **kw):
    return [(r.relation, r.raw, r.kind) for r in scan_sql(textwrap.dedent(text), **kw)]


# ── Python: Spark reads and writes ───────────────────────────────────────────


def test_spark_reads():
    refs = _refs("""
        a = spark.table("main.silver.orders")
        b = spark.read.table("silver.customers")
        c = spark.readStream.table("bronze.events")
        d = spark.read.format("delta").table("gold.kpis")
    """)
    assert [(r, n) for r, n, _ in refs] == [
        ("reads_from", "main.silver.orders"),
        ("reads_from", "silver.customers"),
        ("reads_from", "bronze.events"),
        ("reads_from", "gold.kpis"),
    ]


def test_spark_writes():
    refs = _refs("""
        df.write.mode("overwrite").saveAsTable("silver.a")
        df.write.insertInto("silver.b")
        df.writeStream.option("x", 1).toTable("silver.c")
        df.writeTo("silver.d").createOrReplace()
    """)
    assert {(r, n) for r, n, _ in refs} == {
        # saveAsTable / toTable create a missing table, so they define it.
        ("defines", "silver.a"), ("writes_to", "silver.b"),
        ("defines", "silver.c"), ("writes_to", "silver.d"),
    }


def test_delta_table_handle_merge_is_write_plain_is_read():
    refs = _refs("""
        from delta.tables import DeltaTable
        def upsert(df):
            dt = DeltaTable.forName(spark, "gold.dim")
            dt.alias("t").merge(df.alias("s"), "t.id = s.id").whenMatchedUpdateAll().execute()
            DeltaTable.forName(spark, "gold.chained").alias("t").merge(df, "x").execute()
            DeltaTable.forName(spark, "gold.lookup").toDF()
    """)
    assert ("writes_to", "gold.dim", "unknown") in refs
    assert ("writes_to", "gold.chained", "unknown") in refs
    assert ("reads_from", "gold.lookup", "unknown") in refs


def test_spark_sql_strings_delegate_to_sql_scanner_with_lines():
    src = textwrap.dedent('''
        def build():
            spark.sql("""
                INSERT INTO gold.agg
                SELECT * FROM silver.orders
            """)
    ''')
    refs = scan_python(src, cfg=DEFAULT)
    by = {(r.relation, r.raw): r.line for r in refs}
    assert by[("writes_to", "gold.agg")] == 4
    assert by[("reads_from", "silver.orders")] == 5
    assert all(r.extra["enclosing"] == ("build", 2) for r in refs)


# ── Python: name evaluation ──────────────────────────────────────────────────


def test_widgets_constants_fstrings_format_concat():
    refs = _refs("""
        import sys
        catalog = dbutils.widgets.get("catalog")
        env = spark.conf.get("pipeline.env")
        SCHEMA = "silver"
        TBL = f"{catalog}.{SCHEMA}.orders"
        spark.table(TBL)
        spark.table("{}.bronze.x".format(catalog))
        spark.table("{cat}.bronze.y".format(cat=catalog))
        spark.table("%s.bronze.z" % catalog)
        spark.table(catalog + ".gold.w")
        spark.table(f"{cfg.catalog}.gold.v")
        spark.table(f"{env}_raw.events")
        spark.table(sys.argv[1] + ".gold.u")
    """)
    names = [n for _, n, _ in refs]
    assert names == [
        "{catalog}.silver.orders", "{catalog}.bronze.x", "{catalog}.bronze.y",
        "{catalog}.bronze.z", "{catalog}.gold.w", "{catalog}.gold.v",
        "{env}_raw.events", "{argv_1}.gold.u",
    ]


def test_unresolvable_names_are_skipped_not_guessed():
    refs = _refs("""
        spark.table(self.table_name)
        spark.table(get_name())
        spark.table(f"{catalog}.{schema}.{table}")
        x = "a" if flag else "b"
        spark.table(x)
    """)
    assert refs == []


def test_conflicting_assignments_do_not_resolve():
    refs = _refs("""
        t = "silver.a"
        t = "silver.b"
        spark.table(t)
    """)
    assert refs == []


def test_function_scope_shadows_module_constant():
    refs = _refs("""
        T = "silver.module"
        def f():
            T = "silver.local"
            return spark.table(T)
        def g():
            return spark.table(T)
    """)
    assert [n for _, n, _ in refs] == ["silver.local", "silver.module"]


# ── Python: DLT / Lakeflow ───────────────────────────────────────────────────


def test_dlt_definitions_and_reads():
    refs = scan_python(textwrap.dedent("""
        import dlt

        @dlt.table(name="orders_clean", comment="x")
        @dlt.expect("valid", "id is not null")
        def orders():
            return dlt.read_stream("orders_raw")

        @dlt.view
        def v_customers():
            return spark.read.table("LIVE.customers")

        @dlt.materialized_view()
        def daily():
            return dlt.read("orders_clean")

        dlt.create_streaming_table("scd_orders")
        dlt.apply_changes(target="scd_orders", source="orders_clean", keys=["id"])
    """), cfg=DEFAULT)
    got = [(r.relation, r.raw, r.kind, r.extra.get("enclosing", (None,))[0]) for r in refs]
    assert ("defines", "orders_clean", "table", "orders") in got
    assert ("reads_from", "orders_raw", "unknown", "orders") in got
    assert ("defines", "v_customers", "view", "v_customers") in got
    assert ("reads_from", "LIVE.customers", "unknown", "v_customers") in got
    assert ("defines", "daily", "materialized_view", "daily") in got
    assert ("defines", "scd_orders", "streaming_table", None) in got
    assert ("writes_to", "scd_orders", "unknown", None) in got
    assert ("reads_from", "orders_clean", "unknown", None) in got


@pytest.mark.parametrize("imp,alias", [
    ("from pyspark import pipelines as dp", "dp"),
    ("import pyspark.pipelines as sdp", "sdp"),
    ("import dlt as d", "d"),
])
def test_lakeflow_and_aliased_imports(imp, alias):
    refs = _refs(f"""
        {imp}

        @{alias}.materialized_view(name="gold_kpis")
        def kpis():
            return spark.read.table("silver_orders")

        {alias}.create_auto_cdc_flow(target="dim", source="silver_orders", keys=["id"])

        @{alias}.append_flow(target="events_all")
        def flow_a():
            return spark.readStream.table("events_a")
    """)
    assert ("defines", "gold_kpis", "materialized_view") in refs
    assert ("writes_to", "dim", "unknown") in refs
    assert ("writes_to", "events_all", "unknown") in refs
    assert ("reads_from", "events_a", "unknown") in refs


def test_from_dlt_import_names():
    refs = _refs("""
        from dlt import table, read
        @table
        def bronze():
            return read("landing")
    """)
    assert ("defines", "bronze", "table") in refs
    assert ("reads_from", "landing", "unknown") in refs


def test_non_dlt_table_decorator_is_ignored():
    refs = _refs("""
        import functools
        @functools.cache
        def f():
            return 1
        @app.table(name="x")
        def g():
            return 1
    """)
    assert refs == []


# ── Python: team wrappers via config ─────────────────────────────────────────


def test_configured_reader_and_writer_wrappers():
    cfg = DatabricksConfig(readers=frozenset({"read_table"}), writers=frozenset({"upsert_to_delta"}))
    refs = _refs("""
        from utils.io import read_table
        df = read_table("silver.orders")
        helpers.upsert_to_delta(df, table_name="gold.orders")
        other("silver.nope")
    """, cfg)
    assert refs == [("reads_from", "silver.orders", "unknown"), ("writes_to", "gold.orders", "unknown")]
    assert _refs('read_table("silver.orders")') == []  # not configured -> nothing


# ── SQL ──────────────────────────────────────────────────────────────────────


def test_sql_create_variants_and_kinds():
    got = _sql("""
        CREATE OR REFRESH STREAMING TABLE bronze.raw AS SELECT * FROM STREAM read_files('/x');
        CREATE OR REPLACE MATERIALIZED VIEW gold.daily AS SELECT * FROM LIVE.raw;
        CREATE TABLE IF NOT EXISTS silver.t (id INT) USING DELTA;
        CREATE OR REPLACE TEMPORARY VIEW tmp_v AS SELECT * FROM silver.t;
        CREATE LIVE TABLE legacy AS SELECT 1;
    """)
    assert ("defines", "bronze.raw", "streaming_table") in got
    assert ("defines", "gold.daily", "materialized_view") in got
    assert ("reads_from", "LIVE.raw", "unknown") in got
    assert ("defines", "silver.t", "table") in got
    assert ("defines", "tmp_v", "view") in got
    assert ("defines", "legacy", "table") in got
    assert not any("read_files" in n or "delta" == n.lower() for _, n, _ in got)


def test_sql_dml_and_merge():
    got = _sql("""
        INSERT OVERWRITE TABLE gold.a SELECT * FROM silver.b b JOIN silver.c c ON b.id = c.id;
        MERGE INTO gold.dim t USING silver.updates s ON t.id = s.id WHEN MATCHED THEN UPDATE SET *;
        UPDATE gold.x SET a = 1;
        DELETE FROM gold.y WHERE extract(year FROM d) = 2020;
        COPY INTO bronze.landing FROM '/mnt/x' FILEFORMAT = JSON;
    """)
    assert set(got) >= {
        ("writes_to", "gold.a", "unknown"), ("reads_from", "silver.b", "unknown"),
        ("reads_from", "silver.c", "unknown"), ("writes_to", "gold.dim", "unknown"),
        ("reads_from", "silver.updates", "unknown"), ("writes_to", "gold.x", "unknown"),
        ("writes_to", "gold.y", "unknown"), ("writes_to", "bronze.landing", "unknown"),
    }
    names = [n for _, n, _ in got]
    assert "d" not in names and "gold.y" in names
    assert names.count("gold.y") == 1  # DELETE FROM is a write only


def test_sql_ctes_comments_strings_and_paths_excluded():
    got = _sql("""
        -- SELECT * FROM commented.out
        /* FROM block.comment */
        WITH recent AS (SELECT * FROM silver.orders), top AS (SELECT * FROM recent)
        SELECT 'FROM in.string', * FROM top, silver.customers cu
        UNION ALL SELECT * FROM delta.`/mnt/raw/x`;
    """)
    names = {n for _, n, _ in got}
    assert "silver.orders" in names and "silver.customers" in names
    assert not names & {"recent", "top", "commented.out", "block.comment", "in.string"}


def test_sql_placeholders_identifier_and_use():
    got = _sql("""
        USE CATALOG ${var.catalog};
        USE SCHEMA silver;
        SELECT * FROM orders;
        INSERT INTO {{catalog}}.gold.t SELECT * FROM IDENTIFIER(:cat || '.bronze.x');
        SELECT * FROM $env_schema.events;
        SELECT * FROM STREAM(other.s);
    """)
    assert ("reads_from", "{catalog}.silver.orders", "unknown") in got
    assert ("writes_to", "{catalog}.gold.t", "unknown") in got
    assert ("reads_from", "`{cat}`.`bronze`.`x`", "unknown") in got
    # a two-part name picks up the USE CATALOG
    assert ("reads_from", "{catalog}.{env_schema}.events", "unknown") in got
    assert ("reads_from", "{catalog}.other.s", "unknown") in got


def test_sql_default_schema_and_line_numbers():
    refs = scan_sql("SELECT 1;\n\nSELECT *\nFROM orders", default_schema="silver", line_offset=10)
    assert [(r.raw, r.line) for r in refs] == [("silver.orders", 14)]


def test_sql_casts_are_not_placeholders():
    got = _sql("SELECT a::string FROM silver.t WHERE ts > '12:30'")
    assert got == [("reads_from", "silver.t", "unknown")]


# ── ownership ────────────────────────────────────────────────────────────────


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


def test_owns(tmp_path):
    spark_py = _write(tmp_path, "job.py", "df = spark.table('a.b')\n")
    plain_py = _write(tmp_path, "util.py", "def f():\n    return 1\n")
    nb = _write(tmp_path, "nb.py", "# Databricks notebook source\nspark.table('a.b')\n")
    dbx_sql = _write(tmp_path, "a.sql", "CREATE OR REFRESH STREAMING TABLE x AS SELECT 1;\n")
    pg_sql = _write(tmp_path, "b.sql", "CREATE TABLE users (id int);\nSELECT * FROM users;\n")
    three = _write(tmp_path, "c.sql", "SELECT * FROM main.silver.orders;\n")
    tsql = _write(tmp_path, "d.sql", "SELECT * FROM db.dbo.orders WHERE id = @id;\nGO\n")
    assert lineage.owns(spark_py)
    assert not lineage.owns(plain_py)
    assert not lineage.owns(nb)
    assert lineage.owns(dbx_sql)
    assert not lineage.owns(pg_sql)
    assert lineage.owns(three)
    assert not lineage.owns(tsql)
    set_config_override(DatabricksConfig(mode="on"))
    assert lineage.owns(plain_py)
    set_config_override(DatabricksConfig(mode="off"))
    assert not lineage.owns(spark_py)
    set_config_override(DatabricksConfig(readers=frozenset({"load_bronze"})))
    wrapper = _write(tmp_path, "w.py", "x = load_bronze('raw.events')\n")
    assert lineage.owns(wrapper)


# ── extract(): attaching to the core result ──────────────────────────────────


def test_extract_python_attributes_to_functions_and_drops_dlt_stubs(tmp_path):
    p = _write(tmp_path, "pipe.py", """
        import dlt

        @dlt.table(name="orders_clean")
        def orders():
            return dlt.read("orders_raw")

        def load(df):
            df.write.saveAsTable("dev.silver.orders")

        spark.table("silver.top_level")
    """)
    from graphify.extractors.base import set_active_scan_root

    prev = set_active_scan_root(tmp_path)
    try:
        result = lineage.extract(p)
    finally:
        set_active_scan_root(prev)
    by_id = {n["id"]: n for n in result["nodes"]}
    labels = {n["label"]: n for n in result["nodes"]}
    assert "table" not in labels  # the @dlt.table stub is gone
    edges = {(by_id[e["source"]]["label"], e["relation"], by_id[e["target"]]["label"])
             for e in result["edges"] if e["source"] in by_id and e["target"] in by_id}
    assert ("orders()", "writes_to", "orders_clean") in edges
    assert ("orders()", "reads_from", "orders_raw") in edges
    assert ("load()", "writes_to", "silver.orders") in edges
    assert ("pipe.py", "reads_from", "silver.top_level") in edges
    assert labels["orders_clean"]["source_file"] == str(p)  # defined here
    assert labels["silver.orders"]["uc_catalogs"] == ["dev"]


def test_extract_sql_folds_core_tables_onto_uc_tables(tmp_path):
    p = _write(tmp_path, "gold.sql", """
        CREATE TABLE main.gold.agg (id INT) USING DELTA;
        INSERT INTO main.gold.agg SELECT * FROM main.silver.orders;
    """)
    from graphify.extractors.base import set_active_scan_root

    prev = set_active_scan_root(tmp_path)
    try:
        assert lineage.owns(p)
        result = lineage.extract(p)
    finally:
        set_active_scan_root(prev)
    tables = [n for n in result["nodes"] if "agg" in str(n.get("label", "")).lower()]
    assert len(tables) == 1 and tables[0]["type"] == "uc_table"
    ids = {n["id"] for n in result["nodes"]}
    assert all(e["source"] in ids or e["target"] in ids for e in result["edges"])
    rels = {(e["relation"], e["target"]) for e in result["edges"]}
    assert ("writes_to", table_node_id("gold.agg")) in rels
    assert ("reads_from", table_node_id("silver.orders")) in rels


def test_end_to_end_one_table_node_across_files(tmp_path):
    """A writer using a widget-driven catalog and a reader using a literal
    catalog meet on ONE uc_table node through the full extract() pipeline."""
    _write(tmp_path, "jobs/write_orders.py", """
        catalog = dbutils.widgets.get("catalog")

        def write(df):
            df.write.saveAsTable(f"{catalog}.silver.orders")
    """)
    _write(tmp_path, "jobs/report.py", """
        def report():
            return spark.table("dev.silver.orders")
    """)
    _write(tmp_path, "sql/agg.sql", """
        INSERT INTO ${var.catalog}.gold.agg SELECT * FROM ${var.catalog}.silver.orders;
    """)
    from graphify.extract import extract

    paths = sorted(tmp_path.rglob("*.py")) + sorted(tmp_path.rglob("*.sql"))
    result = extract(paths, root=tmp_path)
    ids = {n["id"] for n in result["nodes"] if n.get("type") == "uc_table" and n.get("label") == "silver.orders"}
    assert len(ids) == 1, ids  # no per-file disambiguation of the table id
    from graphify.build import build_from_json

    G = build_from_json(result)
    tid = ids.pop()
    assert G.nodes[tid]["type"] == "uc_table"
    labels = {n: d.get("label") for n, d in G.nodes(data=True)}
    incoming = set()
    for u, v, d in G.edges(data=True):
        for a, b in ((u, v), (v, u)):
            if b == tid and d.get("relation") in ("reads_from", "writes_to"):
                incoming.add((labels.get(a), d.get("relation")))
    assert ("write()", "writes_to") in incoming
    assert ("report()", "reads_from") in incoming
    assert ("agg.sql", "reads_from") in incoming
