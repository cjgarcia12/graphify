"""Unit tests for the Databricks corpus pass (resolve.run) and cache salt."""
from __future__ import annotations

from pathlib import Path

import pytest

from graphify.extractors.databricks import cache_salt
from graphify.extractors.databricks.config import (
    DatabricksConfig,
    parse_config,
    set_config_override,
)
from graphify.extractors.databricks.names import table_node_id
from graphify.extractors.databricks.resolve import run


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    set_config_override(None)


def _table(label, **kw):
    return {"id": table_node_id(label), "label": label, "type": "uc_table",
            "source_file": "", "source_location": "", "uc_catalogs": [], "uc_kind": "unknown", **kw}


def _edge(src, tgt, rel, sf="a.py", **kw):
    return {"source": src, "target": tgt, "relation": rel, "source_file": sf, **kw}


def test_noop_without_databricks_nodes():
    nodes = [{"id": "a", "label": "a"}]
    edges = [_edge("a", "a", "calls"), _edge("a", "a", "calls")]
    run(nodes, edges)
    assert len(edges) == 2  # untouched, not even deduped


def test_merges_copies_keeping_definition_and_union():
    nodes = [
        _table("silver.orders", uc_catalogs=["dev"]),
        _table("silver.orders", source_file="w.py", source_location="L3",
               uc_catalogs=["{catalog}"], uc_kind="table"),
        _table("silver.orders", uc_catalogs=["prod", "dev"]),
    ]
    run(nodes, [])
    assert len(nodes) == 1
    n = nodes[0]
    assert n["source_file"] == "w.py" and n["source_location"] == "L3"
    assert n["uc_catalogs"] == ["dev", "{catalog}", "prod"]
    assert n["uc_kind"] == "table"


def test_read_and_write_by_one_symbol_fold_into_one_edge():
    t = table_node_id("silver.x")
    nodes = [_table("silver.x"), {"id": "f", "label": "f()"}]
    edges = [_edge("f", t, "reads_from"), _edge("f", t, "writes_to"), _edge("f", t, "writes_to")]
    run(nodes, edges)
    assert len(edges) == 1
    assert edges[0]["relation"] == "writes_to" and edges[0]["also_reads"] is True


def test_dlt_names_qualified_by_pipeline_schema():
    nodes = [
        {"id": "pipe", "label": "p", "type": "dab_pipeline", "uc_schema_norm": "silver",
         "uc_catalog_norm": "{catalog}", "source_file": "resources/p.yml"},
        {"id": "file_dlt", "label": "dlt.py", "source_file": "src/dlt.py"},
        {"id": "file_other", "label": "other.py", "source_file": "src/other.py"},
        _table("orders", source_file="src/dlt.py", uc_kind="table"),
        _table("silver.orders", uc_catalogs=["{catalog}"]),
    ]
    edges = [
        _edge("pipe", "file_dlt", "pipeline_source", sf="resources/p.yml"),
        _edge("fn", table_node_id("orders"), "writes_to", sf="src/dlt.py", context="defines"),
        _edge("reader", table_node_id("silver.orders"), "reads_from", sf="src/other.py"),
        # A bare name outside any pipeline file stays bare.
        _edge("other", table_node_id("orders"), "reads_from", sf="src/other.py"),
    ]
    run(nodes, edges)
    by_label = {n["label"]: n for n in nodes}
    q = by_label["silver.orders"]
    assert q["source_file"] == "src/dlt.py" and q["uc_kind"] == "table"
    assert {e["source"] for e in edges if e["target"] == q["id"]} == {"fn", "reader"}
    assert "orders" in by_label  # still referenced from src/other.py


def test_keep_mode_folds_env_catalogs_into_placeholder():
    set_config_override(DatabricksConfig(catalog_identity="keep"))
    nodes = [
        {"id": "v", "label": "catalog", "type": "dab_variable", "dab_default": "dev",
         "dab_env_values": {"prod": "prod"}},
        _table("dev.silver.orders", uc_catalogs=["dev"]),
        _table("{catalog}.silver.orders", uc_catalogs=["{catalog}"]),
        _table("finance.silver.orders", uc_catalogs=["finance"]),
    ]
    edges = [_edge("a", table_node_id("dev.silver.orders"), "reads_from")]
    run(nodes, edges)
    labels = sorted(n["label"] for n in nodes if n.get("type") == "uc_table")
    assert labels == ["finance.silver.orders", "{catalog}.silver.orders"]
    assert edges[0]["target"] == table_node_id("{catalog}.silver.orders")


def test_env_catalogs_annotated_from_bundle_and_config():
    set_config_override(DatabricksConfig(env_catalogs=frozenset({"staging"})))
    nodes = [
        {"id": "v", "label": "main_catalog", "type": "dab_variable", "dab_default": "dev"},
        _table("s.t", uc_catalogs=["dev", "staging", "{catalog}", "finance"]),
    ]
    run(nodes, [])
    assert nodes[1]["uc_env_catalogs"] == ["dev", "staging", "{catalog}"]


def test_config_parsing_and_fingerprint():
    cfg = parse_config("""
        # comment
        databricks=on
        databricks.catalog_identity=keep
        databricks.readers= read_table , Load_Bronze
        databricks.env_catalogs=dev,prod
        databricks.catalog_identity=bogus
        unrelated=1
    """)
    assert cfg.mode == "on" and cfg.catalog_identity == "keep"
    assert cfg.readers == frozenset({"read_table", "load_bronze"})
    assert cfg.fingerprint() and DatabricksConfig().fingerprint() == ""


def test_cache_salt_tracks_config_and_run_targets(tmp_path: Path):
    nb = tmp_path / "nb.py"
    nb.write_text("# Databricks notebook source\n# MAGIC %run ./helpers\n")
    plain = tmp_path / "plain.py"
    plain.write_text("x = 1\n")
    assert cache_salt(plain, tmp_path) is None
    before = cache_salt(nb, tmp_path)
    (tmp_path / "helpers.py").write_text("def h():\n    pass\n")
    after = cache_salt(nb, tmp_path)
    assert before and after and before != after
    (tmp_path / ".graphifyrc").write_text("databricks.readers=read_table\n")
    assert "databricks=" in (cache_salt(plain, tmp_path) or "")
