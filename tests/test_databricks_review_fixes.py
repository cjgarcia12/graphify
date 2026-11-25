"""Regressions for the PR #1 review findings (Databricks fork)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from graphify.extractors.base import set_active_scan_root
from graphify.extractors.databricks import config as db_config
from graphify.extractors.databricks.names import TableRef, emit_table_refs, table_node_id


def test_distinct_table_keys_never_share_an_id():
    """make_id folds '.' and '_' together; uc ids must keep a_b.c and a.b_c apart."""
    assert table_node_id("a_b.c") != table_node_id("a.b_c")
    assert table_node_id("silver.orders") != table_node_id("silver._orders")
    assert table_node_id("silver.orders") == table_node_id("silver.orders")

    nodes: list[dict] = []
    edges: list[dict] = []
    emit_table_refs(
        [TableRef("a_b.c", "reads_from", 1), TableRef("a.b_c", "writes_to", 2)],
        file_nid="f", file_path="nb.py", nodes=nodes, edges=edges, keep_catalog=False,
    )
    assert sorted(n["label"] for n in nodes) == ["a.b_c", "a_b.c"]


def test_complex_bundle_variables_never_leak_nested_secrets(tmp_path: Path):
    pytest.importorskip("yaml")
    from graphify.extractors.databricks import bundle

    (tmp_path / "databricks.yml").write_text(textwrap.dedent("""
        bundle:
          name: b
        variables:
          connection:
            type: complex
            default:
              host_alias: warehouse
              client_secret: fake-secret-value
          cluster:
            type: complex
            default:
              node_type_id: i3.xlarge
        targets:
          prod:
            variables:
              connection:
                default:
                  nested:
                    password: fake-password-value
    """), encoding="utf-8")
    previous = set_active_scan_root(tmp_path.resolve())
    try:
        result = bundle.extract(tmp_path / "databricks.yml")
    finally:
        set_active_scan_root(previous)

    text = json.dumps(result)
    assert "fake-secret-value" not in text
    assert "fake-password-value" not in text
    # A complex variable with nothing secret inside is still recorded.
    cluster = next(n for n in result["nodes"] if n.get("label") == "cluster")
    assert cluster.get("dab_default") == {"node_type_id": "i3.xlarge"}


def test_fresh_worker_routes_with_project_config(tmp_path: Path):
    """A spawn worker starts with no activated root; routing must still see the
    project's .graphifyrc (here: a team reader wrapper that is the only signal)."""
    from graphify.extract import _extract_single_file

    (tmp_path / ".graphifyrc").write_text("databricks.readers=load_bronze\n", encoding="utf-8")
    src = tmp_path / "job.py"
    src.write_text('def run():\n    return load_bronze("silver.orders")\n', encoding="utf-8")

    # Simulate a freshly spawned worker: no activated root, no active scan root.
    db_config.activate_root(None)
    previous = set_active_scan_root(None)
    try:
        _, result = _extract_single_file((0, str(src), str(tmp_path), str(tmp_path / "cache")))
    finally:
        set_active_scan_root(previous)
        db_config.activate_root(None)

    tables = [n for n in result["nodes"] if n.get("type") == "uc_table"]
    assert [n["label"] for n in tables] == ["silver.orders"]
    assert any(e["relation"] == "reads_from" and e["target"] == tables[0]["id"] for e in result["edges"])
