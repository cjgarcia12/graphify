"""Databricks Asset Bundle extraction (graphify/extractors/databricks/bundle.py)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from graphify.extractors.base import set_active_scan_root  # noqa: E402
from graphify.extractors.databricks import bundle  # noqa: E402
from graphify.ids import make_id  # noqa: E402


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _write(tmp_path, "databricks.yml", """
        bundle:
          name: sales_lakehouse
        include:
          - resources/*.yml
        variables:
          catalog:
            description: Target catalog
            default: dev
          warehouse_id:
            lookup:
              warehouse: Shared
          api_token:
            default: not-a-real-token-value
        workspace:
          host: https://adb-123.azuredatabricks.net
        targets:
          dev:
            mode: development
            default: true
            variables:
              catalog: dev
          prod:
            mode: production
            workspace:
              host: https://adb-999.azuredatabricks.net
            variables:
              catalog: prod
            resources:
              jobs:
                daily_sales:
                  tasks:
                    - task_key: prod_only_audit
                      depends_on:
                        - task_key: ingest
                      notebook_task:
                        notebook_path: ./src/audit.py
    """)
    _write(tmp_path, "resources/jobs.yml", """
        resources:
          jobs:
            daily_sales:
              name: daily-sales-${bundle.target}
              schedule:
                quartz_cron_expression: "0 0 6 * * ?"
              parameters:
                - name: catalog
                  default: ${var.catalog}
              tasks:
                - task_key: ingest
                  notebook_task:
                    notebook_path: ../src/ingest
                    base_parameters:
                      catalog: ${var.catalog}
                - task_key: transform
                  depends_on:
                    - task_key: ingest
                  pipeline_task:
                    pipeline_id: ${resources.pipelines.sales_dlt.id}
                - task_key: publish
                  depends_on:
                    - task_key: transform
                  python_wheel_task:
                    package_name: sales_lib
                    entry_point: publish
                  libraries:
                    - whl: ../dist/*.whl
                    - pypi:
                        package: requests>=2.31
                - task_key: missing_nb
                  notebook_task:
                    notebook_path: ../src/does_not_exist
    """)
    _write(tmp_path, "resources/pipelines.yml", """
        resources:
          schemas:
            silver:
              catalog_name: ${var.catalog}
              name: silver
          pipelines:
            sales_dlt:
              name: sales-dlt
              catalog: ${var.catalog}
              schema: ${resources.schemas.silver.name}
              serverless: true
              libraries:
                - glob:
                    include: ../src/dlt/**
                - notebook:
                    path: ../src/dlt_extra
    """)
    _write(tmp_path, "src/ingest.py", "# Databricks notebook source\nprint('ingest')\n")
    _write(tmp_path, "src/audit.py", "# Databricks notebook source\nprint('audit')\n")
    _write(tmp_path, "src/dlt/bronze.py", "import dlt\n")
    _write(tmp_path, "src/dlt/nested/silver.sql", "SELECT 1\n")
    _write(tmp_path, "src/dlt_extra.ipynb", '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}\n')
    return tmp_path


@pytest.fixture(autouse=True)
def _scan_root(tmp_path: Path):
    prev = set_active_scan_root(tmp_path.resolve())
    yield
    set_active_scan_root(prev)


def _by_id(result: dict) -> dict[str, dict]:
    return {n["id"]: n for n in result["nodes"]}


def _edges(result: dict, relation: str) -> set[tuple[str, str]]:
    return {(e["source"], e["target"]) for e in result["edges"] if e["relation"] == relation}


def _fid(p: Path) -> str:
    return make_id(str(p))


# ── ownership ────────────────────────────────────────────────────────────────


def test_owns_bundle_files(repo: Path):
    assert bundle.owns(repo / "databricks.yml")
    assert bundle.owns(repo / "resources/jobs.yml")
    assert bundle.owns(repo / "resources/pipelines.yml")


@pytest.mark.parametrize("rel,body", [
    (".github/workflows/ci.yml", "on: [push]\njobs:\n  build:\n    runs-on: ubuntu-latest\n    steps: []\n"),
    ("docker-compose.yml", "services:\n  web:\n    image: nginx\n"),
    ("k8s/deploy.yaml", "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: x\n"),
    ("chart/values.yaml", "resources:\n  limits:\n    cpu: 100m\n"),
    ("serverless.yml", "service: x\nresources:\n  Resources: {}\n"),
    ("cfn.yaml", "Resources:\n  Bucket:\n    Type: AWS::S3::Bucket\n"),
])
def test_does_not_own_other_yaml(tmp_path: Path, rel: str, body: str):
    assert not bundle.owns(_write(tmp_path, rel, body))


def test_route_hook_sends_bundle_yaml_to_bundle_extractor(repo: Path):
    from graphify.extractors.databricks import route

    assert route(repo / "resources/jobs.yml") is bundle.extract
    assert route(_write(repo, "docker-compose.yml", "services: {}\n")) is None


# ── per-file extraction ──────────────────────────────────────────────────────


def test_root_bundle_file(repo: Path):
    path = repo / "databricks.yml"
    res = bundle.extract(path)
    nodes = _by_id(res)
    bnid = make_id("dab_bundle", "sales_lakehouse")
    assert nodes[bnid]["type"] == "dab_bundle"
    assert nodes[bnid]["source_location"] == "L1"
    assert (bnid, _fid(repo / "resources/jobs.yml")) in _edges(res, "includes")
    assert (bnid, _fid(repo / "resources/pipelines.yml")) in _edges(res, "includes")

    var = nodes[make_id("dab_variable", "catalog")]
    assert var["dab_default"] == "dev"
    assert var["dab_env_values"] == {"dev": "dev", "prod": "prod"}
    assert nodes[make_id("dab_variable", "warehouse_id")]["dab_lookup"] == "warehouse"
    # secret-shaped variable keeps its node but never its value
    assert "dab_default" not in nodes[make_id("dab_variable", "api_token")]

    dev = nodes[make_id("dab_target", "dev")]
    assert dev["dab_mode"] == "development" and dev["dab_default"] is True
    assert (bnid, make_id("dab_target", "prod")) in _edges(res, "targets")

    # The prod-only task appears, marked with its target, and links its notebook.
    task = nodes[make_id("dab_task", "daily_sales", "prod_only_audit")]
    assert task["targets"] == ["prod"]
    assert (task["id"], _fid(repo / "src/audit.py")) in _edges(res, "runs_notebook")
    assert (task["id"], make_id("dab_task", "daily_sales", "ingest")) in _edges(res, "depends_on")
    # The job is declared in jobs.yml: only overridden here, so its node from
    # this file is the override's declaration (merges by id corpus-wide).
    assert nodes[make_id("dab_job", "daily_sales")]["targets"] == ["prod"]

    text = repr(res)
    assert "adb-123" not in text and "adb-999" not in text and "not-a-real-token-value" not in text


def test_jobs_file(repo: Path):
    path = repo / "resources/jobs.yml"
    res = bundle.extract(path)
    nodes = _by_id(res)
    jnid = make_id("dab_job", "daily_sales")
    job = nodes[jnid]
    assert job["type"] == "dab_job" and job["source_location"] == "L3"
    assert job["dab_schedule"] == "0 0 6 * * ?"
    assert job["dab_parameters"] == ["catalog"]

    t = {k: make_id("dab_task", "daily_sales", k) for k in ("ingest", "transform", "publish", "missing_nb")}
    assert {(jnid, v) for v in t.values()} <= _edges(res, "has_task")
    assert _edges(res, "depends_on") == {(t["transform"], t["ingest"]), (t["publish"], t["transform"])}

    assert (t["ingest"], _fid(repo / "src/ingest.py")) in _edges(res, "runs_notebook")
    assert nodes[t["ingest"]]["dab_parameters"] == ["catalog"]
    assert nodes[t["ingest"]]["dab_task_type"] == "notebook_task"
    # pipeline declared in another file: edge to its mergeable id, no stub node
    pnid = make_id("dab_pipeline", "sales_dlt")
    assert (t["transform"], pnid) in _edges(res, "runs_pipeline")
    assert pnid not in nodes
    assert (t["publish"], make_id("pkg", "sales_lib")) in _edges(res, "runs_package")
    assert (t["publish"], make_id("pkg", "requests")) in _edges(res, "uses_library")
    assert nodes[t["publish"]]["dab_entry_point"] == "publish"
    # unresolved notebook: raw path kept, no edge
    assert nodes[t["missing_nb"]]["dab_notebook_path"] == "../src/does_not_exist"
    assert not [e for e in res["edges"] if e["source"] == t["missing_nb"] and e["relation"] == "runs_notebook"]
    # ${var.catalog} -> variable edge (variable declared elsewhere: edge only)
    assert (t["ingest"], make_id("dab_variable", "catalog")) in _edges(res, "uses_variable")
    assert make_id("dab_variable", "catalog") not in nodes


def test_pipelines_file(repo: Path):
    res = bundle.extract(repo / "resources/pipelines.yml")
    nodes = _by_id(res)
    pnid = make_id("dab_pipeline", "sales_dlt")
    p = nodes[pnid]
    assert p["uc_catalog"] == "${var.catalog}" and p["uc_catalog_norm"] == "{catalog}"
    assert p["uc_schema_norm"] == "silver"
    assert p["dab_serverless"] is True
    sources = {t for s, t in _edges(res, "pipeline_source") if s == pnid}
    assert sources == {
        _fid(repo / "src/dlt/bronze.py"),
        _fid(repo / "src/dlt/nested/silver.sql"),
        _fid(repo / "src/dlt_extra.ipynb"),
    }
    snid = make_id("uc_schema", "silver")
    assert nodes[snid]["type"] == "uc_schema" and nodes[snid]["uc_catalogs"] == ["{catalog}"]
    assert (pnid, snid) in _edges(res, "references")


def test_anchors_and_merge_keys(tmp_path: Path):
    _write(tmp_path, "src/a.py", "x = 1\n")
    path = _write(tmp_path, "resources/anchors.yml", """
        resources:
          jobs:
            j:
              tasks:
                - &base
                  task_key: a
                  notebook_task:
                    notebook_path: ../src/a
                - <<: *base
                  task_key: b
                  depends_on: [{task_key: a}]
    """)
    res = bundle.extract(path)
    a, b = make_id("dab_task", "j", "a"), make_id("dab_task", "j", "b")
    assert (b, a) in _edges(res, "depends_on")
    assert {(a, _fid(tmp_path / "src/a.py")), (b, _fid(tmp_path / "src/a.py"))} <= _edges(res, "runs_notebook")


def test_for_each_and_run_job(tmp_path: Path):
    path = _write(tmp_path, "resources/loop.yml", """
        resources:
          jobs:
            fanout:
              tasks:
                - task_key: each
                  for_each_task:
                    inputs: "[1, 2]"
                    task:
                      task_key: each_iteration
                      run_job_task:
                        job_id: ${resources.jobs.child.id}
            child:
              tasks:
                - task_key: c
                  condition_task: {op: EQUAL_TO, left: "1", right: "1"}
    """)
    res = bundle.extract(path)
    outer = make_id("dab_task", "fanout", "each")
    inner = make_id("dab_task", "fanout", "each_iteration")
    assert (outer, inner) in _edges(res, "has_task")
    assert (inner, make_id("dab_job", "child")) in _edges(res, "runs_job")
    assert _by_id(res)[make_id("dab_task", "child", "c")]["dab_condition"] == "1 EQUAL_TO 1"


def test_invalid_yaml_returns_error(tmp_path: Path):
    path = _write(tmp_path, "databricks.yml", "bundle: [unclosed\n")
    res = bundle.extract(path)
    assert res["nodes"] == [] and "error" in res


def test_unknown_tags_do_not_fail(tmp_path: Path):
    path = _write(tmp_path, "databricks.yml", "bundle:\n  name: !custom x\n")
    assert make_id("dab_bundle", "x") in _by_id(bundle.extract(path))


# ── end to end through graphify.extract ──────────────────────────────────────


def test_end_to_end_graph(repo: Path):
    from graphify.build import build_from_json
    from graphify.extract import collect_files, extract

    result = extract(collect_files(repo), root=repo)
    G = build_from_json(result, directed=True, root=repo)
    jnid = make_id("dab_job", "daily_sales")
    jobs = [n for n, d in G.nodes(data=True) if d.get("type") == "dab_job" and n.startswith("dab_job_daily_sales")]
    assert jobs == [jnid], jobs  # one job node across jobs.yml + databricks.yml override

    def has_edge(u: str, v: str, relation: str) -> bool:
        return G.has_edge(u, v) and G.edges[u, v].get("relation") == relation

    ingest = make_id("dab_task", "daily_sales", "ingest")
    transform = make_id("dab_task", "daily_sales", "transform")
    publish = make_id("dab_task", "daily_sales", "publish")
    assert has_edge(transform, ingest, "depends_on")
    assert has_edge(publish, transform, "depends_on")
    assert has_edge(jnid, ingest, "has_task")
    assert has_edge(transform, make_id("dab_pipeline", "sales_dlt"), "runs_pipeline")

    # task -> notebook lands on the real notebook file node in the final graph
    nb_targets = [v for _, v, d in G.out_edges(ingest, data=True) if d.get("relation") == "runs_notebook"]
    assert len(nb_targets) == 1
    nb = G.nodes[nb_targets[0]]
    assert nb.get("source_file", "").replace("\\", "/").endswith("src/ingest.py")

    pipe_sources = {
        G.nodes[v].get("source_file", "").replace("\\", "/")
        for _, v, d in G.out_edges(make_id("dab_pipeline", "sales_dlt"), data=True)
        if d.get("relation") == "pipeline_source"
    }
    assert any(s.endswith("src/dlt/bronze.py") for s in pipe_sources)

    # the variable declared in databricks.yml and used from jobs.yml is one node
    var = make_id("dab_variable", "catalog")
    assert G.nodes[var].get("source_file", "").endswith("databricks.yml")
    assert has_edge(ingest, var, "uses_variable")
