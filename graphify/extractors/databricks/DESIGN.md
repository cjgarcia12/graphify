# Databricks support: design and contracts

Goal: a Databricks repo (Asset Bundles, Unity Catalog, notebooks, DLT/Lakeflow
pipelines) graphs as one connected system: bundle -> job -> task -> notebook ->
tables <- other notebooks/pipelines. Must work for *any* team's repo: auto-detect
first, `.graphifyrc` `databricks.*` keys (see `config.py`) for everything else.

Fork hygiene: keep core edits minimal and marked `# Databricks fork:` so upstream
merges stay easy. Current core touch points:

- `graphify/extract.py`: `_get_extractor` calls `databricks.route(path)`;
  `_cache_salt` adds `databricks.cache_salt` (config fingerprint, resolved `%run`
  targets) at the four AST-cache load/save sites; `extract()` calls
  `config.activate_root(root)`; `extract_python` accepts `source=` bytes and
  `collect_files` includes `.ipynb`.
- `graphify/detect.py`: `.ipynb` is code (word count from cell sources only),
  `.ipynb_checkpoints` skipped.
- `graphify/extractors/resolution.py`: `MERGEABLE_NODE_TYPES` skip id disambiguation.
- `graphify/build.py`: `build()` (before dedup) and `build_from_json` run
  `resolve.run` — every full, incremental and watch rebuild passes through one of them.
- `graphify/report.py`: appends `report_section.lineage_section(G)`.
- `graphify/cache.py`: `_AST_CACHE_SCHEMA` 56.

User-facing documentation: `docs/databricks.md`.

## Modules and owners

| Module | Public API |
|---|---|
| `names.py` | `parse_table_name`, `table_key`, `table_node_id`, `TableRef`, `emit_table_refs`, `MERGEABLE_NODE_TYPES` |
| `config.py` | `get_config() -> DatabricksConfig` (`.readers`, `.writers`, `.env_catalogs`, `.catalog_identity`, `.mode`) |
| `lineage.py` | `owns(path)`, `extract(path)`, `scan_python(source, *, cfg=None) -> list[TableRef]`, `scan_sql(text, *, default_schema=None) -> list[TableRef]`, `attach(refs, result, path)` |
| `notebook.py` | `owns(path)`, `extract(path)`, `split_cells(path) -> list[Cell]` |
| `bundle.py` | `owns(path)`, `extract(path)` |
| `paths.py` | `resolve_workspace_path(ref, base_dir) -> Path | None`, `file_ref_id(path) -> str` |
| `resolve.py` | `run(nodes, edges, root)` — corpus pass (below) |
| `report_section.py` | `lineage_section(G) -> list[str]` |

Every `extract(path)` takes only `path` (the core calls `extractor(path)`), reads
the scan root from `graphify.extractors.base.active_scan_root()`, and returns the
standard `{"nodes", "edges", ...}` dict. For `.py`/`.sql` it must **augment** the
core extractor's result (`extract_python(path, root=active_scan_root())`,
`extract_sql(path, content=...)`), never replace it, so every Python/SQL
post-pass (raw_calls, imports, resolution) keeps working.

## Shared conventions

- **Line numbers** are 1-based lines of the real file (`source_location="L<n>"`).
  `.ipynb`: lines of the concatenated "virtual source" (see `notebook.py`), and
  nodes/edges additionally carry `cell` (0-based code-cell index).
- **Tables**: always via `names.emit_table_refs` -> one `uc_table` node per
  `schema.table` corpus-wide; `reads_from` / `writes_to` (`context="defines"` for
  CREATE / saveAsTable / @dlt.table) from the enclosing function node when the
  ref is inside one, else from the file node.
- **Placeholders**: catalog/schema written as `${var.x}`, `{x}`, widget values,
  `:param`, `{{x}}`, `IDENTIFIER(:x)` normalize to `{x}` (`names.normalize_placeholders`).
- **File references** (`%run`, `dbutils.notebook.run`, bundle `notebook_path`,
  `python_file`, pipeline `libraries`): target id is `_make_id(str(resolved_path))`
  exactly as `extractors/bash.py` does for `source`, where `resolved_path` has the
  same absolute/relative form as the `path` the extractor was given. Workspace
  paths may omit the extension (try `.py`, `.ipynb`, `.sql`, `.scala`, `.r`) and
  may be prefixed by `${workspace.file_path}` / `/Workspace/...` (strip, then
  resolve relative to the referencing file's directory, then the bundle root).
- **Relations** (new): `runs_notebook` (notebook/task -> notebook file),
  `runs_file` (task -> python/sql file), `has_task` (job -> task),
  `depends_on` (task -> task), `runs_pipeline` / `runs_job` (task -> pipeline/job),
  `pipeline_source` (pipeline -> library file), `uses_variable` (resource -> variable),
  `includes` (bundle -> included YAML file), `targets` (bundle -> target).
- **Bundle entity ids** (mergeable types): `make_id("dab_job", job_key)`,
  `make_id("dab_task", job_key, task_key)`, `make_id("dab_pipeline", key)`,
  `make_id("dab_variable", name)`, `make_id("dab_target", name)`,
  `make_id("dab_bundle", name)`; `source_file` set by the file that *declares*
  it, else `""` with `origin_file`.
- **Never emit secrets**: no `host`, tokens, `spark_env_vars` values, or
  `client_secret`-like keys. Variable values only when they are not
  secret-shaped (reuse `graphify.security` redaction if available).

## Corpus pass (`resolve.py`)

Runs on the combined extraction inside `build()` / `build_from_json`;
idempotent, a no-op without Databricks nodes. In order:

1. Merge same-id mergeable nodes (definition's `source_file` wins, list/dict
   attributes unioned).
2. Learn env catalogs (bundle `*catalog*` variables' default + per-target values,
   plus `databricks.env_catalogs`); under `catalog_identity=keep` re-key
   `<env>.s.t` onto `{var}.s.t`.
3. Late-bind bundle paths: tasks' `dab_path_refs` and pipelines'
   `dab_library_refs` (scan-root-relative paths/globs) are matched against file
   nodes present in this build, so a cached bundle YAML still links notebooks
   added later.
4. Qualify one-part tables referenced from pipeline library files with the
   pipeline's schema (and catalog under `keep`).
5. Annotate `uc_env_catalogs`.
6. Wheel tasks: package node (from a matching `pyproject.toml` if no manifest
   node exists) and `runs_entry_point` via `[project.scripts]` / entry points,
   falling back to function `entry_point` in package `package_name`.
7. Drop duplicate edges; fold a symbol's read+write of one table into one
   `writes_to` edge with `also_reads=True` (the undirected graph keeps one edge
   per pair).

Edge direction: the built graph is undirected; `build_from_json` records
`_src`/`_tgt` on every edge and `graph.json` exports `source`/`target` in that
direction, so `depends_on` reads task -> the task it waits on.
