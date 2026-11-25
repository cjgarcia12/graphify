# Databricks support

This fork of graphify understands Databricks repositories: Asset Bundles (DABs),
Unity Catalog tables, notebooks and DLT / Lakeflow pipelines. It is graphed as one
connected system:

```
bundle -> job -> task -> notebook / file / pipeline / wheel -> function
                                   |
                       reads_from / writes_to
                                   v
                      Unity Catalog table <- every other reader and writer
```

Nothing extra to install or turn on. Run graphify as usual (`graphify update .`
for code-only, or `/graphify .` in your assistant). Databricks files are
detected automatically, and everything below can be tuned per repo in
`.graphifyrc`.

## What is extracted

### Notebooks

| Format | Detected by |
|---|---|
| Source `.py` / `.r` | first line `# Databricks notebook source` |
| Source `.sql` | first line `-- Databricks notebook source` |
| Source `.scala` | first line `// Databricks notebook source` |
| `.ipynb` | any Jupyter notebook (Databricks-exported or plain) |

- Cells are split on `COMMAND ----------`. `%sql`, `%python`, `%md`, `%run`, `%pip`
  and other magics are honored, and so is the Databricks cell-language metadata in `.ipynb`.
- Python cells are parsed by graphify's normal Python extractor, so functions,
  classes, imports and calls work exactly as in a `.py` module. Line numbers point
  at the real file. In `.ipynb`, nodes also carry `cell` (0-based cell index).
- `%run ./other` creates a `runs_notebook` edge. When the target is a `.py` file, it
  is also treated like `from other import *`, so calls into functions it defines
  resolve across notebooks.
- `dbutils.notebook.run("./child", ...)` creates a `runs_notebook` edge from the
  calling function.
- `dbutils.widgets.text|dropdown|...` and SQL `CREATE WIDGET` are recorded as
  `db_widgets` on the notebook node.
- `%md` headings become `notebook_section` nodes.
- `.ipynb` outputs are ignored and never counted or parsed.

### Unity Catalog table lineage

Every table becomes one `uc_table` node for the whole repo, no matter which file,
language or environment mentions it. The edges are:
- `writes_to`: from the writing function, or the notebook/file when the write is at top level.
  `context=defines` marks the definition, which also supplies the node's `source_file`.
- `reads_from`: from the reading function, or the notebook/file.
- `also_reads=true`: set on the `writes_to` edge when the same symbol both reads and writes.

| Source | Patterns |
|---|---|
| PySpark | `spark.table`, `spark.read[Stream].table`, `.saveAsTable`, `.toTable`, `.insertInto`, `.writeTo`, `DeltaTable.forName` (a write when followed by `merge`/`update`/`delete`), `spark.sql("...")` |
| DLT / Lakeflow | `@dlt.table` / `@dlt.view` / `@dlt.materialized_view`, `from pyspark import pipelines as dp` (`@dp.table`, `@dp.materialized_view`, ...), `dlt.read[_stream]`, `create_streaming_table`, `apply_changes` / `create_auto_cdc_flow`, `append_flow` (any import alias) |
| Databricks SQL | `CREATE [OR REPLACE / OR REFRESH] [STREAMING] TABLE`, `[MATERIALIZED] VIEW`, CTAS, `INSERT INTO/OVERWRITE`, `MERGE INTO ... USING`, `UPDATE`, `DELETE`, `COPY INTO`, `FROM`/`JOIN`, `STREAM(x)`, `LIVE.x`, `IDENTIFIER(...)`, `USE CATALOG/SCHEMA`, in `.sql` files, `%sql` cells and `spark.sql` strings |
| Your own helpers | anything listed in `databricks.readers` / `databricks.writers` (see below) |

Table names are evaluated from string literals, f-strings, `.format`, `%` and `+`,
and from constants in the same file. Values that come from outside the code are
kept as placeholders:
- `dbutils.widgets.get("catalog")` becomes `{catalog}`;
- `spark.conf.get`, task values, `os.environ` and `sys.argv` are handled the same way;
- `${var.catalog}` becomes `{catalog}`;
- SQL `:catalog`, `{{catalog}}` and `$catalog` also become `{catalog}`.

A name that can't be evaluated is skipped rather than guessed.

**Identity.** By default a table is identified by `schema.table`. The catalog is
usually an environment switch, so `${var.catalog}.silver.orders`,
`{catalog}.silver.orders` and `dev.silver.orders` are one node. The catalogs
actually used are listed in `uc_catalogs`, and the subset that are environment
switches is listed in `uc_env_catalogs`. Graphify learns those from bundle
variables named `*catalog*` across targets.

**DLT datasets.** A one-part name (`@dlt.table def orders_clean`) in a file that a
bundle pipeline lists under `libraries` is qualified with that pipeline's schema
(`silver.orders_clean`). It is the same node a downstream notebook reads as
`{catalog}.silver.orders_clean`.

### Asset Bundles

`databricks.yml` and every YAML file it `include`s (any file whose top level is
bundle `resources:` / `targets:`) produce the following. GitHub workflows,
compose, k8s and Helm files are left to the generic YAML extractor.

| Node type | What |
|---|---|
| `dab_bundle` | the bundle; `includes` edges to resource files |
| `dab_variable` | variables, with `dab_default` and per-target `dab_env_values` |
| `dab_target` | targets (mode, variable overrides); resources overridden per target |
| `dab_job` / `dab_task` | jobs and their tasks (`has_task`); `depends_on` from a task to the task it waits on |
| `dab_pipeline` | pipelines with `uc_catalog` / `uc_schema`; `pipeline_source` to each library file (globs included) |
| `uc_schema` / `uc_volume` / `dab_resource` | schemas, volumes, and every other resource kind |

Task edges:
- `runs_notebook`: to notebooks;
- `runs_file`: to Python and SQL files;
- `runs_pipeline` / `runs_job`: to the pipeline or job a task triggers;
- `runs_package`: to the wheel's package;
- `runs_entry_point`: to the function the wheel's entry point names, found through
  `[project.scripts]` / entry points in `pyproject.toml`;
- `uses_variable`: to bundle variables;
- `references`: to other resources named by `${resources...}`.

Hosts, tokens, `run_as`, permissions and secret-shaped values are never emitted.

### Report

`GRAPH_REPORT.md` gets a **Data lineage (Databricks)** section with two parts:
- the busiest tables, shown as `writers -> table -> readers`;
- every job's tasks in run order.

## Tailoring it to your repo (`.graphifyrc`)

Put any of these keys in `<repo>/.graphifyrc`, one per line. All are optional.

```ini
# auto (default): notebooks and bundles by header/filename; table lineage in any
# .py mentioning spark/pyspark/dlt/dbutils/DeltaTable or one of your helpers.
# on: scan every .py for lineage. off: disable all Databricks handling.
databricks=auto

# Your team's wrappers around spark.table / saveAsTable. The table name is the
# first string argument or a table=/table_name=/name=/target= keyword. Matched
# on the call's last segment, so read_table also matches utils.io.read_table(...).
databricks.readers=read_table,load_bronze,read_delta
databricks.writers=write_table,upsert_to_delta,merge_into

# Catalog names that are environment switches. Merged with the values your
# bundle's *catalog* variables take per target.
databricks.env_catalogs=dev,staging,prod

# drop (default): identity is schema.table. keep: catalog.schema.table, for
# lakehouses where the same schema.table exists in genuinely different catalogs.
# Under keep, env catalogs above still fold into the {catalog} placeholder form.
databricks.catalog_identity=drop

# Schemas that never become nodes (information_schema is always skipped).
databricks.ignore_schemas=tmp,scratch
```

Changing these keys invalidates the cache for the affected files automatically.

## Checking the result

```bash
graphify update .
graphify query "who writes silver.orders"
graphify path "sales_daily.ingest_orders" "gold.daily_revenue"
graphify explain "silver.orders_clean"
```

Or open `graphify-out/GRAPH_REPORT.md` and look for **Data lineage (Databricks)**.
If a table you expect is missing, the name was probably built somewhere graphify
can't evaluate. Common causes are a helper in another module, or a value passed
in from a config object. Add the helper to `databricks.readers` / `databricks.writers`,
or pass a literal or f-string to it.

## Known limits

- Temp views (`createOrReplaceTempView`) and path-based reads (`spark.read.load(path)`) are not tables.
- A table name assembled by a function in another module is not evaluated.
- `%run` of an `.ipynb` or of a path that is not a Python identifier gets the
  `runs_notebook` edge, but calls into it fall back to name matching.
- Scala/R notebook cells contribute the notebook node and its links, not symbols.
- Without the optional `tree-sitter-sql` grammar (`pip install graphifyy[sql]`),
  `.sql` files get table lineage but not their routine/procedure symbols.
