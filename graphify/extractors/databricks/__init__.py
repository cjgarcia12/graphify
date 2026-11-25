"""Databricks support: notebooks, Unity Catalog lineage, Asset Bundles.

Layout (each module owns one concern; see DESIGN.md):

``names``     Unity Catalog name normalization + the shared ``uc_table`` node contract.
``config``    Per-project tuning from ``.graphifyrc`` ``databricks.*`` keys.
``lineage``   Table reads/writes/definitions found in Python (PySpark, DLT/Lakeflow,
              spark.sql strings, team wrapper functions) and Databricks SQL text.
``notebook``  Source-format ``.py``/``.sql`` notebooks and ``.ipynb``: cell splitting,
              per-language delegation, ``%run`` / ``dbutils.notebook.run`` links.
``bundle``    Databricks Asset Bundles: ``databricks.yml`` + included resource YAML.
``resolve``   Corpus-level pass run after per-file extraction.

The core touches this package only through :func:`route` (extractor choice per
file) and the corpus pass, so the fork stays mergeable with upstream graphify.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def route(path: Path) -> Callable[..., dict] | None:
    """The Databricks extractor that owns ``path``, or None to fall through to
    graphify's normal dispatch. Must be cheap: it runs for every file.

    Each sub-module exposes an ``owns(path) -> bool`` sniff and an ``extract``
    entry point; a module that is not importable (still being built, or its
    optional dependency is missing) is simply skipped.
    """
    from graphify.extractors.databricks.config import get_config

    if not get_config().enabled:
        return None
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        return _probe("bundle", path)
    if suffix in (".py", ".sql", ".ipynb", ".scala", ".r"):
        # A notebook first; otherwise a plain Spark .py module or Databricks SQL
        # file still gets table lineage on top of the core extractor's output.
        return _probe("notebook", path) or (
            _probe("lineage", path) if suffix in (".py", ".sql") else None
        )
    return None


def _probe(module: str, path: Path) -> Any:
    import importlib

    try:
        mod = importlib.import_module(f"graphify.extractors.databricks.{module}")
    except ImportError:
        return None
    owns = getattr(mod, "owns", None)
    if owns is not None and owns(path):
        return mod.extract
    return None


_SALTED_SUFFIXES = (".py", ".sql", ".ipynb", ".yml", ".yaml", ".scala", ".r")


def cache_salt(path: Path, root: Path | None) -> str | None:
    """Extra AST-cache key material for a file this package may extract.

    * a non-default ``databricks.*`` config changes routing and naming, so its
      fingerprint salts every file the package could own;
    * a notebook's ``%run`` becomes a star import only when the target exists,
      so the sorted resolved targets salt the notebook: adding or deleting a
      target re-extracts it.
    """
    if path.suffix.lower() not in _SALTED_SUFFIXES:
        return None
    from graphify.extractors.databricks.config import load_config

    cfg = load_config(Path(root) if root is not None else None)
    parts: list[str] = []
    if cfg.fingerprint():
        parts.append(f"databricks={cfg.fingerprint()}")
    if cfg.enabled and path.suffix.lower() != ".yml" and path.suffix.lower() != ".yaml":
        runs = _run_salt(path, root)
        if runs:
            parts.append(f"runs={runs}")
    return ";".join(parts) or None


def _run_salt(path: Path, root: Path | None) -> str | None:
    try:
        with open(path, "rb") as fh:
            data = fh.read(4 * 1024 * 1024)
    except OSError:
        return None
    if b"%run" not in data:
        return None
    import hashlib

    from graphify.extractors.base import set_active_scan_root
    from graphify.extractors.databricks import notebook
    from graphify.extractors.databricks.paths import resolve_workspace_path

    previous = set_active_scan_root(Path(root).resolve() if root is not None else None)
    try:
        if not notebook.owns(path):
            return None
        try:
            cells = notebook.split_cells(path)
        except Exception:
            return None
        bundle_root = notebook._bundle_root(path)
        targets = []
        for cell in cells:
            if cell.language == "run":
                t = resolve_workspace_path(notebook._run_target(cell.magic_args or ""), path.parent,
                                           bundle_root=bundle_root)
                if t is not None and root is not None:
                    import os
                    t = os.path.relpath(t, root)  # portable across checkouts
                targets.append(str(t) if t is not None else "-")
    finally:
        set_active_scan_root(previous)
    return hashlib.sha1("\n".join(sorted(targets)).encode()).hexdigest()[:12] if targets else None
