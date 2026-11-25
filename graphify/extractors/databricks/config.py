"""Per-project Databricks tuning, read from ``<root>/.graphifyrc``.

Every repo wraps Spark differently, so the defaults are auto-detected and every
assumption can be overridden. Keys (all optional)::

    # auto: notebooks/bundles detected by header and filename; lineage scanned in
    # any .py that mentions spark/dlt/pyspark. on: scan every .py. off: disable.
    databricks=auto

    # drop (default): a table's identity is schema.table, catalogs recorded on the
    # node. keep: catalog.schema.table (multi-catalog lakehouses).
    databricks.catalog_identity=drop

    # Team helpers that take a table name as their first string argument (or a
    # table=/table_name=/name=/target= keyword). Bare or dotted names both match
    # the call's last segment: read_table matches utils.read_table(...).
    databricks.readers=read_table,load_bronze
    databricks.writers=write_table,upsert_to_delta

    # Catalog names that are environment switches, merged with those learned
    # from bundle variable values per target.
    databricks.env_catalogs=dev,staging,prod

    # Schemas never turned into nodes (information_schema always is).
    databricks.ignore_schemas=tmp,scratch

This module re-reads the rc file only when its mtime changes, so it is cheap to
call from inside per-file extractors (which run under the active scan root,
also in worker processes).
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from graphify.extractors.base import active_scan_root

RC_FILENAME = ".graphifyrc"


@dataclass(frozen=True)
class DatabricksConfig:
    mode: Literal["auto", "on", "off"] = "auto"
    catalog_identity: Literal["drop", "keep"] = "drop"
    readers: frozenset[str] = field(default_factory=frozenset)
    writers: frozenset[str] = field(default_factory=frozenset)
    env_catalogs: frozenset[str] = field(default_factory=frozenset)
    ignore_schemas: frozenset[str] = field(default_factory=frozenset)

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def fingerprint(self) -> str:
        """Stable digest of everything that changes extraction output, for the
        AST cache salt. The default config fingerprints to ``""``."""
        if self == DEFAULT_CONFIG:
            return ""
        parts = [self.mode, self.catalog_identity]
        for s in (self.readers, self.writers, self.env_catalogs, self.ignore_schemas):
            parts.append(",".join(sorted(s)))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


DEFAULT_CONFIG = DatabricksConfig()


def _csv(value: str) -> frozenset[str]:
    return frozenset(v.strip().lower() for v in value.split(",") if v.strip())


def parse_config(text: str) -> DatabricksConfig:
    """Parse the ``databricks*`` keys out of a .graphifyrc body. Invalid values
    fall back to the default for that key; other keys are ignored."""
    values: dict = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip() if not raw.lstrip().startswith("#") else ""
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip().lower(), val.strip()
        if key == "databricks" and val.lower() in ("auto", "on", "off"):
            values["mode"] = val.lower()
        elif key == "databricks.catalog_identity" and val.lower() in ("drop", "keep"):
            values["catalog_identity"] = val.lower()
        elif key in ("databricks.readers", "databricks.writers",
                     "databricks.env_catalogs", "databricks.ignore_schemas"):
            values[key.split(".", 1)[1]] = _csv(val)
    return DatabricksConfig(**values)


_lock = threading.Lock()
_cache: dict[str, tuple[float, DatabricksConfig]] = {}
_override: DatabricksConfig | None = None
# The root of the extraction in flight, for code that runs outside an
# extractor call (``_get_extractor`` routing in the parent process).
_activated_root: Path | None = None


def load_config(root: Path | None) -> DatabricksConfig:
    if root is None:
        return DEFAULT_CONFIG
    rc = Path(root) / RC_FILENAME
    try:
        mtime = rc.stat().st_mtime
    except OSError:
        return DEFAULT_CONFIG
    key = str(rc)
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] == mtime:
            return hit[1]
    try:
        cfg = parse_config(rc.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        cfg = DEFAULT_CONFIG
    with _lock:
        _cache[key] = (mtime, cfg)
    return cfg


def get_config() -> DatabricksConfig:
    """The config for the extraction in flight (the active scan root's rc), or
    the test override set by :func:`set_config_override`."""
    if _override is not None:
        return _override
    return load_config(active_scan_root() or _activated_root)


def activate_root(root: Path | None) -> None:
    """Record the scan root ``extract()`` is working under (mirrors
    ``rcfile.activate_language_overrides``)."""
    global _activated_root
    _activated_root = Path(root).resolve() if root is not None else None


def set_config_override(cfg: DatabricksConfig | None) -> None:
    """Tests only: force a config regardless of the scan root."""
    global _override
    _override = cfg
