"""Resolve Databricks workspace-style references to files in the scanned repo.

Notebook references (``%run ../utils/helpers``, ``dbutils.notebook.run("./ingest")``,
bundle ``notebook_path: ../src/ingest``) usually omit the file extension and may
carry a workspace prefix. They resolve to a real file in the corpus or to nothing:
an unresolvable reference never mints a phantom file node.
"""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import _make_id, active_scan_root

NOTEBOOK_EXTENSIONS = (".py", ".ipynb", ".sql", ".scala", ".r", ".R")

# ${workspace.file_path}/src/x, ${workspace.root_path}/files/src/x
_WORKSPACE_VAR_PREFIX_RX = re.compile(r"^\$\{\s*workspace\.(?:file_path|root_path)\s*\}(?:/files)?/?")


def _within(path: Path, root: Path | None) -> bool:
    if root is None:
        return True
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def resolve_workspace_path(
    ref: str, base_dir: Path, *, bundle_root: Path | None = None
) -> Path | None:
    """The corpus file ``ref`` names, or None.

    Tried in order: relative to ``base_dir`` (the referencing file's folder),
    then relative to ``bundle_root``. Each candidate is tried as written, then
    with each notebook extension. Absolute workspace paths (``/Workspace/...``,
    ``/Repos/...``, ``/Users/...``) cannot be mapped to the repo and yield None
    unless their tail exists under the bundle root. Results outside the scan
    root are rejected. The returned path keeps ``base_dir``'s absolute/relative
    form, so ``_make_id(str(result))`` matches the core file-node convention.
    """
    if not ref or not isinstance(ref, str):
        return None
    ref = ref.strip().strip("\"'")
    if not ref or "://" in ref or "${" in _WORKSPACE_VAR_PREFIX_RX.sub("", ref):
        return None
    ref = _WORKSPACE_VAR_PREFIX_RX.sub("", ref)
    scan_root = active_scan_root()
    bases: list[Path] = []
    if ref.startswith("/"):
        if bundle_root is None:
            return None
        # /Workspace/Users/me/.bundle/x/dev/files/src/nb -> try the tail after /files/
        tail = ref.split("/files/", 1)[1] if "/files/" in ref else None
        if not tail:
            return None
        ref = tail
        bases.append(bundle_root)
    else:
        bases.append(base_dir)
        if bundle_root is not None and bundle_root != base_dir:
            bases.append(bundle_root)
    for base in bases:
        candidate = base / ref
        options = [candidate] + [candidate.with_name(candidate.name + ext) for ext in NOTEBOOK_EXTENSIONS]
        for opt in options:
            try:
                if opt.is_file() and _within(opt, scan_root):
                    return Path(_normpath(opt))
            except OSError:
                continue
    return None


def _normpath(p: Path) -> str:
    """Collapse ``..`` without resolving symlinks or changing abs/rel form."""
    import os

    return os.path.normpath(str(p))


def file_ref_id(path: Path) -> str:
    """Node id of the file node for ``path`` (the core's ``_make_id(str(path))``)."""
    return _make_id(str(path))
