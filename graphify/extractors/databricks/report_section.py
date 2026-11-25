"""The "Data lineage (Databricks)" section of GRAPH_REPORT.md."""
from __future__ import annotations

from graphify.extractors.databricks.names import TABLE_NODE_TYPE


def _ends(G, u, v, data) -> tuple:
    """Edge endpoints in their recorded direction (undirected graphs keep it in _src/_tgt)."""
    return data.get("_src", u), data.get("_tgt", v)


def lineage_section(G, *, top_tables: int = 25, top_refs: int = 6) -> list[str]:
    """Markdown lines for the section, or ``[]`` when the graph has no Databricks nodes."""
    tables = {n for n, d in G.nodes(data=True) if d.get("type") == TABLE_NODE_TYPE}
    jobs = {n for n, d in G.nodes(data=True) if d.get("type") == "dab_job"}
    if not tables and not jobs:
        return []

    def label(n) -> str:
        return str(G.nodes[n].get("label", n)) if n in G else str(n)

    writers: dict = {t: set() for t in tables}
    readers: dict = {t: set() for t in tables}
    has_task: dict = {}
    upstream: dict = {}
    for u, v, d in G.edges(data=True):
        src, tgt = _ends(G, u, v, d)
        rel = d.get("relation")
        if tgt in tables and rel == "writes_to":
            writers[tgt].add(src)
            if d.get("also_reads"):
                readers[tgt].add(src)
        elif tgt in tables and rel == "reads_from":
            readers[tgt].add(src)
        elif rel == "has_task" and src in jobs:
            has_task.setdefault(src, set()).add(tgt)
        elif rel == "depends_on":
            upstream.setdefault(src, set()).add(tgt)

    lines = ["", "## Data lineage (Databricks)"]
    if tables:
        ranked = sorted(tables, key=lambda t: (-(len(writers[t]) + len(readers[t])), label(t)))
        shown = ranked[:top_tables]
        lines += ["", f"Tables ({len(tables)} total, busiest {len(shown)} shown): "
                      "writers -> `table` -> readers"]
        for t in shown:
            def fmt(ns: set) -> str:
                names = sorted(label(n) for n in ns)
                more = f" +{len(names) - top_refs}" if len(names) > top_refs else ""
                return ", ".join(names[:top_refs]) + more if names else "-"
            kind = G.nodes[t].get("uc_kind")
            kind = f" ({kind})" if kind and kind != "unknown" else ""
            lines.append(f"- {fmt(writers[t])} -> `{label(t)}`{kind} -> {fmt(readers[t])}")
    if jobs:
        lines += ["", "Jobs (tasks in run order; `a <- b` means a waits on b):"]
        for job in sorted(jobs, key=label):
            tasks = has_task.get(job, set())
            order = _topo(tasks, upstream)
            parts = []
            for t in order:
                deps = sorted(label(d).split(".")[-1] for d in upstream.get(t, set()) if d in tasks)
                name = label(t).split(".")[-1]
                parts.append(f"{name} <- {', '.join(deps)}" if deps else name)
            lines.append(f"- `{label(job)}`: " + ("; ".join(parts) if parts else "(no tasks)"))
    return lines


def _topo(tasks: set, upstream: dict) -> list:
    remaining = set(tasks)
    order: list = []
    while remaining:
        ready = sorted((t for t in remaining
                        if not (upstream.get(t, set()) & remaining)), key=str)
        if not ready:  # a cycle: emit the rest as-is
            ready = sorted(remaining, key=str)
        for t in ready:
            order.append(t)
            remaining.discard(t)
        if not ready:
            break
    return order
