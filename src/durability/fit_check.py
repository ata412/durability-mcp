"""Command line check of a FIT folder, without any MCP client: durability-mcp-check <folder>."""

from __future__ import annotations

import argparse
import json

from durability.config import load_config
from durability.fit_tool import CAVEAT, FitFolder, expected_next, trend


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise durability from a folder of FIT files. Output contains no GPS coordinates.")
    parser.add_argument("folder", help="folder that holds .fit files directly (not in subfolders)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of a table")
    args = parser.parse_args()
    config = load_config()
    rows = FitFolder(args.folder, config).runs()
    summary, expected = trend(rows, config), expected_next(rows, config)
    if args.json:
        print(json.dumps({"runs": rows, "trend": summary, "expected_next": expected}, indent=2, default=str))
        return
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"{len(rows)} outdoor runs")
    for column in ("Date", "File", "Min", "Status", "Drift bpm/h", "Magnitude", "Why not evaluable"):
        table.add_column(column)
    for r in rows:
        why = r.get("reason_detail") or r["reason_text"] or ""
        table.add_row(r["date"], r["file"], f"{r['duration_min']:.0f}" if r["duration_min"] is not None else "", r["status"], str(r["drift_bpm_h"] or ""), str(r["magnitude"] or ""), why)
    console.print(table)
    counts = summary["status_counts"]
    console.print(f"signal {counts['signal']}, no_signal {counts['no_signal']}, not_evaluable {counts['not_evaluable']}")
    d, m = expected["drift_bpm_h"], expected["magnitude"]
    console.print(f"Expected next evaluable run: drift {d['expected']} ± {d['typical_error']} bpm/h, magnitude {m['expected']} ± {m['typical_error']} (typical errors; from {d['own_runs_used']} own runs)")
    console.print(f"[dim]{expected['evidence']}\n{CAVEAT}[/dim]")


if __name__ == "__main__":
    main()
