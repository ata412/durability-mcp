#!/usr/bin/env python
"""Terminal demo of the local MCP tool on SYNTHETIC files. Starts the real server and calls it over stdio.

    durability-mcp-demo          # press Enter between steps
    durability-mcp-demo --no-pause
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from typing import Any

from mcp import Client, StdioServerParameters
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from durability.config import load_config
from durability.fit_demo import build_demo_folder

STATUS_STYLE = {"signal": "bold red", "no_signal": "green", "not_evaluable": "yellow"}
console = Console()


def payload(result: Any) -> dict[str, Any]:
    return json.loads(result.content[0].text)


def step(number: int, title: str, call: str, pause: bool) -> None:
    if pause and sys.stdin.isatty():
        console.input("\n[dim]Press Enter for the next step[/dim]")
    console.print()
    console.rule(f"[bold]Step {number}. {title}")
    console.print(f"[dim]MCP call:[/dim] [cyan]{call}[/cyan]")


def banner() -> None:
    console.print(Panel("[bold]SYNTHETIC DEMO DATA[/bold]\nGenerated running files, not a real person and not research results.\nThe server, QC and index code are the real ones. The tool makes no prediction.", style="yellow", title="Local durability tool"))


async def run(folder: str, scenarios: list[dict[str, Any]], pause: bool) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-m", "durability.mcp_server"], env={**os.environ, "DURABILITY_FIT_DIR": folder, "DURABILITY_LOG_LEVEL": "CRITICAL"})
    by_status = {s["expect"]: s["file"] for s in reversed(scenarios)}
    async with Client(params, read_timeout_seconds=300) as client:
        step(1, "The server offers three read-only tools", "tools/list", pause)
        table = Table(show_lines=False)
        table.add_column("Tool", style="cyan")
        table.add_column("Read-only")
        table.add_column("Purpose")
        for tool in (await client.list_tools()).tools:
            table.add_row(tool.name, "yes" if tool.annotations.read_only_hint else "NO", tool.description.split(".")[0])
        console.print(table)

        step(2, "List the runs in the folder", "list_durability_runs()", pause)
        listed = payload(await client.call_tool("list_durability_runs", {}))
        console.print(listed["summary"])
        table = Table(pad_edge=False)
        for column in ("File", "Date", "Min", "Status", "First failing rule"):
            table.add_column(column, no_wrap=True)
        for row in listed["runs"]:
            table.add_row(row["file"], row["date"], f"{row['duration_min']:.0f}", f"[{STATUS_STYLE[row['status']]}]{row['status']}[/]", row["reason"] or "")
        console.print(table)

        step(3, "One run where both gate conditions are met", f'get_durability_indices("{by_status["signal"]}")', pause)
        one = payload(await client.call_tool("get_durability_indices", {"file": by_status["signal"]}))
        run_ = one["run"]
        console.print(one["summary"])
        console.print(f"Drift {run_['drift_bpm_h']} bpm/h (gate: above 2.0)   Magnitude {run_['magnitude']} (gate: above 1.03)   Status [{STATUS_STYLE[run_['status']]}]{run_['status']}[/]")
        console.print(f"[dim]{one['caveat']}[/dim]")

        step(4, "A run the tool refuses to score, and says why", f'get_durability_indices("{by_status["not_evaluable"]}")', pause)
        one = payload(await client.call_tool("get_durability_indices", {"file": by_status["not_evaluable"]}))
        console.print(one["summary"])
        console.print("Drift: none   Magnitude: none   [dim](the tool returns no number for a run it cannot evaluate)[/dim]")
        assert one["run"]["drift_bpm_h"] is None and one["run"]["magnitude"] is None

        step(5, "Trend against the runner's own earlier runs", "get_durability_trend()", pause)
        trend = payload(await client.call_tool("get_durability_trend", {}))
        console.print(trend["summary"])
        drift = trend["standard_drift_bpm_h"]
        console.print(f"Standard drift: earlier runs {drift['baseline_mean']} bpm/h, latest {drift['recent_mean']} bpm/h -> [bold]{drift['recent_vs_own_baseline']}[/bold]")
        console.print(f"[dim]{trend['experimental_note']}[/dim]")

        step(6, "Privacy and safety checks, run live", 'get_durability_indices("../outside.fit")', pause)
        refused = await client.call_tool("get_durability_indices", {"file": "../outside.fit"})
        console.print(f"File outside the folder: [bold]{'refused' if refused.is_error else 'NOT REFUSED'}[/bold] ({refused.content[0].text})")
        everything = json.dumps([listed, one, trend]).lower()
        leaked = any(word in everything for word in ("latitude", "longitude", "position_lat"))
        console.print(f"Coordinates in any response above: [bold]{'FOUND' if leaked else 'none'}[/bold]")
    console.print()
    console.print(Panel("All values above come from SYNTHETIC files whose answers were set in advance. This shows the tool works as designed.\nIt does not show accuracy on real runners. On the one real export tested, most runs were not evaluable (MEASURED).", style="yellow"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-pause", action="store_true", help="run every step without waiting for Enter")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    config = load_config()
    banner()
    with tempfile.TemporaryDirectory(prefix="durability_demo_") as folder:
        with console.status("Generating synthetic FIT files"):
            scenarios = build_demo_folder(folder, config)
        console.print(f"{len(scenarios)} synthetic FIT files generated (seed {config['fit_demo']['seed']}).")
        asyncio.run(run(folder, scenarios, not args.no_pause))


if __name__ == "__main__":
    main()
