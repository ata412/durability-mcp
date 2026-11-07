"""Local MCP server for durability summaries. Run: DURABILITY_FIT_DIR=<folder> python -m durability.mcp_server"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from durability.config import load_config
from durability.fit_tool import CAVEAT, FitFolder, expected_next, trend

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def build_server(folder: str) -> MCPServer:
    config = load_config()
    fits = FitFolder(folder, config)
    server = MCPServer("durability", log_level=os.environ.get(config["fit_tool"]["log_level_env"], "INFO"))

    @server.tool(annotations=READ_ONLY)
    def list_durability_runs(start_date: str | None = None, end_date: str | None = None, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        """List outdoor runs with status signal, no_signal or not_evaluable. Dates are YYYY-MM-DD. No GPS is returned."""
        rows = fits.runs(start_date, end_date)
        keep = ("file", "date", "duration_min", "status", "reason")
        page = [{k: r[k] for k in keep} for r in rows[max(offset, 0) : max(offset, 0) + max(min(limit, 200), 1)]]
        return {"summary": f"{len(rows)} outdoor runs found; showing {len(page)} from offset {max(offset, 0)}.", "total": len(rows), "runs": page}

    @server.tool(annotations=READ_ONLY)
    def get_durability_indices(file: str) -> dict[str, Any]:
        """Return drift, magnitude, onset and gate status for one run, or why it cannot be evaluated. No GPS is returned."""
        try:
            run = fits.run(file)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        text = {"signal": "Both gate conditions were met in this run.", "no_signal": "The gate conditions were not both met in this run."}.get(run["status"], f"Not evaluable: {run['reason_text']}")
        return {"summary": text, "run": run, "caveat": CAVEAT}

    @server.tool(annotations=READ_ONLY)
    def get_durability_trend(start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        """Summarise runs over time and compare the latest runs with this runner's own earlier runs. No prediction is made."""
        result = trend(fits.runs(start_date, end_date), config)
        counts = result["status_counts"]
        return {"summary": f"{result['runs']} outdoor runs: {counts['signal']} signal, {counts['no_signal']} no signal, {counts['not_evaluable']} not evaluable.", **result}

    @server.tool(annotations=READ_ONLY)
    def get_expected_next_run(start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        """Expected drift and magnitude for the next evaluable run: own average shrunk toward the population. Not a forecast model."""
        rows = fits.runs(start_date, end_date)
        result = expected_next(rows, config)
        d, m = result["drift_bpm_h"], result["magnitude"]
        return {"summary": f"Expected next run: drift {d['expected']} bpm/h (typical error {d['typical_error']}), magnitude {m['expected']} (typical error {m['typical_error']}), from {d['own_runs_used']} own evaluable runs.", **result}

    return server


def main() -> None:
    env = load_config()["fit_tool"]["fit_dir_env"]
    folder = os.environ.get(env)
    if not folder:
        raise SystemExit(f"Set {env} to a folder of FIT files")
    build_server(folder).run()


if __name__ == "__main__":
    main()
