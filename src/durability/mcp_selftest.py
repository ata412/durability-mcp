"""Call the local durability MCP server over stdio, as an AI client would, but without any AI.

Run: python -m durability.mcp_selftest /path/to/fit/folder
It starts the server as a child process, lists its tools and calls every tool that needs no
arguments. Output contains no GPS coordinates.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

NO_ARGUMENT_TOOLS = ("list_durability_runs", "get_durability_trend", "get_expected_next_run", "get_fusion_predictions", "get_e31_predictions")


async def run(folder: str, as_json: bool) -> int:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "durability.mcp_server"],
                                   env={**os.environ, "DURABILITY_FIT_DIR": folder})
    report: dict[str, object] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [t.name for t in (await session.list_tools()).tools]
            report["tools"] = tools
            if not as_json:
                print("Tools offered by the server:", ", ".join(tools))
            for name in NO_ARGUMENT_TOOLS:
                if name not in tools:
                    continue
                result = await session.call_tool(name, {})
                body = json.loads(result.content[0].text)
                report[name] = body
                if not as_json:
                    print(f"\n[{name}]\n{body.get('summary', '')}")
    if as_json:
        print(json.dumps(report, indent=2, default=str))
    elif not any(t in report for t in ("get_fusion_predictions", "get_e31_predictions")):
        print("\nNo model tool: install with the [models] extra to add get_fusion_predictions.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the durability MCP server over the real MCP protocol, without an AI client.")
    parser.add_argument("folder", help="folder that holds .fit files directly (not in subfolders)")
    parser.add_argument("--json", action="store_true", help="print every tool result as JSON")
    args = parser.parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Not a folder: {folder}")
    raise SystemExit(asyncio.run(run(folder, args.json)))


if __name__ == "__main__":
    main()
