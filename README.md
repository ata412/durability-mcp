# durability-mcp

A local, read-only MCP tool that summarises running durability from your own FIT files.

Your files stay on your computer: the tool runs on your machine, reads one folder, and returns no GPS coordinates. It makes no prediction. This is a research measure, not a diagnosis.

Requires [uv](https://docs.astral.sh/uv/).

## Try it on generated files

```bash
uvx --from git+https://github.com/ata412/durability-mcp durability-mcp-demo
```

The demo uses SYNTHETIC files whose answers were set in advance. It shows that the tool works as designed, not that it is accurate.

## Use it with your own files

Add this to your MCP client's configuration, such as Claude Desktop:

```json
{
  "mcpServers": {
    "durability": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/ata412/durability-mcp", "durability-mcp"],
      "env": {"DURABILITY_FIT_DIR": "/path/to/your/fit/folder"}
    }
  }
}
```

Your MCP client may send tool results to its own AI service, so read its privacy terms. Results can include dates, durations and index values.

| Tool | Returns |
|---|---|
| `list_durability_runs` | Outdoor runs with status and reason |
| `get_durability_indices` | Drift, magnitude, onset and status for one run |
| `get_durability_trend` | Status counts and latest runs against your own earlier runs |

Statuses: `signal` (both gate conditions met), `no_signal` (QC passed, gate not met), `not_evaluable` (first failing QC rule given).

## Limits

- Tested on one runner's COROS export only, where most runs were not evaluable (MEASURED: 73 of 76). Other brands may behave differently.
- Needs an outdoor run with GPS and heart rate, 45 to 240 minutes long, at a steady pace.
- One run is weak evidence. Prefer the trend over several runs.
