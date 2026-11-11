# durability-mcp

A local, read-only MCP tool that summarises running durability from your own FIT files.

Your files stay on your computer: the tool runs on your machine, reads one folder, and returns no GPS coordinates. Its only forward estimate is a shrunk personal average for your next run, because no trained model beat that baseline before a run starts. This is a research measure, not a diagnosis.

Requires [uv](https://docs.astral.sh/uv/).

## Try it on generated files

```bash
uvx --from git+https://github.com/ata412/durability-mcp durability-mcp-demo
```

The demo uses SYNTHETIC files whose answers were set in advance. It shows that the tool works as designed, not that it is accurate.

## Check your own files without an AI client

```bash
uvx --from git+https://github.com/ata412/durability-mcp durability-mcp-check /path/to/your/fit/folder
```

Put the `.fit` files directly in the folder. Nothing leaves your computer. Add `--json` for machine-readable output.

## Use it with an MCP client

Add this to your MCP client's configuration:

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
| `get_expected_next_run` | Expected drift and magnitude for your next evaluable run, with typical error |

Statuses: `signal` (both gate conditions met), `no_signal` (QC passed, gate not met), `not_evaluable` (first failing QC rule given).

## Model tool (optional, research use)

Install the `[models]` extra to add `get_fusion_predictions`. It runs the saved EAT fusion model (a stack of tree experts and the E31 blend, combined by an expert-attention network) on runs that pass the gate, and lists E31, the training mean and your own shrunk average beside what actually happened.

```json
{
  "mcpServers": {
    "durability": {
      "command": "uvx",
      "args": ["--from", "durability-mcp[models] @ git+https://github.com/ata412/durability-mcp", "durability-mcp"],
      "env": {"DURABILITY_FIT_DIR": "/path/to/your/fit/folder"}
    }
  }
}
```

- The models were trained on FitRec (Endomondo) and GoldenCheetah runs. FitRec is licensed for academic use only, so use these files for research and do not redistribute them.
- On development data (MEASURED EXPLORATORY, not an independent test) EAT tied E31 on drift and reduced magnitude error by about 1.3%. On the one real watch export tested, only one run could be scored, so nothing is known about accuracy on real watch files.
- The model libraries are large (PyTorch, LightGBM, XGBoost, CatBoost). Without the extra, the tool runs without this model tool.

## Limits

- Tested on one runner's COROS export only, where most runs were not evaluable (MEASURED: 73 of 76). Other brands may behave differently.
- Needs an outdoor run with GPS and heart rate, 45 to 240 minutes long, at a steady pace.
- One run is weak evidence. Prefer the trend over several runs.
