"""Local-only MCP tool that runs the saved EAT fusion model (with E31 alongside) on a FIT folder.

The models are derived from FitRec (academic use, no redistribution), so this module is never copied
into the public durability-mcp package. Every number it returns is MEASURED EXPLORATORY.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PROVENANCE = "MEASURED EXPLORATORY; EAT FUSION (FITREC + GOLDENCHEETAH); LOCAL ONLY; NOT VALIDATED ON FIT FILES"
CAVEAT = (
    "EAT scores only runs that pass the conjunctive gate (drift > 2.0 bpm/h AND magnitude > 1.03). "
    "On development data it tied E31 on drift and beat it by about 1.3% on magnitude. On the one real export "
    "tested only one run could be scored, so no conclusion about real watch files is possible. "
    "Treat it as a research comparison, not advice."
)
MODELS = ("eat", "e31", "training_mean", "own_shrunk")


def summarise(runs, digits: int = 4) -> dict[str, Any]:
    """Per-run EAT and E31 predictions next to observed values and both baselines, plus MAE over scored runs."""
    rows, mae = [], {}
    for _, r in runs.iterrows():
        rows.append({"activity_id": str(r["activity_id"]), "date": str(r["start_time"])[:10],
                     **{k: round(float(r[k]), digits) for k in r.index if k.split("_")[0] in ("drift", "magnitude")}})
    for target in ("drift", "magnitude"):
        if len(runs):
            mae[target] = {m: round(float(np.mean(np.abs(runs[f"{target}_{m}"] - runs[f"{target}_observed"]))), digits) for m in MODELS}
    return {"runs": rows, "mae": mae}


def register(server, folder: str, config: dict[str, Any], annotations) -> None:
    cache: dict[str, Any] = {}

    @server.tool(annotations=annotations)
    def get_fusion_predictions() -> dict[str, Any]:
        """Run the saved EAT fusion model (and E31) on gate-passing runs and compare with what happened and with two baselines. Local only. No GPS is returned."""
        if "result" not in cache:
            from durability.eat_artifact import predict_fit_folder

            cache["result"] = predict_fit_folder(folder, "local_fit", config)
        result = cache["result"]
        body = summarise(result["runs"])
        counts = result["summary"]["counts"]
        text = f"EAT scored {len(body['runs'])} gate-passing run(s) out of {counts.get('runs', '?')} outdoor runs."
        for target, unit in (("drift", " bpm/h"), ("magnitude", "")):
            if body["mae"]:
                d = body["mae"][target]
                text += f" {target.capitalize()} MAE: EAT {d['eat']}, E31 {d['e31']}, training mean {d['training_mean']}, own shrunk {d['own_shrunk']}{unit}."
        return {"summary": text, "counts": counts, **body, "provenance": PROVENANCE, "caveat": CAVEAT}
