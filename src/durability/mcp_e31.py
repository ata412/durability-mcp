"""Local-only MCP tool that runs the saved E31 blend on a FIT folder.

The E31 models are derived from FitRec (academic use, no redistribution), so this module is never
copied into the public durability-mcp package. Every number it returns is PILOT MEASURED and the
model has been checked on only one real FIT export, where one run passed the gate.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PROVENANCE = "PILOT MEASURED; E31 BLEND FROM FITREC + GOLDENCHEETAH; LOCAL ONLY; NOT VALIDATED ON FIT FILES"
CAVEAT = (
    "E31 scores only runs that pass the conjunctive gate (drift > 2.0 bpm/h AND magnitude > 1.03). "
    "On one real export it scored 1 of 76 runs and did not beat the runner's own shrunk average overall. "
    "Treat it as a research comparison, not advice."
)


def summarise(runs, digits: int = 4) -> dict[str, Any]:
    """Per-run E31 predictions next to observed values and both baselines, plus MAE over scored runs."""
    rows, mae = [], {}
    for _, r in runs.iterrows():
        rows.append({"activity_id": str(r["activity_id"]), "date": str(r["start_time"])[:10],
                     **{k: round(float(r[k]), digits) for k in r.index if k.split("_")[0] in ("drift", "magnitude")}})
    for target in ("drift", "magnitude"):
        if len(runs):
            mae[target] = {m: round(float(np.mean(np.abs(runs[f"{target}_{m}"] - runs[f"{target}_observed"]))), digits) for m in ("e31", "training_mean", "own_shrunk")}
    return {"runs": rows, "mae": mae}


def register(server, folder: str, config: dict[str, Any], annotations) -> None:
    cache: dict[str, Any] = {}

    @server.tool(annotations=annotations)
    def get_e31_predictions() -> dict[str, Any]:
        """Run the saved E31 model on gate-passing runs and compare with what happened and with two baselines. Local only. No GPS is returned."""
        if "result" not in cache:
            from durability.e31_artifact import predict_fit_folder

            cache["result"] = predict_fit_folder(folder, "local_fit", config)
        result = cache["result"]
        body = summarise(result["runs"])
        counts = result["summary"]["counts"]
        text = f"E31 scored {len(body['runs'])} gate-passing run(s) out of {counts.get('runs', '?')} outdoor runs."
        if body["mae"]:
            d = body["mae"]["drift"]
            text += f" Drift MAE: E31 {d['e31']}, training mean {d['training_mean']}, own shrunk {d['own_shrunk']} bpm/h."
        return {"summary": text, "counts": counts, **body, "provenance": PROVENANCE, "caveat": CAVEAT}
