#!/usr/bin/env python3
"""Writes models/model_manifest.json: a single machine-readable file describing
exactly what a future real-time controller must load to run the frozen
baseline policies (spec section 20). Pulls from the already-written final
training configs and the final_all_valid scaler -- no new information is
invented here, only consolidated.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths

ROOT = paths.ROOT
MODELS_DIR = ROOT / "models"


def main():
    manifest = {"schema_version": 1, "networks": {}}
    for network in ("lateral", "longitudinal"):
        final_cfg = json.load(open(MODELS_DIR / network / "final" / "config.json"))
        scaler = json.load(open(ROOT / final_cfg["scaler_path"]))
        manifest["networks"][network] = {
            "checkpoint_path": f"models/{network}/final/checkpoint.pt",
            "architecture": final_cfg["architecture"],
            "feature_order": final_cfg["feature_order"],
            "label_order": final_cfg["label_order"],
            "input_normalization": {
                "policy": "z-score (train-fold/final-train statistics only), clip to [clip_min, clip_max]",
                "zscored_variables": scaler["feature_order_zscored"],
                "not_zscored_variables": scaler["feature_order_not_zscored"],
                "mean": scaler["mean"], "std": scaler["std"],
                "clip_min": scaler["clip_min"], "clip_max": scaler["clip_max"],
                "scaler_source_file": final_cfg["scaler_path"],
                "scaler_schema_version": scaler["schema_version"],
            },
            "output_saturation": {
                "lateral": {"aileron_command": [-1.0, 1.0], "rudder_command": [-1.0, 1.0]},
                "longitudinal": {"elevator_effective": [-1.0, 1.0], "throttle_command": [0.0, 1.0]},
            }[network],
            "labels_normalized": False,
            "n_epochs_trained": final_cfg["n_epochs_trained"],
            "training_config": final_cfg["training_config"],
            "dataset_version": final_cfg["dataset_version"],
            "n_train_rows": final_cfg["n_train_rows"],
            "excluded_phases_from_training": ["TOUCHDOWN_CONFIRMED", "BOUNCE", "ROLLOUT", "COMPLETE", "GO_AROUND"],
            "eligible_training_phases": ["FINAL_APPROACH_ESTABLISHED", "FLARE", "CONTACT_CANDIDATE"],
        }
    manifest["notes"] = (
        "Offline behavioral-cloning baseline only. Inputs must be assembled in "
        "exactly `feature_order`, z-scored per `input_normalization` (skip the "
        "listed not_zscored_variables), and outputs saturated per "
        "`output_saturation` before being sent to the simulator. Real-time "
        "telemetry, closed-loop inference, and command transmission are NOT "
        "implemented by this package -- see reports/training_report.md for scope."
    )
    out_path = MODELS_DIR / "model_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
