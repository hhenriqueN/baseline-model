"""Loads the two FROZEN final checkpoints (spec: do not retrain or modify)
and the scaler referenced by models/model_manifest.json, and runs inference.

Elevator/trim decomposition (spec: "Handle elevator_effective =
elevator_command + elevator_trim correctly. Do not double-count trim."):

The c172p JSBSim flight-control system (Aircraft/c172p/c172p.xml, channel
"Pitch") computes the physical elevator surface deflection as:
    fcs/pitch-trim-sum = clip(gain(rate_limit(elevator-cmd-norm)) + pitch-trim-cmd-norm, -1, 1)
i.e. the RAW /controls/flight/elevator command (after an internal rate
limiter and a dynamic-pressure-based gain schedule) is summed with
/controls/flight/elevator-trim inside the FDM. The network was trained to
predict `elevator_effective = elevator_command + elevator_trim` as a SINGLE
scalar label (pipeline/columns.py, spec section 16) -- i.e. it already
represents the desired COMBINED raw-command-plus-trim value the pilot used
at that instant (before the FDM's internal rate/gain shaping, which the
label does not model either).

We hold /controls/flight/elevator-trim at ONE fixed value for the whole
controlled flight (set once at initialization from the project's own
demonstration data -- config.NominalInitCondition.elevator_trim_fixed) and
send the network's full predicted `elevator_effective`, MINUS that fixed
trim, to /controls/flight/elevator every cycle:

    elevator_command_sent = clip(elevator_effective_predicted - trim_fixed, -1, 1)
    elevator_trim_sent = trim_fixed  (unchanged, set once)

so elevator_command_sent + elevator_trim_sent == elevator_effective_predicted
by construction -- the trim is counted exactly once. Nothing else in this
session ever writes /controls/flight/elevator-trim (no joystick/keyboard
trim-rocker input is bound in a scripted session), so it cannot drift and
create a second, uncounted trim contribution.
"""
from __future__ import annotations

import json

import torch

from pipeline import paths as pipeline_paths
from training.models import LateralMLP, LongitudinalMLP

MODEL_CLASSES = {"lateral": LateralMLP, "longitudinal": LongitudinalMLP}


class LoadedNetwork:
    def __init__(self, network: str, manifest_entry: dict):
        self.network = network
        self.feature_order = manifest_entry["feature_order"]
        self.label_order = manifest_entry["label_order"]
        self.output_saturation = manifest_entry["output_saturation"]
        self.model = MODEL_CLASSES[network]()
        ckpt_path = pipeline_paths.ROOT / manifest_entry["checkpoint_path"]
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint_path = ckpt_path

    @torch.no_grad()
    def predict(self, feature_vector: list[float]) -> list[float]:
        x = torch.tensor([feature_vector], dtype=torch.float32)
        y = self.model(x)
        return y.squeeze(0).tolist()


class ModelBundle:
    def __init__(self, manifest_path=None):
        manifest_path = manifest_path or (pipeline_paths.ROOT / "models" / "model_manifest.json")
        self.manifest_path = manifest_path
        with open(manifest_path) as f:
            self.manifest = json.load(f)
        self.lateral = LoadedNetwork("lateral", self.manifest["networks"]["lateral"])
        self.longitudinal = LoadedNetwork("longitudinal", self.manifest["networks"]["longitudinal"])
        self.scaler_path = pipeline_paths.ROOT / self.manifest["networks"]["lateral"]["input_normalization"]["scaler_source_file"]
        with open(self.scaler_path) as f:
            self.scaler = json.load(f)

    def validate(self):
        assert len(self.lateral.feature_order) == 11, "lateral input dimension must be 11"
        assert len(self.longitudinal.feature_order) == 12, "longitudinal input dimension must be 12"
        assert self.lateral.label_order == ["aileron_command", "rudder_command"]
        assert self.longitudinal.label_order == ["elevator_effective", "throttle_command"]


def compose_elevator_command(elevator_effective_pred: float, elevator_trim_fixed: float) -> float:
    return max(-1.0, min(1.0, elevator_effective_pred - elevator_trim_fixed))


def saturate(value: float, lo: float, hi: float) -> tuple[float, bool]:
    if value < lo:
        return lo, True
    if value > hi:
        return hi, True
    return value, False
