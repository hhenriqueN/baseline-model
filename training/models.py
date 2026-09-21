"""Baseline network architectures (spec sections 15-16).

Lateral MLP:   11 in -> Linear(11,64) -> SiLU -> Linear(64,64) -> SiLU -> Linear(64,2)
Longitudinal:  12 in -> shared trunk (64,64 SiLU) -> elevator head Linear(64,1) linear,
               throttle head Linear(64,1) -> sigmoid.

No BatchNorm, no dropout (spec section 15). Regularization is weight decay only,
applied at the optimizer level (spec section 17), not architecturally here.
"""
from __future__ import annotations

import torch
from torch import nn


class LateralMLP(nn.Module):
    """Inputs: LATERAL_FEATURE_ORDER (11). Outputs: [aileron_command, rudder_command]."""

    def __init__(self, n_inputs: int = 11, n_hidden: int = 64, n_outputs: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LongitudinalMLP(nn.Module):
    """Inputs: LONGITUDINAL_FEATURE_ORDER (12). Outputs: [elevator_effective, throttle_command]."""

    def __init__(self, n_inputs: int = 12, n_hidden: int = 64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.SiLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.SiLU(),
        )
        self.elevator_head = nn.Linear(n_hidden, 1)
        self.throttle_head = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        elevator = self.elevator_head(z)
        throttle = torch.sigmoid(self.throttle_head(z))
        return torch.cat([elevator, throttle], dim=1)


ARCHITECTURE_CONFIG = {
    "lateral": {
        "class": "LateralMLP",
        "n_inputs": 11, "n_hidden": 64, "n_outputs": 2,
        "layers": ["Linear(11,64)", "SiLU", "Linear(64,64)", "SiLU", "Linear(64,2)"],
        "output_activation": ["linear", "linear"],
        "batch_norm": False, "dropout": False,
    },
    "longitudinal": {
        "class": "LongitudinalMLP",
        "n_inputs": 12, "n_hidden": 64, "n_outputs": 2,
        "layers": ["Linear(12,64)", "SiLU", "Linear(64,64)", "SiLU",
                    "elevator_head=Linear(64,1)[linear]", "throttle_head=Linear(64,1)[sigmoid]"],
        "output_activation": ["linear", "sigmoid"],
        "batch_norm": False, "dropout": False,
    },
}
