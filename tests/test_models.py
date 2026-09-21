"""Model unit tests required by spec section 21 "Model tests"."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.models import LateralMLP, LongitudinalMLP
from training.train_core import set_seed, TRAINING_CONFIG


def test_deterministic_init_with_seed():
    set_seed(TRAINING_CONFIG["seed"])
    m1 = LateralMLP()
    set_seed(TRAINING_CONFIG["seed"])
    m2 = LateralMLP()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_lateral_tensor_dimensions():
    m = LateralMLP()
    x = torch.randn(37, 11)
    y = m(x)
    assert y.shape == (37, 2)


def test_longitudinal_tensor_dimensions():
    m = LongitudinalMLP()
    x = torch.randn(37, 12)
    y = m(x)
    assert y.shape == (37, 2)


def test_lateral_forward_finite():
    m = LateralMLP()
    x = torch.randn(200, 11) * 5
    y = m(x)
    assert torch.isfinite(y).all()


def test_longitudinal_forward_finite():
    m = LongitudinalMLP()
    x = torch.randn(200, 12) * 5
    y = m(x)
    assert torch.isfinite(y).all()


def test_longitudinal_throttle_output_in_0_1():
    m = LongitudinalMLP()
    x = torch.randn(500, 12) * 10  # extreme inputs -- sigmoid must still saturate correctly
    y = m(x)
    throttle = y[:, 1]
    assert (throttle >= 0.0).all() and (throttle <= 1.0).all()


def test_saturated_surface_commands_in_bounds():
    m = LateralMLP()
    x = torch.randn(500, 11) * 10
    y = m(x)
    y_sat = torch.clamp(y, -1.0, 1.0)
    assert (y_sat >= -1.0).all() and (y_sat <= 1.0).all()

    m2 = LongitudinalMLP()
    y2 = m2(torch.randn(500, 12) * 10)
    elevator_sat = torch.clamp(y2[:, 0], -1.0, 1.0)
    assert (elevator_sat >= -1.0).all() and (elevator_sat <= 1.0).all()


@pytest.mark.parametrize("ModelCls,n_in", [(LateralMLP, 11), (LongitudinalMLP, 12)])
def test_finite_loss_and_gradients(ModelCls, n_in):
    set_seed(TRAINING_CONFIG["seed"])
    m = ModelCls()
    x = torch.randn(64, n_in)
    y = torch.randn(64, 2)
    pred = m(x)
    loss = nn.functional.mse_loss(pred, y)
    assert torch.isfinite(loss)
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("ModelCls,n_in", [(LateralMLP, 11), (LongitudinalMLP, 12)])
def test_checkpoint_save_and_reload_matches(ModelCls, n_in, tmp_path):
    set_seed(TRAINING_CONFIG["seed"])
    m = ModelCls()
    x = torch.randn(50, n_in)
    with torch.no_grad():
        y_before = m(x).clone()

    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save(m.state_dict(), ckpt_path)

    m2 = ModelCls()
    m2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    m2.eval()
    with torch.no_grad():
        y_after = m2(x)

    assert torch.allclose(y_before, y_after, atol=1e-6)


def test_no_batchnorm_or_dropout_layers():
    for m in (LateralMLP(), LongitudinalMLP()):
        for module in m.modules():
            assert not isinstance(module, (nn.BatchNorm1d, nn.Dropout)), \
                f"{type(m).__name__} must not contain BatchNorm/Dropout per spec section 15"
