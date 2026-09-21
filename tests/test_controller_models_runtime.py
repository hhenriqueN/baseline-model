import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controller.models_runtime import ModelBundle, compose_elevator_command, saturate


def test_model_bundle_loads_and_validates():
    bundle = ModelBundle()
    bundle.validate()
    assert bundle.lateral.checkpoint_path.exists()
    assert bundle.longitudinal.checkpoint_path.exists()


def test_model_bundle_predicts_finite_2d_outputs():
    bundle = ModelBundle()
    lat_pred = bundle.lateral.predict([0.0] * 11)
    lon_pred = bundle.longitudinal.predict([0.0] * 12)
    assert len(lat_pred) == 2 and all(v == v for v in lat_pred)
    assert len(lon_pred) == 2 and all(v == v for v in lon_pred)


def test_compose_elevator_command_no_double_counting():
    # If the network predicts exactly the fixed trim as the "effective"
    # value, the composed raw command must be 0 (all of the effect already
    # accounted for by trim, none needed from the fast elevator channel).
    trim = 0.031
    assert compose_elevator_command(elevator_effective_pred=trim, elevator_trim_fixed=trim) == 0.0

    # elevator_command + elevator_trim must reconstruct elevator_effective
    # exactly (before saturation).
    pred = 0.2
    cmd = compose_elevator_command(pred, trim)
    assert abs((cmd + trim) - pred) < 1e-9


def test_compose_elevator_command_saturates_to_valid_range():
    assert compose_elevator_command(elevator_effective_pred=5.0, elevator_trim_fixed=0.0) == 1.0
    assert compose_elevator_command(elevator_effective_pred=-5.0, elevator_trim_fixed=0.0) == -1.0


def test_saturate_basic():
    assert saturate(0.5, -1, 1) == (0.5, False)
    assert saturate(1.5, -1, 1) == (1.0, True)
    assert saturate(-1.5, -1, 1) == (-1.0, True)
