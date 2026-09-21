import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from controller.config import SafetyEnvelope, RolloutConfig
from controller.safety import SafetyMonitor
from controller.rollout_controller import RolloutController
from controller.telemetry_io import Telemetry


def make_monitor():
    return SafetyMonitor(SafetyEnvelope(telemetry_stale_s=0.2, max_consecutive_saturated_cycles=3), 0.0)


def test_no_telemetry_yet_aborts():
    m = make_monitor()
    d = m.check_telemetry(None, None)
    assert d.should_abort and d.reason == "no_telemetry_received_yet"


def test_stale_telemetry_aborts():
    m = make_monitor()
    telem = Telemetry(fields={"x": 1.0}, received_wall_time=time.monotonic() - 1.0)
    d = m.check_telemetry(telem, telem.received_wall_time)
    assert d.should_abort
    assert "stale" in d.reason


def test_fresh_telemetry_ok():
    m = make_monitor()
    telem = Telemetry(fields={"x": 1.0}, received_wall_time=time.monotonic())
    d = m.check_telemetry(telem, telem.received_wall_time)
    assert not d.should_abort


def test_nan_telemetry_aborts():
    m = make_monitor()
    telem = Telemetry(fields={"x": float("nan")}, received_wall_time=time.monotonic())
    d = m.check_telemetry(telem, telem.received_wall_time)
    assert d.should_abort and "nan" in d.reason


def test_envelope_cross_track_violation():
    m = make_monitor()
    d = m.check_envelope(cross_track_m=999, heading_error_deg=0, roll_deg=0, pitch_deg=0, agl_ft=500, airspeed_kt=65)
    assert d.should_abort and "cross_track" in d.reason


def test_envelope_within_bounds_ok():
    m = make_monitor()
    d = m.check_envelope(cross_track_m=10, heading_error_deg=5, roll_deg=5, pitch_deg=3, agl_ft=500, airspeed_kt=65)
    assert not d.should_abort


def test_saturation_tracking_aborts_after_threshold():
    m = make_monitor()
    for _ in range(2):
        d = m.track_saturation("aileron", True)
        assert not d.should_abort
    d = m.track_saturation("aileron", True)
    assert d.should_abort


def test_saturation_resets_on_non_saturated_cycle():
    m = make_monitor()
    m.track_saturation("aileron", True)
    m.track_saturation("aileron", True)
    d = m.track_saturation("aileron", False)
    assert not d.should_abort
    d2 = m.track_saturation("aileron", True)
    assert not d2.should_abort  # counter was reset, this is only cycle 1 again


def test_crashed_flag_aborts():
    m = make_monitor()
    assert m.check_crashed(1).should_abort
    assert not m.check_crashed(0).should_abort


def test_rollout_controller_rudder_corrects_toward_runway_heading():
    rc = RolloutController(RolloutConfig(), runway_heading_deg=0.0)
    rc.reset(0.0)
    cmd = rc.compute(t=0.1, heading_deg=10.0, groundspeed_kt=30.0)
    assert cmd["rudder"] > 0  # nose right of centerline heading -> correct back (sign per rudder_kp>0 convention)
    cmd2 = rc.compute(t=0.1, heading_deg=-10.0, groundspeed_kt=30.0)
    assert cmd2["rudder"] < 0


def test_rollout_controller_throttle_idle_and_progressive_brakes():
    rc = RolloutController(RolloutConfig(brake_ramp_duration_s=2.0, brake_max=0.7), runway_heading_deg=0.0)
    rc.reset(0.0)
    early = rc.compute(t=0.1, heading_deg=0.0, groundspeed_kt=40.0)
    late = rc.compute(t=2.5, heading_deg=0.0, groundspeed_kt=40.0)
    assert early["throttle"] == 0.0 and late["throttle"] == 0.0
    assert late["brake_left"] > early["brake_left"]
    assert late["brake_left"] <= 0.7 + 1e-9
