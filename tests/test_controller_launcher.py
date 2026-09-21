import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths
from pipeline.runway import RunwayConfig, along_cross_track
from controller import config as C
from controller.launcher import compute_initial_position, build_fgfs_argv, write_protocol_file, find_fgfs

RUNWAY_CFG = RunwayConfig.from_yaml(paths.CONFIG_DIR / "runway.yaml")


def test_compute_initial_position_matches_runway_geometry():
    init = C.NominalInitCondition()
    pos = compute_initial_position(RUNWAY_CFG, init)
    along, cross = along_cross_track([pos["lat"]], [pos["lon"]], RUNWAY_CFG)
    assert along[0] == pytest.approx(init.along_track_m, abs=0.01)
    assert cross[0] == pytest.approx(init.cross_track_m, abs=0.01)
    assert pos["heading"] == RUNWAY_CFG.true_heading_deg
    assert pos["altitude_ft"] == init.initial_msl_altitude_ft
    assert pos["vc_kt"] == init.airspeed_kt


def test_build_fgfs_argv_contains_required_flags():
    init = C.NominalInitCondition()
    argv = build_fgfs_argv(Path("/fake/fgfs"), Path("/fake/fgroot"), Path("/fake/fghome"),
                             RUNWAY_CFG, init, telemetry_port=5500, command_port=5510)
    joined = " ".join(argv)
    assert "--aircraft=c172p" in joined
    assert "--vc=65.0" in joined
    assert "--in-air" in joined
    assert "socket,out,10,127.0.0.1,5500,udp,baseline_controller" in joined
    assert "socket,in,10,127.0.0.1,5510,udp,baseline_controller" in joined
    assert "gear-down=true" in joined
    assert "--metar=" in joined
    assert str(argv[0]) == "/fake/fgfs"


def test_write_protocol_file_creates_xml(tmp_path):
    out = write_protocol_file(tmp_path)
    assert out.exists()
    assert out.name == "baseline_controller.xml"
    content = out.read_text()
    assert "<generic>" in content and "<output>" in content and "<input>" in content


def test_find_fgfs_raises_clear_error_when_nothing_found(monkeypatch):
    monkeypatch.setattr(C, "FGFS_CANDIDATES", [Path("/nonexistent/fgfs")])
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(FileNotFoundError):
        find_fgfs(None)


def test_find_fgfs_explicit_path_must_exist():
    with pytest.raises(FileNotFoundError):
        find_fgfs("/definitely/not/here/fgfs")
