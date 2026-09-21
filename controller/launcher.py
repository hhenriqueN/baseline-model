"""Finds and launches the installed FlightGear executable directly (bypassing
the Qt launcher/"Fly" button) with the nominal-approach initial condition
computed from the project's own runway configuration.

Passing explicit simulation flags (--aircraft, --lat/--lon, etc.) is what
makes modern FlightGear boot straight into the simulator instead of showing
the launcher UI -- confirmed by inspecting a prior interactive session's log
(~/Library/Application Support/FlightGear/fgfs.log), which shows the exact
same bundle used this way, and by the running process's call stack
(fgMainInit / fgOSMainLoop -- the real simulator, not a separate launcher
process).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pipeline.runway import RunwayConfig, point_at_along_cross
from controller import config as C
from controller.protocol import build_protocol_xml


def find_fgfs(explicit_path: str | None = None) -> Path:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"--fgfs-path given but not found: {p}")
        return p
    for cand in C.FGFS_CANDIDATES:
        if cand.exists():
            return cand
    found = shutil.which("fgfs")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "Could not auto-detect the FlightGear executable. Tried: "
        + ", ".join(str(c) for c in C.FGFS_CANDIDATES)
        + ". Pass --fgfs-path /path/to/fgfs (or the macOS .app's Contents/MacOS/FlightGear binary)."
    )


def write_protocol_file(fg_root: Path) -> Path:
    proto_dir = fg_root / "Protocol"
    proto_dir.mkdir(parents=True, exist_ok=True)
    out_path = proto_dir / f"{C.PROTOCOL_NAME}.xml"
    out_path.write_text(build_protocol_xml())
    return out_path


def compute_initial_position(runway_cfg: RunwayConfig, init: C.NominalInitCondition) -> dict:
    lat, lon = point_at_along_cross(init.along_track_m, init.cross_track_m, runway_cfg)
    return {
        "lat": lat, "lon": lon,
        "heading": runway_cfg.true_heading_deg,
        "altitude_ft": init.initial_msl_altitude_ft,
        "vc_kt": init.airspeed_kt,
    }


def build_fgfs_argv(fgfs_path: Path, fg_root: Path, fg_home: Path, runway_cfg: RunwayConfig,
                      init: C.NominalInitCondition, telemetry_port: int, command_port: int) -> list[str]:
    pos = compute_initial_position(runway_cfg, init)
    argv = [
        str(fgfs_path),
        f"--fg-root={fg_root}",
        # skip the Qt launcher ("Fly" button). The macOS bundle forces
        # FG_LAUNCHER=1 via Info.plist; --launcher=0 is the documented override.
        "--launcher=0",
        "--aircraft=c172p",
        f"--lat={pos['lat']:.8f}",
        f"--lon={pos['lon']:.8f}",
        f"--heading={pos['heading']:.3f}",
        f"--altitude={pos['altitude_ft']:.1f}",
        f"--vc={pos['vc_kt']:.1f}",
        "--in-air",
        "--roll=0",
        # realistic approach configuration (spec: avoid the unconfigured
        # transient at t=0 seen in the raw demonstrations; start already
        # stabilized -- see controller/config.py docstring for the exact
        # source numbers).
        f"--prop:/controls/flight/flaps={init.flaps_init}",
        f"--prop:/controls/engines/engine[0]/throttle={init.throttle_init}",
        f"--prop:/controls/flight/elevator-trim={init.elevator_trim_fixed}",
        "--prop:/controls/gear/gear-down=true",
        # autopilot must stay disabled -- never engaged, default-off state.
        "--prop:/autopilot/locks/passive-mode=true",
        # calm weather for the first test (spec).
        f"--metar={init.metar}",
        "--real-weather-fetch=0",
        "--ai-traffic=0",
        "--random-objects=0",
        "--sound=0",
        "--timeofday=noon",
        "--terrasync=1",
        "--sentry=0",
        # telemetry OUT (FlightGear -> controller) and commands IN (controller -> FlightGear)
        f"--generic=socket,out,{int(C.CONTROL_HZ)},127.0.0.1,{telemetry_port},udp,{C.PROTOCOL_NAME}",
        f"--generic=socket,in,{int(C.CONTROL_HZ)},127.0.0.1,{command_port},udp,{C.PROTOCOL_NAME}",
        # secondary props/telnet channel, used only for one-off readiness
        # checks and init verification (not the 10 Hz control path).
        f"--telnet=socket,bi,10,127.0.0.1,{C.TELNET_PORT},tcp",
    ]
    return argv


class FlightGearProcess:
    def __init__(self, argv: list[str], log_path: Path):
        self.argv = argv
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def launch(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "w")
        self._log_file.write("# exact launch command:\n# " + " ".join(self.argv) + "\n\n")
        self._log_file.flush()
        self.proc = subprocess.Popen(
            self.argv, stdout=self._log_file, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        return self.proc

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def terminate(self, timeout: float = 10.0):
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5.0)
        try:
            self._log_file.close()
        except Exception:
            pass
