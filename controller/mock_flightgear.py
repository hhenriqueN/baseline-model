#!/usr/bin/env python3
"""Mock FlightGear: stands in for the real simulator on the same generic-
protocol UDP ports, so the full controller loop (telemetry -> features ->
inference -> FSM -> safety -> commands -> logging) can be validated
end-to-end without a live 3D FlightGear process.

It replays a scripted, physically-plausible straight-in approach: starts at
the project's nominal initial condition (~3km out, 800ft MSL, 65kt, aligned),
descends at roughly the nominal 3-degree glideslope, crosses the flare
threshold, touches down with both main gear, and rolls out to a stop --
purely kinematic (no aerodynamics), but enough to drive the Flight Manager
through every phase and exercise the whole pipeline.

This is NOT a substitute for the real live-FlightGear validation (spec
steps 2-6) -- it only proves the CONTROLLER's own code paths are wired
correctly; it cannot prove FlightGear itself launches or that the model's
commands actually fly the real aircraft. See the final report for what this
does and does not establish.
"""
from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runway import RunwayConfig, point_at_along_cross
from controller import config as C
from controller.protocol import OUTPUT_FIELDS


def make_telemetry_line(vals: dict) -> bytes:
    parts = []
    for c in OUTPUT_FIELDS:
        v = vals.get(c.name, 0)
        parts.append(str(int(v)) if c.type == "int" else str(float(v)))
    return (",".join(parts) + "\n").encode("ascii")


def run(telemetry_port: int, command_port: int, hz: float, runway_cfg: RunwayConfig, duration_s: float):
    out_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    in_sock.bind(("127.0.0.1", command_port))
    in_sock.settimeout(0.01)

    dt = 1.0 / hz
    t = 0.0
    along_m = -3000.0
    cross_m = 0.0
    heading = runway_cfg.true_heading_deg
    airspeed = 65.0
    gs = 65.0
    agl = 800.0 - runway_cfg.threshold_elevation_ft  # rough AGL near spawn (flat terrain assumption)
    touchdown_t = None
    n_sent = 0
    print(f"[mock-fg] streaming telemetry on 127.0.0.1:{telemetry_port}, reading commands on 127.0.0.1:{command_port}")

    last_cmd = {}
    while t < duration_s:
        # simple kinematic descent toward the runway along the centerline
        glideslope_rad = math.radians(runway_cfg.glideslope_deg)
        if agl > runway_cfg.flare_agl_ft:
            sink_fps = airspeed * 1.6878 * math.sin(glideslope_rad)  # kt->fps * sin(glide)
        else:
            sink_fps = 3.0  # gentle flare sink rate
        agl_new = agl - sink_fps * dt
        dist_traveled_m = gs * 0.5144 * dt
        along_m += dist_traveled_m
        agl = max(agl_new, 0.0)

        wow = agl <= 0.05
        if wow and touchdown_t is None:
            touchdown_t = t

        lat, lon = point_at_along_cross(along_m, cross_m, runway_cfg)
        msl = runway_cfg.threshold_elevation_ft + agl

        both_main = touchdown_t is not None and (t - touchdown_t) > 0.05
        vals = {
            "sim_time": t, "lat_deg": lat, "lon_deg": lon, "alt_msl_ft": msl, "alt_agl_ft": agl,
            "heading_deg": heading, "roll_deg": 0.0, "pitch_deg": 3.0, "alpha_deg": 4.0,
            "sideslip_rad": 0.0, "roll_rate_degps": 0.0, "pitch_rate_degps": 0.0, "yaw_rate_degps": 0.0,
            "airspeed_kt": airspeed, "vertical_speed_fps": -sink_fps if agl > 0 else 0.0,
            "vbody_fps": 0.0, "groundspeed_kt": gs,
            "aileron_pos": last_cmd.get("aileron", 0), "elevator_pos": last_cmd.get("elevator", 0),
            "rudder_pos": last_cmd.get("rudder", 0), "elevator_trim_pos": last_cmd.get("elevator_trim", 0.031),
            "flaps_pos": 0.99, "throttle_pos": last_cmd.get("throttle", 0.5),
            "gear_wow_nose": 0, "gear_wow_left": 1 if both_main else 0, "gear_wow_right": 1 if both_main else 0,
            "gear_compression_nose": 0.0, "gear_compression_left": 0.3 if both_main else 0.0,
            "gear_compression_right": 0.3 if both_main else 0.0,
            "wind_speed_kt": 0.0, "wind_from_heading_deg": 0.0, "crashed": 0, "frozen": 0,
        }
        out_sock.sendto(make_telemetry_line(vals), ("127.0.0.1", telemetry_port))
        n_sent += 1

        if touchdown_t is not None:
            gs = max(0.0, gs - 4.0 * dt)  # decelerate on rollout
            airspeed = gs

        try:
            data, _ = in_sock.recvfrom(4096)
            parts = data.decode("ascii").strip().split(",")
            names = ["aileron", "elevator", "rudder", "elevator_trim", "throttle", "brake_left", "brake_right"]
            last_cmd = {n: float(p) for n, p in zip(names, parts)}
        except socket.timeout:
            pass

        time.sleep(dt)
        t += dt
        if gs < 0.5 and touchdown_t is not None and (t - touchdown_t) > 5:
            print(f"[mock-fg] aircraft stopped at t={t:.1f}s, {n_sent} telemetry samples sent")
            break

    print(f"[mock-fg] done, sent {n_sent} telemetry samples")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry-port", type=int, default=C.TELEMETRY_PORT)
    p.add_argument("--command-port", type=int, default=C.COMMAND_PORT)
    p.add_argument("--hz", type=float, default=C.CONTROL_HZ)
    p.add_argument("--duration-s", type=float, default=90.0)
    p.add_argument("--runway-config", default=str(Path(__file__).resolve().parent.parent / "data/config/runway.yaml"))
    args = p.parse_args()
    rcfg = RunwayConfig.from_yaml(args.runway_config)
    run(args.telemetry_port, args.command_port, args.hz, rcfg, args.duration_s)
