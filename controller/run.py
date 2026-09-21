#!/usr/bin/env python3
"""Real-time FlightGear closed-loop controller entry point.

    python -m controller.run --launch-flightgear --scenario nominal --mode shadow
    python -m controller.run --launch-flightgear --scenario nominal --mode active

Startup sequence (spec):
    Launch FlightGear -> wait for valid telemetry -> verify/apply aircraft
    initialization -> disable autopilot -> load models and scaler ->
    validate feature calculation -> begin control at 10 Hz.

Emergency stop: Ctrl+C (SIGINT) at any time triggers a graceful shutdown
(rollout-safe neutral commands sent once, FlightGear terminated, summary
written with abort_reason="user_interrupt").
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.runway import RunwayConfig
from pipeline import paths as pipeline_paths
import yaml as _yaml

from controller import config as C
from controller.launcher import find_fgfs, write_protocol_file, build_fgfs_argv, FlightGearProcess
from controller.telemetry_io import TelemetryReceiver
from controller.command_io import CommandSender
from controller.features_online import OnlineFeatureBuilder, to_normalized_vectors
from controller.flight_manager import FlightManager
from controller.rollout_controller import RolloutController
from controller.models_runtime import ModelBundle, compose_elevator_command, saturate
from controller.safety import SafetyMonitor
from controller.run_logger import RunLogger


class RateLimiter:
    def __init__(self, limits: C.RateLimits):
        self.limits = limits
        self._last = {"aileron": 0.0, "rudder": 0.0, "elevator": 0.0, "throttle": None}

    def apply(self, name: str, target: float, max_delta: float) -> float:
        last = self._last[name]
        if last is None:
            self._last[name] = target
            return target
        delta = max(-max_delta, min(max_delta, target - last))
        out = last + delta
        self._last[name] = out
        return out

    def limit(self, aileron, rudder, elevator, throttle):
        return (
            self.apply("aileron", aileron, self.limits.aileron_per_cycle),
            self.apply("rudder", rudder, self.limits.rudder_per_cycle),
            self.apply("elevator", elevator, self.limits.elevator_per_cycle),
            self.apply("throttle", throttle, self.limits.throttle_per_cycle),
        )


def wait_for_valid_telemetry(receiver: TelemetryReceiver, timeout_s: float = 180.0,
                               required_consecutive: int = 5) -> bool:
    """Real readiness check (spec: not a fixed sleep): polls for a run of
    consecutive, finite, sane telemetry samples with advancing sim_time."""
    deadline = time.monotonic() + timeout_s
    consecutive = 0
    last_sim_time = None
    last_print = 0.0
    while time.monotonic() < deadline:
        t = receiver.latest()
        now = time.monotonic()
        if now - last_print > 5.0:
            print(f"[startup] waiting for valid telemetry... ({receiver.n_received} datagrams received so far)")
            last_print = now
        if t is not None and t.is_finite():
            sim_time = t.get("sim_time")
            sane = (
                sim_time is not None and sim_time > 0 and
                -90 <= t.get("lat_deg", 999) <= 90 and -180 <= t.get("lon_deg", 999) <= 180 and
                t.get("frozen", 1) == 0
            )
            advancing = last_sim_time is None or sim_time > last_sim_time
            if sane and advancing:
                consecutive += 1
                last_sim_time = sim_time
                if consecutive >= required_consecutive:
                    print(f"[startup] valid telemetry confirmed ({consecutive} consecutive samples, "
                          f"sim_time={sim_time:.2f}s)")
                    return True
            else:
                consecutive = 0
        time.sleep(0.1)
    return False


def build_argparser():
    p = argparse.ArgumentParser(description="Real-time FlightGear closed-loop baseline controller")
    p.add_argument("--launch-flightgear", action="store_true", help="Launch FlightGear directly (bypassing the launcher)")
    p.add_argument("--scenario", default="nominal", choices=list(C.SCENARIOS.keys()))
    p.add_argument("--mode", required=True, choices=["shadow", "active"],
                    help="shadow: predict+log only, never sends model commands. active: full closed-loop control.")
    p.add_argument("--fgfs-path", default=None, help="Explicit path to the fgfs/FlightGear executable")
    p.add_argument("--fg-root", default=str(C.DEFAULT_FG_ROOT))
    p.add_argument("--fg-home", default=str(C.DEFAULT_FG_HOME))
    p.add_argument("--telemetry-port", type=int, default=C.TELEMETRY_PORT)
    p.add_argument("--command-port", type=int, default=C.COMMAND_PORT)
    p.add_argument("--max-duration-s", type=float, default=600.0)
    p.add_argument("--startup-timeout-s", type=float, default=180.0)
    p.add_argument("--runway-config", default=str(pipeline_paths.CONFIG_DIR / "runway.yaml"))
    p.add_argument("--preprocessing-config", default=str(pipeline_paths.CONFIG_DIR / "preprocessing.yaml"))
    return p


class Controller:
    def __init__(self, args):
        self.args = args
        self.init = C.SCENARIOS[args.scenario]
        self.runway_cfg = RunwayConfig.from_yaml(args.runway_config)
        with open(args.preprocessing_config) as f:
            self._prep_cfg = _yaml.safe_load(f)
        floor_m = self._prep_cfg["glideslope_feature"]["floor_distance_to_threshold_m"]
        self.feature_builder = OnlineFeatureBuilder(self.runway_cfg, floor_m)
        fsm_cfg = C.FSMConfig()
        self.flight_manager = FlightManager(fsm_cfg, C.CONTROL_DT, self.runway_cfg.flare_agl_ft)
        self.rollout_controller = RolloutController(C.RolloutConfig(), self.runway_cfg.true_heading_deg)
        self.safety = SafetyMonitor(C.SafetyEnvelope(), self.runway_cfg.true_heading_deg)
        self.rate_limiter = RateLimiter(C.RateLimits())
        self.logger = RunLogger(args.mode, args.scenario)
        self.models = None
        self.fg_process = None
        self.telemetry_receiver = None
        self.command_sender = None
        self._abort_reason = None
        self._shutdown = False
        self._last_status_print = 0.0

    def load_models(self):
        self.models = ModelBundle()
        self.models.validate()
        print(f"[startup] loaded lateral checkpoint: {self.models.lateral.checkpoint_path}")
        print(f"[startup] loaded longitudinal checkpoint: {self.models.longitudinal.checkpoint_path}")
        print(f"[startup] loaded scaler: {self.models.scaler_path}")

    def validate_feature_calculation(self):
        """Sanity check the online feature pipeline against a synthetic
        telemetry sample before trusting it in the control loop."""
        fake = {
            "lat_deg": self.runway_cfg.threshold_lat_deg - 0.03, "lon_deg": self.runway_cfg.threshold_lon_deg,
            "heading_deg": self.runway_cfg.true_heading_deg, "alt_msl_ft": self.runway_cfg.threshold_elevation_ft + 800,
            "alt_agl_ft": 800, "airspeed_kt": 65, "roll_deg": 0, "pitch_deg": 3, "alpha_deg": 3,
            "sideslip_rad": 0, "roll_rate_degps": 0, "pitch_rate_degps": 0, "yaw_rate_degps": 0,
            "vertical_speed_fps": -10, "vbody_fps": 0, "flaps_pos": self.init.flaps_init,
        }
        raw = self.feature_builder.build_raw_features(fake)
        lat_vec, lon_vec = to_normalized_vectors(raw, self.models.scaler)
        assert len(lat_vec) == 11 and len(lon_vec) == 12
        assert all(abs(v) < 100 for v in lat_vec + lon_vec), "normalized features wildly out of range"
        print("[startup] online feature pipeline validated against a synthetic sample")

    def launch_flightgear(self):
        fgfs_path = find_fgfs(self.args.fgfs_path)
        fg_root = Path(self.args.fg_root)
        fg_home = Path(self.args.fg_home)
        write_protocol_file(fg_root)
        argv = build_fgfs_argv(fgfs_path, fg_root, fg_home, self.runway_cfg, self.init,
                                 self.args.telemetry_port, self.args.command_port)
        print(f"[startup] detected FlightGear executable: {fgfs_path}")
        print("[startup] launch command:\n  " + " ".join(argv))
        self.logger.log_launch_command(argv)
        self.fg_process = FlightGearProcess(argv, self.logger.fgfs_log_path())
        self.fg_process.launch()

    def start_io(self):
        self.telemetry_receiver = TelemetryReceiver(self.args.telemetry_port)
        self.telemetry_receiver.start()
        self.command_sender = CommandSender(self.args.command_port)
        self.command_sender.start()

    def log_init_conditions(self):
        from controller.launcher import compute_initial_position
        pos = compute_initial_position(self.runway_cfg, self.init)
        self.logger.log_init_conditions({
            "scenario": self.args.scenario, "mode": self.args.mode,
            "initial_position": pos, "init_condition": self.init,
            "weather_metar": self.init.metar,
            "runway_config_path": self.args.runway_config,
        })

    def send_neutral_and_stop(self):
        if self.command_sender:
            try:
                self.command_sender.send({"aileron": 0, "elevator": 0, "rudder": 0,
                                            "throttle": 0, "elevator_trim": self.init.elevator_trim_fixed,
                                            "brake_left": 1.0, "brake_right": 1.0})
            except Exception:
                pass

    def shutdown(self, abort_reason: str | None):
        self._abort_reason = abort_reason
        self.send_neutral_and_stop()
        if self.telemetry_receiver:
            self.telemetry_receiver.stop()
        if self.command_sender:
            self.command_sender.stop()
        if self.fg_process:
            print("[shutdown] terminating FlightGear...")
            self.fg_process.terminate()
        self.write_summary()
        self.logger.close()

    def write_summary(self):
        s = self.flight_manager.state
        touchdown_metrics = {}
        if s.touchdown_confirmed_t is not None:
            touchdown_metrics = getattr(self, "_touchdown_snapshot", {})
        summary = {
            "run_dir": str(self.logger.run_dir),
            "mode": self.args.mode, "scenario": self.args.scenario,
            "outcome": "aborted" if self._abort_reason else "completed",
            "abort_reason": self._abort_reason,
            "n_cycles": self.logger.n_cycles,
            "final_phase": s.phase,
            "touchdown_confirmed": s.touchdown_confirmed_t is not None,
            "touchdown_confirmed_t": s.touchdown_confirmed_t,
            "rollout_confirmed_t": s.rollout_confirmed_t,
            "n_bounce_events": s.n_bounces,
            "n_unconfirmed_contacts": s.n_unconfirmed_contacts,
            "touchdown_metrics": touchdown_metrics,
        }
        self.logger.log_transitions([
            {"t": tr.t, "previous_state": tr.previous_state, "new_state": tr.new_state, "reason": tr.reason}
            for tr in s.transitions
        ])
        self.logger.log_summary(summary)
        print("\n=== POST-FLIGHT SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print(f"Full log: {self.logger.run_dir}")

    def run_loop(self):
        cycle_period = C.CONTROL_DT
        start_wall = time.monotonic()
        last_cycle_start = None
        while not self._shutdown:
            cycle_start = time.monotonic()
            if last_cycle_start is not None:
                overrun = (cycle_start - last_cycle_start) - cycle_period
            else:
                overrun = 0.0
            last_cycle_start = cycle_start

            if cycle_start - start_wall > self.args.max_duration_s:
                self.shutdown("max_duration_reached")
                return

            telem = self.telemetry_receiver.latest()
            last_recv = telem.received_wall_time if telem else None
            decision = self.safety.check_telemetry(telem, last_recv)
            if decision.should_abort:
                if telem is None:
                    time.sleep(0.05)
                    continue
                self.shutdown(decision.reason)
                return

            fields = telem.fields
            raw = self.feature_builder.build_raw_features(fields)
            lat_vec, lon_vec = to_normalized_vectors(raw, self.models.scaler)

            phase = self.flight_manager.step(
                t=fields["sim_time"], agl_ft=fields["alt_agl_ft"],
                wow_nose=bool(fields["gear_wow_nose"]), wow_left=bool(fields["gear_wow_left"]),
                wow_right=bool(fields["gear_wow_right"]), groundspeed_kt=fields["groundspeed_kt"],
            )

            lat_pred = self.models.lateral.predict(lat_vec)
            lon_pred = self.models.longitudinal.predict(lon_vec)
            dm = self.safety.check_model_output("lateral", lat_pred)
            if dm.should_abort:
                self.shutdown(dm.reason); return
            dm = self.safety.check_model_output("longitudinal", lon_pred)
            if dm.should_abort:
                self.shutdown(dm.reason); return

            aileron_raw, rudder_raw = lat_pred
            elevator_eff_raw, throttle_raw = lon_pred
            elevator_raw = compose_elevator_command(elevator_eff_raw, self.init.elevator_trim_fixed)

            aileron_sat, ail_s = saturate(aileron_raw, *self.models.lateral.output_saturation["aileron_command"])
            rudder_sat, rud_s = saturate(rudder_raw, *self.models.lateral.output_saturation["rudder_command"])
            elevator_sat, elv_s = saturate(elevator_raw, -1.0, 1.0)
            throttle_sat, thr_s = saturate(throttle_raw, *self.models.longitudinal.output_saturation["throttle_command"])

            saturation_flags = {"aileron": ail_s, "rudder": rud_s, "elevator": elv_s, "throttle": thr_s}
            abort = None
            for ch, sat in saturation_flags.items():
                d = self.safety.track_saturation(ch, sat)
                if d.should_abort:
                    abort = d.reason
            if self.flight_manager.is_neural_controlled():
                # min/max airspeed, bank, pitch envelope only applies while
                # the MLPs are flying -- during ROLLOUT/COMPLETE, decelerating
                # toward a stop (airspeed -> 0) is the CORRECT, intended
                # behavior of the deterministic ground controller, not a
                # safety violation.
                d = self.safety.check_envelope(
                    cross_track_m=raw["cross_track_error_m"],
                    heading_error_deg=(fields["heading_deg"] - self.runway_cfg.true_heading_deg + 180) % 360 - 180,
                    roll_deg=fields["roll_deg"], pitch_deg=fields["pitch_deg"],
                    agl_ft=fields["alt_agl_ft"], airspeed_kt=fields["airspeed_kt"],
                )
                if d.should_abort:
                    abort = d.reason
            d = self.safety.check_crashed(fields.get("crashed", 0))
            if d.should_abort:
                abort = d.reason

            if self.flight_manager.is_ground_controlled():
                if self.flight_manager.state.phase == "ROLLOUT" and self.rollout_controller._rollout_start_t is None:
                    self.rollout_controller.reset(fields["sim_time"])
                cmd = self.rollout_controller.compute(fields["sim_time"], fields["heading_deg"], fields["groundspeed_kt"])
            else:
                a, r, e, th = self.rate_limiter.limit(aileron_sat, rudder_sat, elevator_sat, throttle_sat)
                cmd = {"aileron": a, "rudder": r, "elevator": e, "throttle": th,
                       "brake_left": 0.0, "brake_right": 0.0}
            cmd["elevator_trim"] = self.init.elevator_trim_fixed

            if self.args.mode == "active":
                self.command_sender.send(cmd)

            if cycle_start - self._last_status_print > 1.0:
                self._last_status_print = cycle_start
                dist_m = -raw["along_track_m"]
                print(f"t={fields['sim_time']:7.1f}s  {phase:26s} dist={dist_m:6.0f}m  "
                      f"agl={fields['alt_agl_ft']:6.0f}ft  xtk={raw['cross_track_error_m']:+6.1f}m  "
                      f"ias={fields['airspeed_kt']:5.1f}kt  vs={fields['vertical_speed_fps']*60:+6.0f}fpm  "
                      f"gs={fields['groundspeed_kt']:5.1f}kt || "
                      f"ail={cmd['aileron']:+.2f} elev={cmd['elevator']:+.2f} rud={cmd['rudder']:+.2f} "
                      f"thr={cmd['throttle']:.2f} brk={cmd['brake_left']:.2f}"
                      + ("  [SHADOW: not sent]" if self.args.mode == "shadow" else ""))

            if phase == "TOUCHDOWN_CONFIRMED" and not hasattr(self, "_touchdown_snapshot"):
                self._touchdown_snapshot = {
                    "t": fields["sim_time"], "lat": fields["lat_deg"], "lon": fields["lon_deg"],
                    "cross_track_error_m": raw["cross_track_error_m"],
                    "heading_error_deg": (fields["heading_deg"] - self.runway_cfg.true_heading_deg + 180) % 360 - 180,
                    "airspeed_kt": fields["airspeed_kt"], "vertical_speed_fps": fields["vertical_speed_fps"],
                    "roll_deg": fields["roll_deg"], "pitch_deg": fields["pitch_deg"],
                }

            compute_latency = time.monotonic() - cycle_start
            self.logger.log_cycle({
                "cycle": self.logger.n_cycles, "sim_time": fields["sim_time"],
                "wall_time": cycle_start, "loop_overrun_s": overrun, "compute_latency_s": compute_latency,
                "raw_telemetry": fields, "raw_features": raw,
                "normalized_lateral": lat_vec, "normalized_longitudinal": lon_vec,
                "phase": phase,
                "raw_predictions": {"aileron": aileron_raw, "rudder": rudder_raw,
                                      "elevator_effective": elevator_eff_raw, "throttle": throttle_raw},
                "commands_sent": cmd if self.args.mode == "active" else None,
                "commands_shadow_only": None if self.args.mode == "active" else cmd,
                "saturation": saturation_flags,
            })

            if abort:
                self.shutdown(abort)
                return

            if phase == "COMPLETE":
                if fields["groundspeed_kt"] < C.RolloutConfig().stop_groundspeed_kt:
                    if not hasattr(self, "_stopped_since"):
                        self._stopped_since = cycle_start
                    elif cycle_start - self._stopped_since > 3.0:
                        self.shutdown(None)  # normal completion
                        return
                else:
                    if hasattr(self, "_stopped_since"):
                        del self._stopped_since

            elapsed = time.monotonic() - cycle_start
            sleep_for = cycle_period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif overrun > self.safety.env.max_loop_overrun_s:
                print(f"[warn] control loop overrun: {overrun:.3f}s")

    def request_shutdown(self):
        self._shutdown = True


def main():
    args = build_argparser().parse_args()
    ctrl = Controller(args)

    def handle_sigint(signum, frame):
        print("\n[abort] Ctrl+C received -- emergency stop")
        ctrl.request_shutdown()
        ctrl.shutdown("user_interrupt")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"[startup] mode={args.mode} scenario={args.scenario} run_dir={ctrl.logger.run_dir}")
    ctrl.log_init_conditions()
    ctrl.load_models()
    ctrl.validate_feature_calculation()

    if args.launch_flightgear:
        ctrl.launch_flightgear()
    else:
        print("[startup] --launch-flightgear not given -- expecting FlightGear already running with matching --generic ports")

    ctrl.start_io()

    ok = wait_for_valid_telemetry(ctrl.telemetry_receiver, timeout_s=args.startup_timeout_s)
    if not ok:
        print("[abort] never received valid telemetry within the startup timeout")
        ctrl.shutdown("startup_telemetry_timeout")
        sys.exit(1)

    print("[startup] beginning 10 Hz control loop")
    ctrl.run_loop()


if __name__ == "__main__":
    main()
