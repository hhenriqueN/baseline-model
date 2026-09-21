"""FlightGear "generic" I/O protocol definition (spec: telemetry + command
interfaces). One field list (OUTPUT_FIELDS / INPUT_FIELDS) is the single
source of truth for both the XML protocol file FlightGear loads and the
Python encoder/decoder below -- they cannot drift apart.

Property paths mirror pipeline/columns.py's FlightGear property names
one-to-one (same physical signals used to build the training data), so the
online feature pipeline can reuse pipeline.runway/pipeline.features exactly.

Communication method: UDP, ASCII "generic" protocol (comma-separated),
matching FlightGear's own Protocol/*.xml examples (see
Protocol/aura-imu.xml, Protocol/avare.xml on a standard FlightGear
install). Chosen over the telnet property-tree interface (also available,
used by the project's own prior interactive sessions per
~/Library/Application Support/FlightGear/fgfs.log) because telnet is a
request/response-per-property protocol unsuited to a 10 Hz, ~35-field
control loop; the generic protocol is a fire-and-forget stream FlightGear's
own I/O manager drives at the requested rate with no per-field round trip.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    name: str
    node: str
    type: str = "double"
    factor: float | None = None


# ---- Output: FlightGear -> controller (telemetry) ----
OUTPUT_FIELDS: list[Chunk] = [
    Chunk("sim_time", "/sim/time/elapsed-sec"),
    Chunk("lat_deg", "/position/latitude-deg"),
    Chunk("lon_deg", "/position/longitude-deg"),
    Chunk("alt_msl_ft", "/position/altitude-ft"),
    Chunk("alt_agl_ft", "/position/altitude-agl-ft"),
    Chunk("heading_deg", "/orientation/heading-deg"),
    Chunk("roll_deg", "/orientation/roll-deg"),
    Chunk("pitch_deg", "/orientation/pitch-deg"),
    Chunk("alpha_deg", "/orientation/alpha-deg"),
    Chunk("sideslip_rad", "/orientation/side-slip-rad"),
    Chunk("roll_rate_degps", "/orientation/roll-rate-degps"),
    Chunk("pitch_rate_degps", "/orientation/pitch-rate-degps"),
    Chunk("yaw_rate_degps", "/orientation/yaw-rate-degps"),
    Chunk("airspeed_kt", "/velocities/airspeed-kt"),
    Chunk("vertical_speed_fps", "/velocities/vertical-speed-fps"),
    Chunk("vbody_fps", "/velocities/vBody-fps"),
    Chunk("groundspeed_kt", "/velocities/groundspeed-kt"),
    Chunk("aileron_pos", "/controls/flight/aileron"),
    Chunk("elevator_pos", "/controls/flight/elevator"),
    Chunk("rudder_pos", "/controls/flight/rudder"),
    Chunk("elevator_trim_pos", "/controls/flight/elevator-trim"),
    Chunk("flaps_pos", "/controls/flight/flaps"),
    Chunk("throttle_pos", "/controls/engines/engine[0]/throttle"),
    Chunk("gear_wow_nose", "/gear/gear[0]/wow", type="int"),
    Chunk("gear_wow_left", "/gear/gear[1]/wow", type="int"),
    Chunk("gear_wow_right", "/gear/gear[2]/wow", type="int"),
    Chunk("gear_compression_nose", "/gear/gear[0]/compression-norm"),
    Chunk("gear_compression_left", "/gear/gear[1]/compression-norm"),
    Chunk("gear_compression_right", "/gear/gear[2]/compression-norm"),
    Chunk("wind_speed_kt", "/environment/wind-speed-kt"),
    Chunk("wind_from_heading_deg", "/environment/wind-from-heading-deg"),
    # /fdm/jsbsim/crash, not /sim/crashed: confirmed by reading the installed
    # c172p's own Nasal/physics.nas -- this is the property JSBSim's ground-
    # impact detector actually sets for THIS aircraft (its crash listener
    # kills the engine off of it); /sim/crashed is a more generic engine-level
    # flag not reliably populated by every aircraft's damage model.
    Chunk("crashed", "/fdm/jsbsim/crash", type="int"),
    Chunk("frozen", "/sim/freeze/master", type="int"),
]

# ---- Input: controller -> FlightGear (commands) ----
INPUT_FIELDS: list[Chunk] = [
    Chunk("aileron", "/controls/flight/aileron"),
    Chunk("elevator", "/controls/flight/elevator"),
    Chunk("rudder", "/controls/flight/rudder"),
    Chunk("elevator_trim", "/controls/flight/elevator-trim"),
    Chunk("throttle", "/controls/engines/engine[0]/throttle"),
    Chunk("brake_left", "/controls/gear/brake-left"),
    Chunk("brake_right", "/controls/gear/brake-right"),
]

VAR_SEP = ","


def build_protocol_xml() -> str:
    def chunk_xml(c: Chunk) -> str:
        factor_line = f"\n      <factor>{c.factor}</factor>" if c.factor is not None else ""
        # an explicit <format> is required: without it FlightGear's ASCII
        # generic writer prints doubles with an integer conversion (garbage).
        fmt = "%d" if c.type == "int" else "%.8f"
        return (
            "    <chunk>\n"
            f"      <name>{c.name}</name>\n"
            f"      <type>{c.type}</type>\n"
            f"      <format>{fmt}</format>\n"
            f"      <node>{c.node}</node>{factor_line}\n"
            "    </chunk>"
        )

    output_chunks = "\n".join(chunk_xml(c) for c in OUTPUT_FIELDS)
    input_chunks = "\n".join(chunk_xml(c) for c in INPUT_FIELDS)
    return f"""<?xml version="1.0"?>
<!-- Auto-generated by controller/protocol.py. Do not hand-edit; regenerate
     via controller.launcher.write_protocol_file(). Single source of truth
     is OUTPUT_FIELDS / INPUT_FIELDS in controller/protocol.py. -->
<PropertyList>
  <generic>
    <output>
      <line_separator>newline</line_separator>
      <var_separator>{VAR_SEP}</var_separator>
{output_chunks}
    </output>
    <input>
      <line_separator>newline</line_separator>
      <var_separator>{VAR_SEP}</var_separator>
{input_chunks}
    </input>
  </generic>
</PropertyList>
"""


def encode_command_line(values: dict) -> bytes:
    """values: {field_name: number}. Missing fields are sent as 0.0."""
    parts = [str(float(values.get(c.name, 0.0))) for c in INPUT_FIELDS]
    return (VAR_SEP.join(parts) + "\n").encode("ascii")


def decode_telemetry_line(line: str) -> dict | None:
    parts = line.strip().split(VAR_SEP)
    if len(parts) != len(OUTPUT_FIELDS):
        return None
    out = {}
    try:
        for c, p in zip(OUTPUT_FIELDS, parts):
            out[c.name] = int(p) if c.type == "int" else float(p)
    except ValueError:
        return None
    return out
