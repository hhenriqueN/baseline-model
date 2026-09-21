"""Canonical property-tree column names used across the pipeline.

Names are the literal header strings found in the raw CSVs. Confirmed by
manual inspection of the header of voo-condicoes-normais.csv on 2026-09-17
(247 columns). If any of the "required" columns below is missing from a raw
file, the pipeline must raise a clear error rather than silently continuing.
"""

# --- metadata / traceability -------------------------------------------------
FRAME = "frame"
SIM_TIME = "sim_time"

# --- position -----------------------------------------------------------------
LON = "/position[0]/longitude-deg[0]"
LAT = "/position[0]/latitude-deg[0]"
ALT_MSL_FT = "/position[0]/altitude-ft[0]"
ALT_AGL_FT = "/position[0]/altitude-agl-ft[0]"
ALT_AGL_M = "/position[0]/altitude-agl-m[0]"          # duplicate of ALT_AGL_FT (unit)
GROUND_ELEV_M = "/position[0]/ground-elev-m[0]"

# --- orientation ---------------------------------------------------------------
HEADING_DEG = "/orientation[0]/heading-deg[0]"
ROLL_DEG = "/orientation[0]/roll-deg[0]"
PITCH_DEG = "/orientation[0]/pitch-deg[0]"
ALPHA_DEG = "/orientation[0]/alpha-deg[0]"
SIDESLIP_RAD = "/orientation[0]/side-slip-rad[0]"
BETA_DEG = "/orientation[0]/beta-deg[0]"               # duplicate of SIDESLIP_RAD (unit, sign TBD)
ROLL_RATE_DEGPS = "/orientation[0]/roll-rate-degps[0]"
PITCH_RATE_DEGPS = "/orientation[0]/pitch-rate-degps[0]"
YAW_RATE_DEGPS = "/orientation[0]/yaw-rate-degps[0]"

# --- velocities ------------------------------------------------------------------
MACH = "/velocities[0]/mach[0]"
AIRSPEED_KT = "/velocities[0]/airspeed-kt[0]"
VERTICAL_SPEED_FPS = "/velocities[0]/vertical-speed-fps[0]"
SPEED_NORTH_FPS = "/velocities[0]/speed-north-fps[0]"
SPEED_EAST_FPS = "/velocities[0]/speed-east-fps[0]"
SPEED_DOWN_FPS = "/velocities[0]/speed-down-fps[0]"    # duplicate candidate of VERTICAL_SPEED_FPS (sign)
VBODY_FPS = "/velocities[0]/vBody-fps[0]"
UBODY_FPS = "/velocities[0]/uBody-fps[0]"
WBODY_FPS = "/velocities[0]/wBody-fps[0]"
GROUNDSPEED_KT = "/velocities[0]/groundspeed-kt[0]"
VIAS_KT = "/fdm[0]/jsbsim[0]/velocities[0]/vias-kts[0]"  # duplicate candidate of AIRSPEED_KT

# --- controls (pilot commands) ----------------------------------------------------
AILERON_CMD = "/controls[0]/flight[0]/aileron[0]"
ELEVATOR_CMD = "/controls[0]/flight[0]/elevator[0]"
RUDDER_CMD = "/controls[0]/flight[0]/rudder[0]"
AILERON_TRIM = "/controls[0]/flight[0]/aileron-trim[0]"
ELEVATOR_TRIM = "/controls[0]/flight[0]/elevator-trim[0]"
RUDDER_TRIM = "/controls[0]/flight[0]/rudder-trim[0]"
FLAPS_CMD = "/controls[0]/flight[0]/flaps[0]"
SPEEDBRAKE = "/controls[0]/flight[0]/speedbrake[0]"
SPOILERS = "/controls[0]/flight[0]/spoilers[0]"

THROTTLE_ENGINE0 = "/controls[0]/engines[0]/engine[0]/throttle[0]"
THROTTLE_ENGINE1 = "/controls[0]/engines[0]/engine[1]/throttle[0]"

# --- brakes / steering (ground only, never neural input) ---------------------------
BRAKE_LEFT = "/controls[0]/gear[0]/brake-left[0]"
BRAKE_RIGHT = "/controls[0]/gear[0]/brake-right[0]"
BRAKE_PARKING = "/controls[0]/gear[0]/brake-parking[0]"

# --- surface positions (leakage-only, never neural input) ----------------------------
SURF_ELEVATOR = "/surface-positions[0]/elevator-pos-norm[0]"
SURF_FLAP = "/surface-positions[0]/flap-pos-norm[0]"
SURF_AILERON_L = "/surface-positions[0]/left-aileron-pos-norm[0]"
SURF_AILERON_R = "/surface-positions[0]/right-aileron-pos-norm[0]"
SURF_RUDDER = "/surface-positions[0]/rudder-pos-norm[0]"

# --- cockpit interface (leakage-only, never neural input) -----------------------------
YOKE_AILERON = "/sim[0]/model[0]/c172p[0]/cockpit[0]/yoke-aileron[0]"
YOKE_ELEVATOR = "/sim[0]/model[0]/c172p[0]/cockpit[0]/yoke-elevator[0]"
PEDALS_RUDDER = "/sim[0]/model[0]/c172p[0]/cockpit[0]/pedals-rudder[0]"
FLAPS_LEVER = "/sim[0]/model[0]/c172p[0]/cockpit[0]/flaps-lever[0]"

# --- gear (indices confirmed empirically: gear[0]=NOSE (steers), gear[1] & gear[2]=MAIN) --
def gear_wow(i: int) -> str:
    return f"/gear[0]/gear[{i}]/wow[0]"


def gear_compression(i: int) -> str:
    return f"/gear[0]/gear[{i}]/compression-norm[0]"


def gear_rollspeed_ms(i: int) -> str:
    return f"/gear[0]/gear[{i}]/rollspeed-ms[0]"


def gear_steering(i: int) -> str:
    return f"/gear[0]/gear[{i}]/steering-norm[0]"


GEAR_NOSE_IDX = 0
GEAR_MAIN_IDX = (1, 2)

# --- environment (for scenario_id confirmation, not a neural input) -------------------
WIND_SPEED_KT = "/environment[0]/wind-speed-kt[0]"
WIND_FROM_HEADING_DEG = "/environment[0]/wind-from-heading-deg[0]"
TURBULENCE_MAGNITUDE = "/environment[0]/turbulence[0]/magnitude-norm[0]"

# ---------------------------------------------------------------------------------------
REQUIRED_SOURCE_COLUMNS = [
    FRAME, SIM_TIME,
    LON, LAT, ALT_MSL_FT, ALT_AGL_FT,
    HEADING_DEG, ROLL_DEG, PITCH_DEG, ALPHA_DEG, SIDESLIP_RAD,
    ROLL_RATE_DEGPS, PITCH_RATE_DEGPS, YAW_RATE_DEGPS,
    AIRSPEED_KT, VERTICAL_SPEED_FPS, VBODY_FPS, GROUNDSPEED_KT,
    AILERON_CMD, ELEVATOR_CMD, ELEVATOR_TRIM, RUDDER_CMD, FLAPS_CMD,
    THROTTLE_ENGINE0,
    gear_wow(0), gear_wow(1), gear_wow(2),
    gear_compression(0), gear_compression(1), gear_compression(2),
    gear_rollspeed_ms(0), gear_rollspeed_ms(1), gear_rollspeed_ms(2),
]

# Columns that must NEVER be used as neural network inputs (section 10 + 11 of spec).
FORBIDDEN_AS_INPUT_EXACT = {
    FRAME, SIM_TIME, LAT, LON,
    SURF_ELEVATOR, SURF_FLAP, SURF_AILERON_L, SURF_AILERON_R, SURF_RUDDER,
    YOKE_AILERON, YOKE_ELEVATOR, PEDALS_RUDDER, FLAPS_LEVER,
    AILERON_TRIM, RUDDER_TRIM, ELEVATOR_TRIM,
    BRAKE_LEFT, BRAKE_RIGHT, BRAKE_PARKING,
    gear_wow(0), gear_wow(1), gear_wow(2),
    gear_compression(0), gear_compression(1), gear_compression(2),
    gear_rollspeed_ms(0), gear_rollspeed_ms(1), gear_rollspeed_ms(2),
    gear_steering(0), gear_steering(1), gear_steering(2),
}
# Substring patterns for the broad prohibited families (cabin, lights, engine internals, etc.)
FORBIDDEN_AS_INPUT_SUBSTRINGS = [
    "surface-positions", "yoke", "pedals", "flaps-lever",
    "brake", "steering-norm", "wow[0]", "compression-norm", "rollspeed-ms",
    "lighting", "door-positions", "circuit-breakers", "damage", "serviceable",
    "crash", "electrical", "fuel", "rpm", "egt", "oil-", "manifold", "mp-osi",
    "battery", "vacuum", "static", "pitot", "autopilot", "nav[", "adf",
    "magneto", "starter", "switches", "anti-ice", "hazards", "cabin",
    "fog", "frost", "icing", "alarms", "faults", "kill-engine", "killed",
    "cranking", "coughing", "exhaust", "instrumentation",
]
