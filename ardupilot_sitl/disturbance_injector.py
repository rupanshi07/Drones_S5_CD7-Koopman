"""
disturbance_injector.py
-----------------------
Inject payload and wind disturbances into ArduPilot SITL via MAVLink.

Disturbance schedule matches Phase-2 (v2_stable_koopman_mpc/comparison/run_comparison.py):
  PAYLOAD_FRAC       = 0.25   (+25% mass)
  PAYLOAD_START_SEC  = 3.0 s  (payload applied at t=3s)
  WIND_START_SEC     = 3.0 s
  WIND_DURATION_SEC  = 2.0 s
  WIND_FORCE_N       = 0.02 N
  EPISODE_LEN_SEC    = 8.0 s  (Phase-2 default)

For SITL (15s default episode), the payload_onset_frac is scaled to produce
the same 3/8 = 0.375 point of the episode, so payload always drops at 37.5%
into each run (consistent with Phase-2's 3s into an 8s episode).

ArduPilot SITL supports injecting forces/wind via:
  1. SIM_WIND_SPD / SIM_WIND_DIR parameters
  2. Payload is modelled as thrust bias (reduces available headroom)
"""

import time
import numpy as np


class DisturbanceInjector:
    """
    Manages payload and wind disturbances for SITL simulation.

    Wind: Set via ArduPilot SIM_WIND_SPD and SIM_WIND_DIR parameters.
          These are supported natively by ArduCopter SITL.

    Payload: Modeled as an extra downward gravity component. We tell the
             controller that effective gravity is (1 + payload_frac) * 9.81,
             which reduces the normalized thrust headroom. This is consistent
             with Phase 2 (where pybullet mass was increased).
    """

    def __init__(self, mav_interface):
        """
        Parameters
        ----------
        mav_interface : MAVLinkInterface instance (already connected)
        """
        self.mav = mav_interface

        self._wind_active = False
        self._payload_active = False
        self._payload_frac = 0.0
        self._wind_speed = 0.0
        self._wind_dir_deg = 0.0

    # ----------------------------------------------------------------------- #
    # Wind
    # ----------------------------------------------------------------------- #
    def set_wind(self, speed_ms, direction_deg=0.0):
        """
        Set SITL wind via ArduPilot parameters.

        Parameters
        ----------
        speed_ms      : float  wind speed in m/s (0 to ~10 m/s)
        direction_deg : float  wind direction in degrees (0 = North/X)
        """
        self.mav.set_param("SIM_WIND_SPD", speed_ms)
        self.mav.set_param("SIM_WIND_DIR", direction_deg)
        # ArduPilot also supports SIM_WIND_TURB for turbulence
        self.mav.set_param("SIM_WIND_TURB", speed_ms * 0.1)

        self._wind_active = speed_ms > 0
        self._wind_speed = speed_ms
        self._wind_dir_deg = direction_deg
        print(f"[Disturbance] Wind: {speed_ms:.2f} m/s at {direction_deg:.1f} deg")

    def clear_wind(self):
        self.mav.set_param("SIM_WIND_SPD", 0.0)
        self.mav.set_param("SIM_WIND_TURB", 0.0)
        self._wind_active = False
        self._wind_speed = 0.0
        print("[Disturbance] Wind cleared.")

    # ----------------------------------------------------------------------- #
    # Payload
    # ----------------------------------------------------------------------- #
    def apply_payload(self, payload_frac=0.25):
        """
        Simulate payload as additional mass fraction.

        Parameters
        ----------
        payload_frac : float  fraction of drone mass added as payload (0.25 = +25%)

        Effect: the controller's gravity-compensation thrust needs to be scaled
        by (1 + payload_frac). We expose this as get_effective_gravity() so
        controllers can compensate (or not, to test disturbance rejection).
        """
        self._payload_active = True
        self._payload_frac = payload_frac
        print(f"[Disturbance] Payload applied: +{payload_frac*100:.0f}% mass")

    def clear_payload(self):
        self._payload_active = False
        self._payload_frac = 0.0
        print("[Disturbance] Payload cleared.")

    # ----------------------------------------------------------------------- #
    # Properties for controllers to query
    # ----------------------------------------------------------------------- #
    @property
    def effective_gravity(self):
        """Effective gravity accounting for payload (for thrust compensation)."""
        return 9.81 * (1.0 + self._payload_frac)

    @property
    def payload_frac(self):
        return self._payload_frac

    @property
    def wind_active(self):
        return self._wind_active

    @property
    def wind_speed(self):
        """Current wind speed in m/s (public property for clean access)."""
        return self._wind_speed

    @property
    def wind_dir_deg(self):
        """Current wind direction in degrees."""
        return self._wind_dir_deg

    @property
    def payload_active(self):
        return self._payload_active

    def get_state_dict(self):
        """Return current disturbance state for logging."""
        return {
            "wind_active": self._wind_active,
            "wind_speed_ms": self._wind_speed,
            "wind_dir_deg": self._wind_dir_deg,
            "payload_active": self._payload_active,
            "payload_frac": self._payload_frac,
        }


# --------------------------------------------------------------------------- #
# Scenario definitions -- IDENTICAL to Phase-2 run_comparison.py
#
# Phase-2 constants (for reference):
#   PAYLOAD_FRAC      = 0.25
#   PAYLOAD_START_SEC = 3.0 s  (out of 8.0 s episode = 0.375 of episode)
#   WIND_START_SEC    = 3.0 s
#   WIND_DURATION_SEC = 2.0 s
#   WIND_FORCE_N      = 0.02 N  (~ 3 m/s wind on a CF2X-like drone)
#
# payload_onset_frac = 0.375 ensures payload drops at 37.5% into each
# episode regardless of --duration, matching Phase-2's 3s/8s = 0.375.
# --------------------------------------------------------------------------- #

SCENARIOS = {
    "clean": {
        "description": "Nominal flight, no disturbances",
        "wind_speed": 0.0,
        "wind_dir": 0.0,
        "payload_frac": 0.0,
        "payload_onset_frac": None,
    },
    "wind": {
        "description": "Wind only (3 m/s, ~0.02N on CF2X -- matches Phase-2 WIND_FORCE_N)",
        "wind_speed": 3.0,
        "wind_dir": 0.0,    # fixed direction for repeatability (Phase-2 used [1,0,0])
        "payload_frac": 0.0,
        "payload_onset_frac": None,
    },
    "payload": {
        "description": "Payload only (+25% mass at 37.5% of episode -- matches Phase-2)",
        "wind_speed": 0.0,
        "wind_dir": 0.0,
        "payload_frac": 0.25,          # matches Phase-2 PAYLOAD_FRAC
        "payload_onset_frac": 0.375,   # 3s / 8s = 0.375, matches Phase-2
    },
    "combined": {
        "description": "Wind + payload together (matches Phase-2 payload+wind condition)",
        "wind_speed": 3.0,
        "wind_dir": 0.0,
        "payload_frac": 0.25,
        "payload_onset_frac": 0.375,
    },
}
