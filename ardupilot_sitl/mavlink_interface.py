"""
mavlink_interface.py
--------------------
MAVLink connection manager for ArduPilot SITL on Windows.

Handles:
  - TCP connection to ArduCopter SITL (Mission Planner starts it on port 5760)
  - State extraction: position (NED), velocity (NED), attitude quaternion,
    angular velocity -- converted to ENU for the Koopman model
  - Command sending: SET_ATTITUDE_TARGET (roll/pitch/yaw rate + thrust)
  - Heartbeat maintenance (required by ArduPilot to stay in GUIDED mode)

Frame conventions
-----------------
ArduPilot uses NED (North-East-Down). Our Phase 2 Koopman model was trained in
the gym-pybullet-drones frame which is ENU-like (x forward, y left, z up).

We do the conversion HERE so the rest of the codebase doesn't need to know.

NED -> ENU:  x_enu =  y_ned
             y_enu =  x_ned
             z_enu = -z_ned
"""

import time
import threading
import numpy as np
from pymavlink import mavutil

# --------------------------------------------------------------------------- #
# Connection defaults
# --------------------------------------------------------------------------- #
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5760          # Mission Planner SITL default TCP port
HEARTBEAT_RATE_HZ = 5        # ArduPilot needs ≥1 Hz heartbeat to stay guided


class MAVLinkInterface:
    """
    Thin wrapper around pymavlink for reading state and sending
    SET_ATTITUDE_TARGET commands to ArduCopter SITL.
    """

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT,
                 source_system=255, source_component=0, use_udp=False):
        self.host = host
        self.port = port
        self.use_udp = use_udp          # True = UDP client, False = TCP client
        self.source_system = source_system
        self.source_component = source_component

        self.conn = None          # pymavlink MAVLink connection
        self._lock = threading.Lock()

        # Latest telemetry (updated by background thread)
        self._pos_ned = np.zeros(3)           # NED position (m)
        self._vel_ned = np.zeros(3)           # NED velocity (m/s)
        self._quat_ned = np.array([1, 0, 0, 0])  # w, x, y, z
        self._ang_vel_body = np.zeros(3)      # rad/s, body frame
        self._armed = False
        self._mode = "UNKNOWN"

        self._heartbeat_thread = None
        self._telemetry_thread = None
        self._running = False

    # ----------------------------------------------------------------------- #
    # Connection
    # ----------------------------------------------------------------------- #
    def connect(self, timeout=30.0, target_system=1):
        """
        Connect to SITL via TCP.  Mission Planner must have started SITL first.

        Note
        ----
        Mission Planner proxies port 5760 and injects its own GCS heartbeat
        (sysid=0). We use wait_heartbeat() to receive the first heartbeat
        (any sysid), then immediately force target_system=1 so all commands
        go to ArduCopter, not Mission Planner's proxy.
        """
        if self.use_udp:
            conn_str = f"udpin:0.0.0.0:{self.port}"
            print(f"[MAVLink] Listening for UDP MAVLink on port {self.port} ...")
            print(f"          (Mission Planner must be forwarding to 127.0.0.1:{self.port})")
        else:
            conn_str = f"tcp:{self.host}:{self.port}"
            print(f"[MAVLink] Connecting to tcp:{self.host}:{self.port} ...")
        try:
            self.conn = mavutil.mavlink_connection(
                conn_str,
                source_system=self.source_system,
                source_component=self.source_component,
            )
        except Exception as e:
            raise ConnectionError(
                f"[MAVLink] Cannot open connection {conn_str}.\n"
                f"  -> Make sure SITL is running in Mission Planner first.\n"
                f"  Original error: {e}"
            )

        # wait_heartbeat() receives the first MAVLink heartbeat on the stream.
        # With Mission Planner SITL this is usually its own GCS heartbeat (sysid=0).
        # That is fine — we override target_system right after.
        print("[MAVLink] Waiting for heartbeat ...")
        hb = self.conn.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise ConnectionError(
                f"[MAVLink] No heartbeat within {timeout}s.\n"
                f"  -> Make sure Mission Planner SITL is running and the drone\n"
                f"     is visible on the HUD (Simulation -> Multirotor -> Run SITL)."
            )
        print(f"[MAVLink] Heartbeat received from sysid={self.conn.target_system}")

        # CRITICAL: Force target to ArduCopter (sysid=1) BEFORE sending any
        # data stream requests or commands.
        self.conn.target_system    = target_system
        self.conn.target_component = 1
        print(f"[MAVLink] Targeting ArduCopter: sysid={target_system}, compid=1")

        # Request data streams we need
        self._request_data_streams()

        # Start background threads
        self._running = True
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop, daemon=True
        )
        self._telemetry_thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

        # Give it a moment to fill in initial state
        time.sleep(1.5)
        print("[MAVLink] Connected and streaming telemetry.")

    def close(self):
        self._running = False
        if self.conn:
            self.conn.close()

    # ----------------------------------------------------------------------- #
    # Data stream requests
    # ----------------------------------------------------------------------- #
    def _request_data_streams(self):
        """Request high-rate position, attitude, and velocity streams."""
        streams = [
            (mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS, 50),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 10),
            (mavutil.mavlink.MAV_DATA_STREAM_POSITION, 50),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 50),
            (mavutil.mavlink.MAV_DATA_STREAM_EXTRA2, 50),
        ]
        for stream_id, rate in streams:
            self.conn.mav.request_data_stream_send(
                self.conn.target_system,
                self.conn.target_component,
                stream_id, rate, 1
            )

    # ----------------------------------------------------------------------- #
    # Background threads
    # ----------------------------------------------------------------------- #
    def _heartbeat_loop(self):
        """Send heartbeat at HEARTBEAT_RATE_HZ to keep ArduPilot in GUIDED."""
        interval = 1.0 / HEARTBEAT_RATE_HZ
        while self._running:
            self.conn.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
            time.sleep(interval)

    def _telemetry_loop(self):
        """Receive and parse incoming MAVLink messages."""
        while self._running:
            msg = self.conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            msg_type = msg.get_type()

            with self._lock:
                if msg_type == "LOCAL_POSITION_NED":
                    self._pos_ned = np.array([msg.x, msg.y, msg.z])
                    self._vel_ned = np.array([msg.vx, msg.vy, msg.vz])

                elif msg_type == "ATTITUDE_QUATERNION":
                    # ArduPilot sends w, x, y, z
                    self._quat_ned = np.array([msg.q1, msg.q2, msg.q3, msg.q4])
                    self._ang_vel_body = np.array([msg.rollspeed,
                                                   msg.pitchspeed,
                                                   msg.yawspeed])

                elif msg_type == "HEARTBEAT":
                    self._armed = bool(
                        msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                    )
                    try:
                        self._mode = mavutil.mode_string_v10(msg)
                    except Exception:
                        self._mode = str(msg.custom_mode)

    # ----------------------------------------------------------------------- #
    # State accessors (NED -> ENU conversion)
    # ----------------------------------------------------------------------- #
    def get_state_enu(self):
        """
        Return the current vehicle state in ENU coordinates, matching the
        Phase 2 gym-pybullet frame so the Koopman model works unchanged.

        Returns
        -------
        pos    : (3,) np.array  -- ENU position in metres
        vel    : (3,) np.array  -- ENU velocity in m/s
        rotmat : (9,) np.array  -- flattened 3x3 rotation matrix (ENU body->world)
        ang_vel: (3,) np.array  -- angular velocity in body frame, rad/s
        """
        with self._lock:
            pos_ned = self._pos_ned.copy()
            vel_ned = self._vel_ned.copy()
            q_ned   = self._quat_ned.copy()   # w, x, y, z
            av      = self._ang_vel_body.copy()

        # NED -> ENU position/velocity
        pos_enu = ned_to_enu(pos_ned)
        vel_enu = ned_to_enu(vel_ned)

        # Rotation matrix from NED quaternion -> convert to ENU rotation matrix
        R_ned = quat_to_rotmat_wxyz(q_ned)    # body->NED
        R_enu = ned_rotmat_to_enu(R_ned)       # body->ENU

        return pos_enu, vel_enu, R_enu.flatten(), av

    def get_state_raw_ned(self):
        """Raw NED state for logging / diagnostics."""
        with self._lock:
            return (self._pos_ned.copy(), self._vel_ned.copy(),
                    self._quat_ned.copy(), self._ang_vel_body.copy())

    @property
    def is_armed(self):
        with self._lock:
            return self._armed

    @property
    def mode(self):
        with self._lock:
            return self._mode

    # ----------------------------------------------------------------------- #
    # Vehicle commands
    # ----------------------------------------------------------------------- #
    def set_mode(self, mode_name):
        """Set flight mode by name, e.g. 'GUIDED_NOGPS', 'GUIDED', 'LOITER'."""
        mode_id = self.conn.mode_mapping().get(mode_name)
        if mode_id is None:
            raise ValueError(f"Unknown mode: {mode_name}. "
                             f"Available: {list(self.conn.mode_mapping().keys())}")
        self.conn.mav.set_mode_send(
            self.conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        print(f"[MAVLink] Mode set to {mode_name}")

    def arm(self, force=False, retries=3):
        """Send ARM command, with retries."""
        if self.is_armed:
            print("[MAVLink] Vehicle is already armed.")
            return True

        param2 = 21196 if force else 0

        for attempt in range(1, retries + 1):
            print(f"[MAVLink] ARM attempt {attempt}/{retries} "
                  f"(force={force}, target_sys={self.conn.target_system}) ...")
            self.conn.mav.command_long_send(
                self.conn.target_system,
                self.conn.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, param2, 0, 0, 0, 0, 0
            )

            t_end = time.time() + 4.0
            while time.time() < t_end:
                if self.is_armed:
                    print("[MAVLink] Armed successfully.")
                    return True
                ack = self.conn.recv_match(type="COMMAND_ACK", blocking=False)
                if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                    if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                        print("[MAVLink] ARM accepted (ACK).")
                        time.sleep(0.5)
                        return self.is_armed
                    else:
                        print(f"[MAVLink] ARM rejected by autopilot (result={ack.result}). "
                              f"Check ARMING_CHECK in Mission Planner.")
                        break
                time.sleep(0.1)

            time.sleep(0.5)

        print("[MAVLink] Arm command not accepted after all retries.")
        return False

    def disarm(self):
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def takeoff_guided(self, altitude_m=1.0):
        """Send TAKEOFF command (requires GPS fix in GUIDED; for GUIDED_NOGPS
        we skip this and go straight to attitude control)."""
        self.conn.mav.command_long_send(
            self.conn.target_system,
            self.conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m
        )

    def send_attitude_target(self, thrust, roll_rate=0.0, pitch_rate=0.0,
                              yaw_rate=0.0, roll=0.0, pitch=0.0, yaw=0.0,
                              use_rates=False):
        """
        Send SET_ATTITUDE_TARGET to ArduCopter.

        Parameters
        ----------
        thrust      : float  collective thrust [0.0 .. 1.0], 0.5 ≈ hover
        roll_rate   : float  rad/s (used if use_rates=True)
        pitch_rate  : float  rad/s
        yaw_rate    : float  rad/s
        roll, pitch : float  rad  -- desired attitude angles (used if use_rates=False)
        yaw         : float  rad  -- desired yaw
        use_rates   : bool   True: command body rates; False: command attitude angles
        """
        if use_rates:
            type_mask = 0b10111000  # ignore attitude, use rates + thrust
        else:
            type_mask = 0b00000111  # ignore rates, use attitude + thrust

        # Convert Euler to quaternion (targets from Koopman model in ENU)
        q = euler_to_quat_wxyz(roll, pitch, yaw)

        self.conn.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,  # time_boot_ms
            self.conn.target_system,
            self.conn.target_component,
            type_mask,
            [q[0], q[1], q[2], q[3]],  # q (w,x,y,z)
            roll_rate, pitch_rate, yaw_rate,
            thrust
        )

    def send_attitude_quaternion_thrust(self, q_wxyz, thrust, body_rates=None):
        """
        Direct quaternion + thrust command. q_wxyz should be in NED frame.
        body_rates (3,): optional roll/pitch/yaw rates in body frame.
        """
        if body_rates is None:
            body_rates = np.zeros(3)
            type_mask = 0b00000111  # ignore rates
        else:
            type_mask = 0b00000000  # use everything

        self.conn.mav.set_attitude_target_send(
            int(time.time() * 1000) & 0xFFFFFFFF,
            self.conn.target_system,
            self.conn.target_component,
            type_mask,
            [float(q_wxyz[0]), float(q_wxyz[1]),
             float(q_wxyz[2]), float(q_wxyz[3])],
            float(body_rates[0]),
            float(body_rates[1]),
            float(body_rates[2]),
            float(thrust)
        )

    def set_param(self, param_id, param_value, param_type=None):
        """Set an ArduPilot parameter by name."""
        if param_type is None:
            param_type = mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        self.conn.mav.param_set_send(
            self.conn.target_system,
            self.conn.target_component,
            param_id.encode("utf-8"),
            float(param_value),
            param_type
        )

    def get_param(self, param_id, timeout=2.0):
        """Request and return a parameter value."""
        self.conn.mav.param_request_read_send(
            self.conn.target_system,
            self.conn.target_component,
            param_id.encode("utf-8"),
            -1
        )
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.conn.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg and msg.param_id.rstrip("\x00") == param_id:
                return msg.param_value
        return None


# --------------------------------------------------------------------------- #
# Frame conversion utilities
# --------------------------------------------------------------------------- #

def ned_to_enu(v_ned):
    """Convert 3-vector from NED to ENU."""
    return np.array([v_ned[1], v_ned[0], -v_ned[2]])


def enu_to_ned(v_enu):
    """Convert 3-vector from ENU to NED."""
    return np.array([v_enu[1], v_enu[0], -v_enu[2]])


def quat_to_rotmat_wxyz(q):
    """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [2*(x*y + w*z),   1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def ned_rotmat_to_enu(R_ned):
    """
    Convert a rotation matrix from body->NED to body->ENU.
    ENU = T @ NED where T = [[0,1,0],[1,0,0],[0,0,-1]]
    R_enu = T @ R_ned
    """
    T = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, -1]], dtype=float)
    return T @ R_ned


def euler_to_quat_wxyz(roll, pitch, yaw):
    """Euler angles (rad) -> quaternion (w,x,y,z), aerospace convention."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def rotmat_to_euler(R):
    """Rotation matrix -> roll, pitch, yaw (rad)."""
    pitch = np.arcsin(-R[2, 0])
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def compute_desired_attitude_from_thrust_vector(thrust_vec_enu, yaw=0.0):
    """
    Given a desired thrust vector in ENU (x, y, z_up) and a desired yaw,
    compute the roll and pitch angles needed to point the drone's thrust
    axis in that direction.

    Used by the Koopman-MPC outer loop to translate (Fx, Fy, Fz) force
    commands into roll/pitch/thrust for SET_ATTITUDE_TARGET.

    Returns: (roll, pitch, thrust_normalized)
    """
    f = np.asarray(thrust_vec_enu, dtype=float)
    f_mag = np.linalg.norm(f)
    if f_mag < 1e-6:
        return 0.0, 0.0, 0.0

    # Desired z-body axis (pointing up in ENU)
    z_des = f / f_mag

    # Gravity-normalized thrust.
    HOVER_THRUST = 0.5
    g = 9.81
    thrust_norm = np.clip(HOVER_THRUST * (f_mag / g), 0.0, 1.0)

    # Roll/pitch from z-body direction (ENU convention)
    x_des, y_des, z_up = z_des
    pitch = -np.arcsin(np.clip(y_des, -1, 1))
    cp = np.cos(pitch)
    roll  = np.arcsin(np.clip(x_des / (cp + 1e-9), -1, 1))

    return roll, pitch, thrust_norm
