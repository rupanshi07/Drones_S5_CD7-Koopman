import os, sys, time, math, threading, csv
import numpy as np
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.entity_pb2 import Entity

WIND_N = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
PAYLOAD_N = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
ONSET_T = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5

KF, MASS, G = 8.54858e-06, 1.52, 9.8
IXX, IYY, IZZ = 0.0347563, 0.07, 0.0977
C_YAW = 0.005
OMEGA_MAX = 800.0
DT = 1.0 / 48.0
TARGET_Z, RAMP = 0.3, 0.048
RUN_S = 15.0

KPZ, KDZ, KIZ, ILZ = 15.0, 8.0, 6.0, 8.0
KPXY, KDXY, KIXY, ILXY = 4.0, 3.6, 2.0, 3.0
TILT_MAX = math.radians(20.0)
WN_A, ZETA_A = 10.0, 0.9
WN_Y, ZETA_Y = 3.0, 1.0

XS = [0.13, -0.13, 0.13, -0.13]
YS = [-0.22, 0.20, 0.22, -0.20]
YAWSIGN = [-1, -1, 1, 1]
MAT = np.array([[KF] * 4,
                [KF * y for y in YS],
                [-KF * x for x in XS],
                [C_YAW * KF * s for s in YAWSIGN]])
MAT_INV = np.linalg.inv(MAT)

def euler(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])

state = {}
lock = threading.Lock()

def on_pose(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    for p in msg.pose:
        if p.name == "x3":
            q = p.orientation
            with lock:
                state["t"] = t
                state["pos"] = np.array([p.position.x, p.position.y, p.position.z])
                state["q"] = (q.x, q.y, q.z, q.w)

node = Node()
if not node.subscribe(Pose_V, "/world/quadcopter/pose/info", on_pose):
    sys.exit("subscribe failed")
pub = node.advertise("/X3/gazebo/command/motor_speed", Actuators)
wpub = node.advertise("/world/quadcopter/wrench/persistent", EntityWrench)

def send(ws):
    msg = Actuators()
    msg.velocity.extend([float(w) for w in ws])
    pub.publish(msg)

def push(fx, fy, fz):
    msg = EntityWrench()
    msg.entity.name = "x3"
    msg.entity.type = Entity.MODEL
    msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = fx, fy, fz
    return wpub.publish(msg)

wait_end = time.time() + 20
while "t" not in state and time.time() < wait_end:
    time.sleep(0.1)
if "t" not in state:
    sys.exit("no pose received - is Gazebo running?")

with lock:
    t0, pos_prev, q0 = state["t"], state["pos"].copy(), state["q"]
t_prev = t0
ang_prev = euler(q0)
z_ref = pos_prev[2]
vel = np.zeros(3)
rate = np.zeros(3)
int_x = int_y = int_z = 0.0
applied = False
log = []
next_print = 0.0
print(f"PID: wind {WIND_N} N (x), payload {PAYLOAD_N} N (down), onset {ONSET_T} s")

wall_end = time.time() + 120
while time.time() < wall_end:
    time.sleep(0.002)
    with lock:
        t, pos, q = state["t"], state["pos"].copy(), state["q"]
    if t - t_prev < DT:
        continue
    dt = t - t_prev
    ts = t - t0
    if not applied and ts >= ONSET_T and (WIND_N > 0 or PAYLOAD_N > 0):
        ok = push(WIND_N, 0.0, -PAYLOAD_N)
        applied = True
        print(f"*** disturbance applied at t={ts:.2f}s (publish ok: {ok})", flush=True)
    ang = euler(q)
    dang = ang - ang_prev
    dang[2] = (dang[2] + math.pi) % (2 * math.pi) - math.pi
    vel = 0.5 * vel + 0.5 * ((pos - pos_prev) / dt)
    rate = 0.5 * rate + 0.5 * (dang / dt)
    pos_prev, ang_prev, t_prev = pos, ang, t
    z_ref = min(z_ref + RAMP * dt, TARGET_Z)
    ex, ey, ez = -pos[0], -pos[1], z_ref - pos[2]
    int_x = max(-ILXY, min(ILXY, int_x + KIXY * ex * dt))
    int_y = max(-ILXY, min(ILXY, int_y + KIXY * ey * dt))
    int_z = max(-ILZ, min(ILZ, int_z + KIZ * ez * dt))
    ax = KPXY * ex - KDXY * vel[0] + int_x
    ay = KPXY * ey - KDXY * vel[1] + int_y
    pitch_des = max(-TILT_MAX, min(TILT_MAX, ax / G))
    roll_des = max(-TILT_MAX, min(TILT_MAX, -ay / G))
    tau_roll = IXX * (WN_A**2 * (roll_des - ang[0]) - 2 * ZETA_A * WN_A * rate[0])
    tau_pitch = IYY * (WN_A**2 * (pitch_des - ang[1]) - 2 * ZETA_A * WN_A * rate[1])
    tau_yaw = IZZ * (WN_Y**2 * (0.0 - ang[2]) - 2 * ZETA_Y * WN_Y * rate[2])
    fz = MASS * G + KPZ * ez - KDZ * vel[2] + int_z
    F = max(fz, 0.0) / max(math.cos(ang[0]) * math.cos(ang[1]), 0.7)
    cmd = np.array([F, tau_roll, tau_pitch, tau_yaw])
    s = np.clip(MAT_INV @ cmd, 0.0, OMEGA_MAX**2)
    w = np.sqrt(s)
    send(w)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(ang[0]) * math.cos(ang[1])))))
    log.append((ts, *pos, *vel, *ang, *w, 1 if applied else 0))
    if ts >= next_print:
        print(f"t={ts:4.1f}s x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:.3f} tilt={tilt:5.2f}deg omega={w.mean():.1f}", flush=True)
        next_print += 1.0
    if tilt > 60:
        print("UNSTABLE: tilt over 60 deg, stopping", flush=True)
        break
    if ts >= RUN_S:
        break

for _ in range(25):
    send([0.0] * 4)
    time.sleep(0.02)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(out_dir, exist_ok=True)
fname = os.path.join(out_dir, f"pid_wind{WIND_N:g}N_payload{PAYLOAD_N:g}N_log.csv")
with open(fname, "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "w0", "w1", "w2", "w3", "disturbance_on"])
    wr.writerows(log)

post = [r for r in log if r[0] >= ONSET_T]
if post:
    print(f"after onset: max |x| = {max(abs(r[1]) for r in post):.3f} m, max |y| = {max(abs(r[2]) for r in post):.3f} m, "
          f"min z = {min(r[3] for r in post):.3f} m, final x = {log[-1][1]:+.3f}, final z = {log[-1][3]:.3f}")
print("saved", fname)
