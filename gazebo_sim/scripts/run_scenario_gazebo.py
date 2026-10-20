import os, sys, time, math, threading
import numpy as np
from scipy.linalg import solve_discrete_are, expm
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.entity_pb2 import Entity

CTRL_NAME, SC = sys.argv[1], sys.argv[2]
WIND_N, PAYLOAD_N, MID = 2.8, 2.0, 7.5
SCEN = {"1": (None, None), "2": (0.0, None), "3": (0.0, None),
        "4": (MID, None), "5": (None, MID), "6": (0.0, MID)}
WIND_ON, PAY_ON = SCEN[SC]

KF, MASS, G = 8.54858e-06, 1.52, 9.8
IXX, IYY, IZZ = 0.0347563, 0.07, 0.0977
C_YAW, OMEGA_MAX = 0.005, 800.0
DT = 1.0 / 48.0
TARGET_Z, RAMP, RUN_S = 0.3, 0.048, 15.0

XS = [0.13, -0.13, 0.13, -0.13]
YS = [-0.22, 0.20, 0.22, -0.20]
YAWSIGN = [-1, -1, 1, 1]
MAT = np.array([[KF] * 4, [KF * y for y in YS], [-KF * x for x in XS],
                [C_YAW * KF * s for s in YAWSIGN]])
MAT_INV = np.linalg.inv(MAT)

def make_lqr():
    n, m = 12, 4
    A = np.zeros((n, n)); A[0, 3] = A[1, 4] = A[2, 5] = 1.0
    A[3, 7] = G; A[4, 6] = -G; A[6, 9] = A[7, 10] = A[8, 11] = 1.0
    B = np.zeros((n, m)); B[5, 0] = 1 / MASS; B[9, 1] = 1 / IXX
    B[10, 2] = 1 / IYY; B[11, 3] = 1 / IZZ
    Mx = np.zeros((n + m, n + m)); Mx[:n, :n] = A * DT; Mx[:n, n:] = B * DT
    E = expm(Mx); Ad, Bd = E[:n, :n], E[:n, n:]
    Q = np.diag([1 / 0.1**2] * 3 + [1 / 0.5**2] * 3 + [1 / 0.2**2] * 2 + [1 / 0.5**2] + [1 / 1.0**2] * 3)
    R = np.diag([1 / 4.0**2, 1 / 0.2**2, 1 / 0.2**2, 1 / 0.02**2])
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    def ctrl(pos, vel, ang, rate, z_ref, dt):
        xt = np.zeros(12); xt[2] = z_ref
        u = -K @ (np.concatenate([pos, vel, ang, rate]) - xt)
        return np.array([max(MASS * G + u[0], 0.0), u[1], u[2], u[3]])
    return ctrl

def make_pid():
    KPZ, KDZ, KIZ, ILZ = 15.0, 8.0, 6.0, 8.0
    KPXY, KDXY, KIXY, ILXY = 4.0, 3.6, 2.0, 3.0
    TMAX, WNA, ZA, WNY, ZY = math.radians(20.0), 10.0, 0.9, 3.0, 1.0
    st = {"ix": 0.0, "iy": 0.0, "iz": 0.0}
    def ctrl(pos, vel, ang, rate, z_ref, dt):
        ex, ey, ez = -pos[0], -pos[1], z_ref - pos[2]
        st["ix"] = max(-ILXY, min(ILXY, st["ix"] + KIXY * ex * dt))
        st["iy"] = max(-ILXY, min(ILXY, st["iy"] + KIXY * ey * dt))
        st["iz"] = max(-ILZ, min(ILZ, st["iz"] + KIZ * ez * dt))
        ax = KPXY * ex - KDXY * vel[0] + st["ix"]
        ay = KPXY * ey - KDXY * vel[1] + st["iy"]
        pd = max(-TMAX, min(TMAX, ax / G))
        rd = max(-TMAX, min(TMAX, -ay / G))
        tr = IXX * (WNA**2 * (rd - ang[0]) - 2 * ZA * WNA * rate[0])
        tp = IYY * (WNA**2 * (pd - ang[1]) - 2 * ZA * WNA * rate[1])
        ty = IZZ * (WNY**2 * (0.0 - ang[2]) - 2 * ZY * WNY * rate[2])
        fz = MASS * G + KPZ * ez - KDZ * vel[2] + st["iz"]
        F = max(fz, 0.0) / max(math.cos(ang[0]) * math.cos(ang[1]), 0.7)
        return np.array([F, tr, tp, ty])
    return ctrl

ctrl = make_lqr() if CTRL_NAME == "lqr" else make_pid()

def euler(q):
    x, y, z, w = q
    return np.array([math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                     math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
                     math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])

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
    msg = Actuators(); msg.velocity.extend([float(w) for w in ws]); pub.publish(msg)

def push(fx, fy, fz):
    msg = EntityWrench(); msg.entity.name = "x3"; msg.entity.type = Entity.MODEL
    msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = fx, fy, fz
    return wpub.publish(msg)

end_wait = time.time() + 30
while "t" not in state and time.time() < end_wait:
    time.sleep(0.1)
if "t" not in state:
    sys.exit("no pose received - is Gazebo running?")
time.sleep(1.0)

with lock:
    t0, pos_prev, q0 = state["t"], state["pos"].copy(), state["q"]
t_prev, ang_prev, z_ref = t0, euler(q0), pos_prev[2]
vel, rate = np.zeros(3), np.zeros(3)
w_act = p_act = False
log_s, log_a, log_t, log_f = [], [], [], []

end = time.time() + 150
while time.time() < end:
    time.sleep(0.002)
    with lock:
        t, pos, q = state["t"], state["pos"].copy(), state["q"]
    if t - t_prev < DT:
        continue
    dt, ts = t - t_prev, t - t0
    nw = WIND_ON is not None and ts >= WIND_ON
    npay = PAY_ON is not None and ts >= PAY_ON
    if nw and not w_act:
        w_act = True
        ok = push(WIND_N, 0.0, 0.0)
        print(f"*** t={ts:.2f}s wind applied (publish ok: {ok})", flush=True)
    if npay and not p_act:
        p_act = True
        ok = push(0.0, 0.0, -PAYLOAD_N)
        print(f"*** t={ts:.2f}s payload applied (publish ok: {ok})", flush=True)
    ang = euler(q)
    da = ang - ang_prev
    da[2] = (da[2] + math.pi) % (2 * math.pi) - math.pi
    vel = 0.5 * vel + 0.5 * ((pos - pos_prev) / dt)
    rate = 0.5 * rate + 0.5 * (da / dt)
    pos_prev, ang_prev, t_prev = pos, ang, t
    z_ref = min(z_ref + RAMP * dt, TARGET_Z)
    cmd = ctrl(pos, vel, ang, rate, z_ref, dt)
    w = np.sqrt(np.clip(MAT_INV @ cmd, 0.0, OMEGA_MAX**2))
    send(w)
    log_s.append(np.concatenate([pos, vel, ang, rate])); log_a.append(w)
    log_t.append(ts); log_f.append(1 if (w_act or p_act) else 0)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(ang[0]) * math.cos(ang[1])))))
    if tilt > 60:
        print("UNSTABLE: tilt over 60 deg", flush=True)
        break
    if ts >= RUN_S:
        break

for _ in range(25):
    send([0.0] * 4)
    time.sleep(0.02)

S, Aa, T, Fl = np.array(log_s), np.array(log_a), np.array(log_t), np.array(log_f)
onset = int(np.argmax(Fl)) if Fl.any() else -1
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(out, exist_ok=True)
fname = os.path.join(out, f"gazebo_{CTRL_NAME}_scenario_{SC}.npz")
np.savez(fname, states=S, actions=Aa, t=T, disturbance_flag=Fl,
         disturbance_onset_step=onset, target_z=TARGET_Z,
         state_columns="x y z vx vy vz roll pitch yaw p q r", action_units="rad/s")
tail = T >= T[-1] - 2.0
fv = MASS * G + (PAYLOAD_N if PAY_ON is not None else 0.0)
fh = WIND_N if WIND_ON is not None else 0.0
exp_w = math.sqrt(math.sqrt(fv**2 + fh**2) / (4 * KF))
print(f"{CTRL_NAME} scenario {SC}: steps={len(T)}, onset step={onset}, "
      f"final x={S[-1,0]:+.3f} y={S[-1,1]:+.3f} z={S[-1,2]:.3f}, max|x|={np.abs(S[:,0]).max():.3f}, min z after 1 s={S[T>1,2].min():.3f}")
print(f"motors last 2 s = {Aa[tail].mean():.1f} rad/s, physics for this scenario = {exp_w:.1f} "
      f"(a big gap means the wrench did not behave as expected)")
print("saved", fname)
