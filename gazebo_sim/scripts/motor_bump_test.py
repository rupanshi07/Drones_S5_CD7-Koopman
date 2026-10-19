import sys, time, math, threading
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators

MOTOR = int(sys.argv[1]) if len(sys.argv) > 1 else 0
KF, MASS, G = 8.54858e-06, 1.52, 9.8
OMEGA_MAX = 800.0
DT = 1.0 / 48.0
TARGET_Z, RAMP = 0.3, 0.048
KP, KD, KI, I_LIM = 15.0, 8.0, 6.0, 8.0
HOLD_T, BUMP_S, BUMP_W = 10.0, 0.3, 60.0

state = {}
lock = threading.Lock()

def euler(q):
    x, y, z, w = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw

def on_pose(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    for p in msg.pose:
        if p.name == "x3":
            q = p.orientation
            with lock:
                state["t"] = t
                state["z"] = p.position.z
                state["q"] = (q.x, q.y, q.z, q.w)

node = Node()
if not node.subscribe(Pose_V, "/world/quadcopter/pose/info", on_pose):
    sys.exit("subscribe failed")
pub = node.advertise("/X3/gazebo/command/motor_speed", Actuators)

def send(ws):
    m = Actuators()
    m.velocity.extend(ws)
    pub.publish(m)

wait_end = time.time() + 20
while "t" not in state and time.time() < wait_end:
    time.sleep(0.1)
if "t" not in state:
    sys.exit("no pose received - is Gazebo running?")

with lock:
    t0, z_prev = state["t"], state["z"]
t_prev, z_ref, integral, vz = t0, z_prev, 0.0, 0.0
bump_t0, ang0, ang1 = None, None, None
HOVER_W = math.sqrt(MASS * G / (4 * KF))

end = time.time() + 120
while time.time() < end:
    time.sleep(0.002)
    with lock:
        t, z, q = state["t"], state["z"], state["q"]
    if t - t_prev < DT:
        continue
    dt = t - t_prev
    ts = t - t0
    if ts < HOLD_T:
        vz = 0.5 * vz + 0.5 * ((z - z_prev) / dt)
        z_ref = min(z_ref + RAMP * dt, TARGET_Z)
        e = z_ref - z
        integral = max(-I_LIM, min(I_LIM, integral + KI * e * dt))
        force = max(MASS * G + KP * e - KD * vz + integral, 0.0)
        w = min(math.sqrt(force / (4 * KF)), OMEGA_MAX)
        send([w] * 4)
    else:
        if bump_t0 is None:
            bump_t0 = t
            ang0 = euler(q)
            print(f"hold done at z={z:.3f} m; bumping motor {MOTOR} by +{BUMP_W} rad/s", flush=True)
        ws = [HOVER_W] * 4
        ws[MOTOR] = HOVER_W + BUMP_W
        send(ws)
        if t - bump_t0 >= BUMP_S:
            ang1 = euler(q)
            break
    z_prev, t_prev = z, t

for _ in range(25):
    send([0.0] * 4)
    time.sleep(0.02)

if ang0 is None or ang1 is None:
    sys.exit("bump did not complete")
d = [math.degrees(b - a) for a, b in zip(ang0, ang1)]
print(f"motor {MOTOR}: d_roll={d[0]:+.3f} deg, d_pitch={d[1]:+.3f} deg, d_yaw={d[2]:+.4f} deg")
sy = "+" if d[0] > 0 else "-"
sx = "+" if d[1] < 0 else "-"
print(f"inferred rotor position: x {sx}, y {sy}   (model file: r0 (+,-), r1 (-,+), r2 (+,+), r3 (-,-))")
print(f"yaw sign for this motor: {'+' if d[2] > 0 else '-'}")
