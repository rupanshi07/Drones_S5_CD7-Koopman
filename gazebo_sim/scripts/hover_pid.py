import os, sys, time, math, threading, csv
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators

KF, MASS, G = 8.54858e-06, 1.52, 9.8
OMEGA_MAX = 800.0
CTRL_HZ = 48.0
DT = 1.0 / CTRL_HZ
TARGET_Z = 0.3
RAMP_RATE = 0.001 * CTRL_HZ      # m/s, same ramp as the PyBullet runs
RUN_SIM_SECONDS = 12.0
KP, KD, KI = 6.0, 5.0, 1.5
I_LIMIT = 2.0                    # N, anti-windup clamp

state = {}
lock = threading.Lock()

def on_pose(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    for p in msg.pose:
        if p.name == "x3":
            q = p.orientation
            with lock:
                state["t"] = t
                state["z"] = p.position.z
                state["qx"], state["qy"] = q.x, q.y

node = Node()
if not node.subscribe(Pose_V, "/world/quadcopter/pose/info", on_pose):
    sys.exit("subscribe failed")
pub = node.advertise("/X3/gazebo/command/motor_speed", Actuators)

def send(w):
    m = Actuators()
    m.velocity.extend([w] * 4)
    pub.publish(m)

wait_end = time.time() + 20
while "t" not in state and time.time() < wait_end:
    time.sleep(0.1)
if "t" not in state:
    sys.exit("no pose received - is Gazebo running?")

with lock:
    t0, z_prev = state["t"], state["z"]
t_prev = t0
z_ref = z_prev
integral = 0.0
vz = 0.0
log = []
next_print = 0.0

wall_end = time.time() + 60
while time.time() < wall_end:
    time.sleep(0.002)
    with lock:
        t, z, qx, qy = state["t"], state["z"], state["qx"], state["qy"]
    if t - t_prev < DT:
        continue
    dt = t - t_prev
    vz = 0.5 * vz + 0.5 * ((z - z_prev) / dt)
    z_prev, t_prev = z, t
    z_ref = min(z_ref + RAMP_RATE * dt, TARGET_Z)
    e = z_ref - z
    integral = max(-I_LIMIT, min(I_LIMIT, integral + KI * e * dt))
    force = max(MASS * G + KP * e - KD * vz + integral, 0.0)
    omega = min(math.sqrt(force / (4 * KF)), OMEGA_MAX)
    send(omega)
    ts = t - t0
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
    log.append((ts, z_ref, z, vz, omega, tilt))
    if ts >= next_print:
        print(f"t={ts:4.1f}s ref={z_ref:.3f} z={z:.3f} vz={vz:+.3f} omega={omega:.1f} tilt={tilt:.2f}deg", flush=True)
        next_print += 1.0
    if ts >= RUN_SIM_SECONDS:
        break

for _ in range(25):
    send(0.0)
    time.sleep(0.02)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "hover_pid_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "z_ref", "z", "vz", "omega", "tilt_deg"])
    w.writerows(log)

tail = [r for r in log if r[0] >= RUN_SIM_SECONDS - 5]
if tail:
    rmse = math.sqrt(sum((r[1] - r[2]) ** 2 for r in tail) / len(tail))
    print(f"RMSE over last 5 s = {rmse:.4f} m, max tilt = {max(r[5] for r in log):.2f} deg")
