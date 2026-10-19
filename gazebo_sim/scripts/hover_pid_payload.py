import os, sys, time, math, threading, csv
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.entity_pb2 import Entity

PAYLOAD_N = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
KF, MASS, G = 8.54858e-06, 1.52, 9.8
OMEGA_MAX = 800.0
CTRL_HZ = 48.0
DT = 1.0 / CTRL_HZ
TARGET_Z = 0.3
RAMP_RATE = 0.001 * CTRL_HZ
RUN_SIM_SECONDS = 15.0
ONSET_T = 7.5
KP, KD, KI = 15.0, 8.0, 6.0
I_LIMIT = 8.0

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
wpub = node.advertise("/world/quadcopter/wrench/persistent", EntityWrench)

def send(w):
    m = Actuators()
    m.velocity.extend([w] * 4)
    pub.publish(m)

def push(fx, fy, fz):
    m = EntityWrench()
    m.entity.name = "x3"
    m.entity.type = Entity.MODEL
    m.wrench.force.x, m.wrench.force.y, m.wrench.force.z = fx, fy, fz
    return wpub.publish(m)

wait_end = time.time() + 20
while "t" not in state and time.time() < wait_end:
    time.sleep(0.1)
if "t" not in state:
    sys.exit("no pose received - is Gazebo running?")

expected = math.sqrt((MASS * G + PAYLOAD_N) / (4 * KF))
print(f"payload {PAYLOAD_N} N at t={ONSET_T}s; expected final omega about {expected:.1f} rad/s (hover is 660)")

with lock:
    t0, z_prev = state["t"], state["z"]
t_prev = t0
z_ref = z_prev
integral = 0.0
vz = 0.0
applied = False
log = []
next_print = 0.0

wall_end = time.time() + 90
while time.time() < wall_end:
    time.sleep(0.002)
    with lock:
        t, z, qx, qy = state["t"], state["z"], state["qx"], state["qy"]
    if t - t_prev < DT:
        continue
    dt = t - t_prev
    ts = t - t0
    if not applied and ts >= ONSET_T and PAYLOAD_N > 0:
        ok = push(0.0, 0.0, -PAYLOAD_N)
        applied = True
        print(f"*** payload force applied at t={ts:.2f}s (publish ok: {ok})", flush=True)
    vz = 0.5 * vz + 0.5 * ((z - z_prev) / dt)
    z_prev, t_prev = z, t
    z_ref = min(z_ref + RAMP_RATE * dt, TARGET_Z)
    e = z_ref - z
    integral = max(-I_LIMIT, min(I_LIMIT, integral + KI * e * dt))
    force = max(MASS * G + KP * e - KD * vz + integral, 0.0)
    omega = min(math.sqrt(force / (4 * KF)), OMEGA_MAX)
    send(omega)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy)))))
    log.append((ts, z_ref, z, vz, omega, tilt, 1 if applied else 0))
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
fname = os.path.join(out_dir, f"hover_pid_payload_{PAYLOAD_N:g}N_log.csv")
with open(fname, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t", "z_ref", "z", "vz", "omega", "tilt_deg", "force_applied"])
    w.writerows(log)

post = [r for r in log if r[0] >= ONSET_T]
if post:
    dip = TARGET_Z - min(r[2] for r in post)
    out = [r for r in post if abs(r[2] - TARGET_Z) > 0.02]
    settle = "not settled" if (out and out[-1][0] >= log[-1][0] - 0.05) else f"{(out[-1][0] - ONSET_T) if out else 0.0:.2f} s"
    tail = [r[4] for r in log if r[0] >= RUN_SIM_SECONDS - 2]
    print(f"max dip after onset = {dip:.3f} m, settling (within 2 cm) = {settle}, "
          f"mean omega last 2 s = {sum(tail)/len(tail):.1f}, max tilt = {max(r[5] for r in log):.2f} deg")
print("saved", fname)
