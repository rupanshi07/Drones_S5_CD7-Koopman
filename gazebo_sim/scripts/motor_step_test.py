import sys, time, threading
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.actuators_pb2 import Actuators

OMEGA = float(sys.argv[1]) if len(sys.argv) > 1 else 700.0
RUN_SIM_SECONDS = 1.5
KF, MASS, G = 8.54858e-06, 1.52, 9.8

samples = []
lock = threading.Lock()

def on_pose(msg):
    t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
    for p in msg.pose:
        if p.name == "x3":
            with lock:
                samples.append((t, p.position.z))

node = Node()
if not node.subscribe(Pose_V, "/world/quadcopter/pose/info", on_pose):
    sys.exit("subscribe failed")
pub = node.advertise("/X3/gazebo/command/motor_speed", Actuators)

def send(w):
    m = Actuators()
    m.velocity.extend([w] * 4)
    pub.publish(m)

wait_end = time.time() + 20
while not samples and time.time() < wait_end:
    time.sleep(0.1)
if not samples:
    sys.exit("no pose received - is Gazebo running and not paused?")

with lock:
    t0, z0 = samples[-1]

loop_end = time.time() + 15
while time.time() < loop_end:
    send(OMEGA)
    time.sleep(0.02)
    with lock:
        t_now = samples[-1][0]
    if t_now - t0 >= RUN_SIM_SECONDS:
        break
for _ in range(25):
    send(0.0)
    time.sleep(0.02)

with lock:
    data = [s for s in samples if s[0] >= t0]
moving = [s for s in data if s[1] > z0 + 0.002]
if len(moving) < 5:
    sys.exit("drone did not lift off - check that OMEGA is above about 660")

ts, zs = moving[0]
num = den = 0.0
for t, z in moving:
    f = 0.5 * (t - ts) ** 2
    num += (z - zs) * f
    den += f * f
a_meas = num / den
a_pred = (4 * KF * OMEGA ** 2 - MASS * G) / MASS
print(f"omega = {OMEGA} rad/s, start z = {z0:.3f} m, final z = {data[-1][1]:.3f} m")
print(f"predicted acceleration = {a_pred:.3f} m/s^2")
print(f"measured acceleration  = {a_meas:.3f} m/s^2")
