import sys, time, threading
from gz.transport13 import Node
from gz.msgs10.pose_v_pb2 import Pose_V

latest = {}
lock = threading.Lock()

def on_pose(msg):
    for p in msg.pose:
        if p.name == "x3":
            with lock:
                latest["t"] = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
                latest["pos"] = (p.position.x, p.position.y, p.position.z)
                latest["quat"] = (p.orientation.x, p.orientation.y,
                                  p.orientation.z, p.orientation.w)

node = Node()
if not node.subscribe(Pose_V, "/world/quadcopter/pose/info", on_pose):
    sys.exit("subscribe failed")

end = time.time() + 5
while time.time() < end:
    time.sleep(0.5)
    with lock:
        if latest:
            print(latest)
