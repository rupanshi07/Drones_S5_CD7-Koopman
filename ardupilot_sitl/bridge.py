"""
bridge.py - MAVLink bridge for ArduCopter SITL on Windows
==========================================================
Run this FIRST, then start SITL in Mission Planner.
"""

import socket
import threading
import sys
import time

SITL_HOST   = "127.0.0.1"
SITL_PORT   = 5760   # ArduCopter SITL server (we connect here)
MP_PORT     = 14552  # Bridge server for Mission Planner to reconnect
SCRIPT_PORT = 14553  # Bridge server for our Python scripts
BIND_ADDR   = "127.0.0.1"  # Loopback only (avoids college network firewall)


def make_server(port, label):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((BIND_ADDR, port))
    except PermissionError as e:
        print(f"\n[Bridge] ERROR: Cannot bind port {port}: {e}")
        print(f"  Try running terminal as Administrator, or check antivirus/firewall.")
        sys.exit(1)
    s.listen(5)
    s.settimeout(0.05)
    print(f"[Bridge] {label} server: TCP {BIND_ADDR}:{port}")
    return s


def connect_to_sitl():
    print(f"\n[Bridge] Waiting for ArduCopter SITL on {SITL_HOST}:{SITL_PORT} ...")
    print(f"  -> Start SITL: Mission Planner -> Simulation -> Multirotor -> Run SITL")
    print(f"  -> Then IMMEDIATELY click DISCONNECT in Mission Planner\n")
    attempt = 0
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((SITL_HOST, SITL_PORT))
            s.settimeout(0.02)
            print(f"\n[Bridge] Connected to SITL!")
            print(f"[Bridge] >>> NOW click DISCONNECT in Mission Planner! <<<\n")
            return s
        except Exception:
            attempt += 1
            sys.stdout.write(".")
            sys.stdout.flush()
            time.sleep(1)


def run_bridge():
    sitl = connect_to_sitl()

    print("[Bridge] Waiting 4s for you to click Disconnect in Mission Planner...")
    time.sleep(4)

    print()
    mp_server     = make_server(MP_PORT,     "Mission Planner")
    script_server = make_server(SCRIPT_PORT, "Python scripts ")

    print(f"""
[Bridge] ===== READY =====
  Step 1: In Mission Planner:
            Top-right dropdown -> TCP
            Host: 127.0.0.1
            Port: {MP_PORT}
            Click Connect

  Step 2: In another terminal:
            python setup_vehicle.py --port {SCRIPT_PORT}

  Step 3: In another terminal:
            python sitl_controller.py --controller koopman --port {SCRIPT_PORT}

[Bridge] Forwarding MAVLink... press Ctrl+C to stop.
""")

    clients = []
    lock = threading.Lock()

    def accept_loop(server, label):
        while True:
            try:
                conn, addr = server.accept()
                conn.settimeout(0.02)
                with lock:
                    clients.append(conn)
                print(f"[Bridge] {label} connected from {addr[0]}:{addr[1]}")
            except socket.timeout:
                pass
            except OSError:
                break

    threading.Thread(
        target=accept_loop, args=(mp_server, "MissionPlanner"), daemon=True
    ).start()
    threading.Thread(
        target=accept_loop, args=(script_server, "Script       "), daemon=True
    ).start()

    while True:
        # SITL → all clients
        try:
            data = sitl.recv(8192)
            if not data:
                print("[Bridge] SITL disconnected.")
                break
            with lock:
                dead = []
                for c in clients:
                    try:
                        c.sendall(data)
                    except Exception:
                        dead.append(c)
                for d in dead:
                    clients.remove(d)
                    print("[Bridge] Client dropped.")
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[Bridge] SITL read error: {e}")
            break

        # Clients → SITL
        with lock:
            for c in clients[:]:
                try:
                    data = c.recv(4096)
                    if data:
                        sitl.sendall(data)
                except socket.timeout:
                    pass
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        run_bridge()
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped.")
