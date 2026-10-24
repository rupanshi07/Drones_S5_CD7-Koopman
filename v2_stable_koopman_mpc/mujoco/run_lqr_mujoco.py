# run_lqr_mujoco.py
"""Thin wrapper to launch the LQR controller simulation via run_sim.py.

Usage mirrors the other controller wrappers:
    python run_lqr_mujoco.py --terrain <terrain> --payload <kg> --wind <m/s>

All arguments are forwarded to ``run_sim.py`` with ``--ctrl lqr``.
"""
import sys
import subprocess

def main():
    extra = sys.argv[1:]
    cmd = [sys.executable, "run_sim.py", "--ctrl", "lqr"] + extra
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
