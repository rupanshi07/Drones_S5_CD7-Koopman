# run_pid_mujoco.py
"""Thin wrapper to launch the PID controller simulation via run_sim.py.

Usage mirrors the other controller wrappers:
    python run_pid_mujoco.py --terrain <terrain> --payload <kg> --wind <m/s>

All arguments are forwarded to ``run_sim.py`` with ``--ctrl pid``.
"""
import sys
import subprocess
import shlex

def main():
    # Build the command line for run_sim.py
    # Preserve any extra args the user passed.
    extra = sys.argv[1:]
    # Ensure the controller flag is present
    cmd = [sys.executable, "run_sim.py", "--ctrl", "pid"] + extra
    # Run the simulation – inherit stdout/stderr so the user sees output.
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
