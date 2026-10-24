'''run_all_with_ffmpeg.py

Automates video capture of MuJoCo drone simulations for each configuration listed in a CSV.

The script:
1. Reads a CSV file containing at least the columns:
   - `controller` (e.g., koopman, pid, lqr)
   - `terrain`   (e.g., flat, hilly)
   - `payload`   (numeric, kg)
   - `wind`      (numeric, m/s)
   - `extra_args` (optional, additional CLI args for the simulation)
2. For each row it builds a descriptive output filename:
   ``{controller}_{terrain}_payload{payload}_wind{wind}.mp4``
   (spaces and special characters are sanitized).
3. Starts an ``ffmpeg`` process that records the MuJoCo window using the
   Windows GDI grabber (``gdigrab``). The window title is assumed to be
   "MuJoCo Simulation" – you can change it via the ``MUJOCO_WINDOW_TITLE``
   environment variable before launching the simulation.
4. Launches the appropriate simulation script (``run_koopman_mujoco.py`` for
   Koopman‑MPC, ``run_pid_mujoco.py`` for PID, ``run_lqr_mujoco.py`` for LQR).
   The script sets ``MUJOCO_GL=glfw`` so that a GUI window is displayed.
5. Waits for the simulation process to finish, then terminates ``ffmpeg``.
6. Repeats for every CSV row, storing the resulting MP4 files in the ``demos``
   sub‑directory.

Prerequisites
--------------
- ``ffmpeg`` must be on the system PATH. Verify with ``ffmpeg -version``.
- MuJoCo simulation scripts must be runnable from this directory.
- The CSV file should be placed alongside this script (or provide its path
  via the ``--csv`` argument).

Usage
-----
```powershell
python run_all_with_ffmpeg.py --csv results.csv
```

If the MuJoCo window title differs, set the environment variable before the
run, e.g.:
```powershell
$env:MUJOCO_WINDOW_TITLE="My Drone Sim"
python run_all_with_ffmpeg.py --csv results.csv
```

The script creates the ``demos`` folder if it does not exist.
''' 

import argparse
import os
import subprocess
import sys
import shlex
import pandas as pd
import time
import signal

def sanitize(s: str) -> str:
    """Replace spaces and problematic characters for filenames."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)

def start_ffmpeg(output_path: str, window_title: str) -> subprocess.Popen:
    """Start ffmpeg capturing the MuJoCo window.

    Returns the Popen handle. The caller should terminate it after the
    simulation finishes.
    """
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # overwrite existing files
        "-f", "gdigrab",
        "-framerate", "30",
        "-i", f"title={window_title}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    # Hide ffmpeg console window on Windows
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags)

def run_simulation(controller: str, terrain: str, payload: str, wind: str, extra_args: str):
    """Launch the appropriate MuJoCo simulation script.

    The function sets ``MUJOCO_GL=glfw`` to force a GUI window.
    """
    env = os.environ.copy()
    env["MUJOCO_GL"] = "glfw"
    # Optionally set a custom window title
    if "MUJOCO_WINDOW_TITLE" in env:
        window_title = env["MUJOCO_WINDOW_TITLE"]
    else:
        window_title = "MuJoCo Simulation"
    # Build the command based on the controller name
    if controller.lower() == "koopman":
        script = "run_koopman_mujoco.py"
    elif controller.lower() == "pid":
        script = "run_pid_mujoco.py"
    elif controller.lower() == "lqr":
        script = "run_lqr_mujoco.py"
    else:
        raise ValueError(f"Unsupported controller: {controller}")

    cmd = [sys.executable, script, "--terrain", terrain, "--payload", payload, "--wind", wind]
    if extra_args:
        # split extra args respecting quotes
        cmd.extend(shlex.split(extra_args))
    # Run the simulation and block until it exits
    subprocess.run(cmd, env=env, check=True)

    def main(csv_path: str, demos_dir: str, window_title: str) -> None:
    """Run simulations for each row in the CSV and record videos.

    Parameters
    ----------
    csv_path: str
        Path to the CSV file with columns ``controller``, ``terrain``, ``payload``
        and either ``wind`` or ``wind_n``.
    demos_dir: str
        Directory where the generated MP4 files will be stored.
    window_title: str
        Title of the MuJoCo window captured by ffmpeg.
    """
    # Resolve absolute paths for safety
    csv_path = os.path.abspath(csv_path)
    demos_dir = os.path.abspath(demos_dir)

    # Load CSV
    df = pd.read_csv(csv_path)

    # Determine which wind column is present (case‑insensitive)
    wind_col = None
    for candidate in ["wind", "wind_n"]:
        if candidate in df.columns.str.lower():
            wind_col = candidate
            break
    if wind_col is None:
        print("CSV is missing a wind column (expected 'wind' or 'wind_n').")
        sys.exit(1)

    # Required columns (lowercase for comparison)
    required = {"controller", "terrain", "payload"}
    missing = required - set(df.columns.str.lower())
    if missing:
        print(f"CSV is missing required columns: {missing}")
        sys.exit(1)

    # Normalise column names to lower case for easy access
    df.columns = [c.lower() for c in df.columns]
    os.makedirs(demos_dir, exist_ok=True)

    for idx, row in df.iterrows():
        controller = str(row["controller"]).strip()
        terrain = str(row["terrain"]).strip()
        payload = str(row["payload"]).strip()
        wind = str(row[wind_col]).strip()
        extra_args = str(row.get("extra_args", "")).strip()
        filename = f"{sanitize(controller)}_{sanitize(terrain)}_payload{sanitize(payload)}_wind{sanitize(wind)}.mp4"
        output_path = os.path.join(demos_dir, filename)
        abs_path = os.path.abspath(output_path)
        print(f"[DEBUG] Full output path: {abs_path}")
        print(f"[+] Recording {output_path} ...")
        ffmpeg_proc = start_ffmpeg(output_path, window_title)
        try:
            run_simulation(controller, terrain, payload, wind, extra_args)
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed for row {idx}: {e}")
        finally:
            # Give ffmpeg a moment to flush buffers
            time.sleep(0.5)
            # Terminate ffmpeg gracefully
            if ffmpeg_proc.poll() is None:
                ffmpeg_proc.send_signal(signal.CTRL_C_EVENT if os.name == "nt" else signal.SIGINT)
                try:
                    ffmpeg_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
        print(f"[+] Saved video: {output_path}\n")

    demos_dir = os.path.abspath(demos_dir)
    df = pd.read_csv(csv_path)
    required = {"controller", "terrain", "payload", "wind"}
    required = {"controller", "terrain", "payload"}
    # Accept either "wind" or "wind_n"
    wind_col = None
    for candidate in ["wind", "wind_n"]:
        if candidate in df.columns.str.lower():
            wind_col = candidate
            break
    if wind_col is None:
        print("CSV is missing a wind column (expected 'wind' or 'wind_n').")
        sys.exit(1)
    # Ensure all required columns are present
    missing = required - set(df.columns.str.lower())
    if missing:
        print(f"CSV is missing required columns: {missing}")
        sys.exit(1)
        # The following lines were duplicated after the exit above and are now removed.

    # Normalise column names to lower case for easy access
    df.columns = [c.lower() for c in df.columns]
    os.makedirs(demos_dir, exist_ok=True)
    for idx, row in df.iterrows():
        controller = str(row["controller"]).strip()
        terrain = str(row["terrain"]).strip()
        payload = str(row["payload"]).strip()
        wind = str(row[wind_col]).strip()
        extra_args = str(row.get("extra_args", "")).strip()
        filename = f"{sanitize(controller)}_{sanitize(terrain)}_payload{sanitize(payload)}_wind{sanitize(wind)}.mp4"
        output_path = os.path.join(demos_dir, filename)
        abs_path = os.path.abspath(output_path)
        print(f"[DEBUG] Full output path: {abs_path}")
        print(f"[+] Recording {output_path} ...")
        ffmpeg_proc = start_ffmpeg(output_path, window_title)
        try:
            run_simulation(controller, terrain, payload, wind, extra_args)
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed for row {idx}: {e}")
        finally:
            # Give ffmpeg a moment to flush buffers
            time.sleep(0.5)
            # Terminate ffmpeg gracefully
            if ffmpeg_proc.poll() is None:
                ffmpeg_proc.send_signal(signal.CTRL_C_EVENT if os.name == "nt" else signal.SIGINT)
                # Wait a short while, then force kill if needed
                try:
                    ffmpeg_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    ffmpeg_proc.kill()
        print(f"[+] Saved video: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MuJoCo simulation videos based on a CSV of parameter sets.")
    parser.add_argument("--csv", type=str, default="results.csv", help="Path to the CSV file containing simulation parameters.")
    parser.add_argument("--out", type=str, default="demos", help="Directory where MP4 files will be stored.")
    parser.add_argument("--title", type=str, default=os.getenv("MUJOCO_WINDOW_TITLE", "MuJoCo Simulation"), help="Window title of the MuJoCo GUI (used for ffmpeg capture).")
    args = parser.parse_args()
    main(args.csv, args.out, args.title)
