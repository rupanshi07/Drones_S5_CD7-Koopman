# ------------------------------------------------------------
# run_all_terrain.py
# ------------------------------------------------------------
# Executes Koopman‑MPC, PID and LQR on a set of terrain presets,
# with optional payload / wind, and writes a CSV summary.
# ------------------------------------------------------------
import subprocess
import csv
from pathlib import Path

# === USER SETTINGS ==================================================
TERRAINS   = ["rolling", "mountain", "canyon", "flat"]    # add/remove as you like
PAYLOAD    = 0.20    # +20 % mass injected at t≈3 s (set 0.0 for none)
WIND_FORCE = 0.015   # N cross‑wind (set 0.0 for none)
WIND_DIR   = 90.0    # degrees, 0° = +X, 90° = +Y
SEED       = 0       # fix terrain randomness for reproducibility
CONTROLLERS = {
    "koopman": "python run_koopman_mujoco.py",
    "pid":     "python run_sim.py --ctrl pid",
    "lqr":     "python run_sim.py --ctrl lqr",
}
# --------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / "demos"
DEMO_DIR.mkdir(exist_ok=True)

# CSV header
CSV_PATH = DEMO_DIR / "summary.csv"
csv_file = open(CSV_PATH, "w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["controller", "terrain", "payload", "wind_N", "wind_dir_deg",
                   "RMSE_m", "max_err_m", "min_AGL_m", "log_file"])

def run_one(controller: str, terrain: str):
    """Run a single episode and capture the log."""
    # Build the command line
    cmd = CONTROLLERS[controller].split()
    cmd += ["--gui", f"--terrain {terrain}",
            f"--seed {SEED}",
            f"--payload {PAYLOAD}",
            f"--wind {WIND_FORCE}",
            f"--wind-dir {WIND_DIR}"]

    # Launch the process (blocking)
    proc = subprocess.run(" ".join(cmd), cwd=BASE_DIR, shell=True,
                          capture_output=True, text=True)

    # Parse the stdout for the metrics (the original script prints them)
    # Example line:
    #   RMSE: 0.189 m   Max Err: 0.391 m   Min AGL: 1.430 m
    rmse = max_err = min_agl = None
    for line in proc.stdout.splitlines():
        if "RMSE:" in line and "Max Err:" in line and "Min AGL:" in line:
            parts = line.replace("RMSE:", "").replace("Max Err:", "").replace("Min AGL:", "").split()
            rmse = float(parts[0])
            max_err = float(parts[1])
            min_agl = float(parts[2])
            break

    # Find the generated .npz log file (pattern: mujoco_run_<controller>.npz)
    log_file = next(BASE_DIR.glob(f"mujoco_run_{controller}.npz"), None)

    # Write a row into the CSV summary
    csv_writer.writerow([controller, terrain, PAYLOAD, WIND_FORCE,
                         WIND_DIR, rmse, max_err, min_agl,
                         str(log_file) if log_file else ""])
    print(f"[DONE] {controller} on {terrain}")

# ------------------------------------------------------------
# Main loop: iterate over terrains & controllers
# ------------------------------------------------------------
for terrain in TERRAINS:
    for ctrl in CONTROLLERS.keys():
        run_one(ctrl, terrain)

csv_file.close()
print(f"\nAll runs finished. Summary CSV written to: {CSV_PATH}")