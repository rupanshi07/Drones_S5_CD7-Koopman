import numpy as np
from scipy.io import savemat

scenarios = [1, 2, 3, 4, 5, 6]
controllers = {"mpc": "scenario", "pid": "pid_scenario", "lqr": "lqr_scenario"}

for label, prefix in controllers.items():
    for s in scenarios:
        data = np.load(f"results/{prefix}_{s}.npz")
        states = data["states"]
        savemat(f"results/matlab_export/{label}_scenario_{s}.mat", {
            "states": states,
            "x": states[:, 0],
            "y": states[:, 1],
            "z": states[:, 2],
            "roll": states[:, 7],
            "pitch": states[:, 8],
            "yaw": states[:, 9]
        })
        print(f"Exported {label}_scenario_{s}.mat")
