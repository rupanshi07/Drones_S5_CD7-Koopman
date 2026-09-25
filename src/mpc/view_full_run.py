import numpy as np
log = np.load("results/crash_diagnostic.npy")
z = log[:,1]
print(f"{'step':>5s} {'t(s)':>6s} {'Z':>8s}")
for i in range(0, len(log), 4):
    print(f"{int(log[i,0]):5d} {log[i,0]/48:6.2f} {z[i]:8.4f}")
