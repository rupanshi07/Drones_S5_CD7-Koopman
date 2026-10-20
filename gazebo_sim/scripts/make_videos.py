import os, sys, math, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "videos")
os.makedirs(OUT, exist_ok=True)

WIND_N, PAY_N, MID = 2.8, 2.0, 7.5
SCEN = {1: (None, None, "Nominal, no disturbance"),
        2: (0.0, None, "Wind 2.8 N from t = 0"),
        3: (0.0, None, "Wind 2.8 N from t = 0 (selector on in the Koopman setup)"),
        4: (MID, None, "Wind 2.8 N from t = 7.5 s"),
        5: (None, MID, "Payload 2 N from t = 7.5 s"),
        6: (0.0, MID, "Wind from t = 0, payload from t = 7.5 s")}
STEP = 2
HALF = 0.13

def render(ctrl, sc):
    path = os.path.join(RES, f"gazebo_{ctrl}_scenario_{sc}.npz")
    if not os.path.exists(path):
        print("missing", path)
        return
    d = np.load(path)
    S, t = d["states"], d["t"]
    x, z = S[:, 0], S[:, 2]
    roll, pitch = S[:, 6], S[:, 7]
    tilt = np.degrees(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1, 1)))
    wmean = d["actions"].mean(axis=1)
    target = float(d["target_z"])
    wind_on, pay_on, desc = SCEN[sc]
    idx = np.arange(0, len(t), STEP)
    fps = max(5, int(round(len(idx) / (t[-1] - t[0]))))

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    axz = fig.add_subplot(gs[0, 1])
    axx = fig.add_subplot(gs[1, 1])
    axw = fig.add_subplot(gs[2, 1])
    fig.suptitle(f"{ctrl.upper()} - scenario {sc}: {desc}", fontsize=13)

    xlo, xhi = min(-0.3, x.min() - 0.2), max(0.8, x.max() + 0.2)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.03, 0.55)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)  [wind pushes along +x]")
    ax.set_ylabel("height z (m)")
    ax.axhspan(-0.03, 0, color="0.8")
    ax.axhline(target, color="g", ls="--", lw=1)
    ax.text(xhi - 0.02, target + 0.01, "target 0.3 m", color="g", ha="right", fontsize=9)
    trail, = ax.plot([], [], color="tab:blue", lw=1, alpha=0.6)
    bar, = ax.plot([], [], color="k", lw=5, solid_capstyle="round")
    thrust, = ax.plot([], [], color="tab:orange", lw=2)
    wind_ann = ax.annotate("", xy=(xlo + 0.2, 0.50), xytext=(xlo + 0.03, 0.50),
                           arrowprops=dict(arrowstyle="-|>", color="tab:red", lw=3))
    wind_txt = ax.text(xlo + 0.03, 0.52, "wind 2.8 N", color="tab:red", fontsize=10)
    pay_ann = ax.annotate("", xy=(xlo + 0.32, 0.40), xytext=(xlo + 0.32, 0.48),
                          arrowprops=dict(arrowstyle="-|>", color="tab:purple", lw=3))
    pay_txt = ax.text(xlo + 0.35, 0.43, "payload 2 N", color="tab:purple", fontsize=10)
    clock = ax.text(xhi - 0.02, 0.50, "", ha="right", fontsize=11)
    for a in (wind_ann, wind_txt, pay_ann, pay_txt):
        a.set_visible(False)

    cursors = []
    for a, y, lab in ((axz, z, "z (m)"), (axx, x, "x (m)"), (axw, wmean, "mean motor (rad/s)")):
        a.plot(t, y, color="tab:blue", lw=1.2)
        a.set_ylabel(lab, fontsize=9)
        a.grid(alpha=0.3)
        for onset, col in ((wind_on, "tab:red"), (pay_on, "tab:purple")):
            if onset is not None:
                a.axvline(onset, color=col, ls=":", lw=1)
        cursors.append(a.axvline(t[0], color="k", lw=1))
    axz.axhline(target, color="g", ls="--", lw=1)
    axw.set_xlabel("time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(k):
        i = idx[k]
        th = pitch[i]
        dx, dz = HALF * math.cos(th), -HALF * math.sin(th)
        bar.set_data([x[i] - dx, x[i] + dx], [z[i] - dz, z[i] + dz])
        thrust.set_data([x[i], x[i] + 0.12 * math.sin(th)], [z[i], z[i] + 0.12 * math.cos(th)])
        trail.set_data(x[:i + 1], z[:i + 1])
        wind_vis = wind_on is not None and t[i] >= wind_on
        pay_vis = pay_on is not None and t[i] >= pay_on
        wind_ann.set_visible(wind_vis); wind_txt.set_visible(wind_vis)
        pay_ann.set_visible(pay_vis); pay_txt.set_visible(pay_vis)
        clock.set_text(f"t = {t[i]:.1f} s   tilt = {tilt[i]:.1f} deg")
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        return ()

    anim = FuncAnimation(fig, update, frames=len(idx), blit=False)
    if shutil.which("ffmpeg"):
        out = os.path.join(OUT, f"gazebo_{ctrl}_scenario_{sc}.mp4")
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=1800))
    else:
        out = os.path.join(OUT, f"gazebo_{ctrl}_scenario_{sc}.gif")
        anim.save(out, writer=PillowWriter(fps=min(fps, 12)))
    plt.close(fig)
    print("saved", out)

ctrls = [sys.argv[1]] if len(sys.argv) > 1 else ["pid", "lqr"]
scs = [int(sys.argv[2])] if len(sys.argv) > 2 else range(1, 7)
for c in ctrls:
    for s in scs:
        render(c, s)
