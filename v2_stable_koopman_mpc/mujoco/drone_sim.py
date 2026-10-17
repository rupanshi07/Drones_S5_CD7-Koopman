"""CF2X drone in MuJoCo: choose terrain, control wind + payload live, compare PID / LQR / Koopman-MPC.

  python drone_sim.py                          # interactive: 3-D viewer + control panel
  python drone_sim.py --terrain mountain --ctrl pid
  python drone_sim.py --stress                 # survival test: max wind / max payload each controller can hold, per terrain
  python drone_sim.py --stress --quick         # smaller / faster version
  python drone_sim.py --demo --terrain rugged --wind 8 --payload 0.4     # headless run, prints numbers

Options: --terrain flat|gentle|rolling|rugged|mountain   --seed N   --ctrl pid|lqr|koopman   --agl 1.5   --amp 2.5
         interactive start values: --wind 8 (m/s)  --dir 90 (deg)  --payload-pct 40
Wind model (see WindModel): wind speed is set at 2 m above ground; the terrain changes it:
  * height profile  v ~ (h/2m)^alpha          (rougher terrain -> bigger alpha)
  * hilltop speed-up / valley shelter         (speedup * (ground height - mean height) / max height)
  * orographic lift: wind blowing up a slope gives an updraft, down a slope a downdraft
  * turbulence (gusts) that grows with terrain roughness and local slope
  Force on the drone = 0.5 * rho * CdA * |v_wind - v_drone| * (v_wind - v_drone)
"""
import sys
import time
import csv
import collections
import numpy as np
import mujoco
from mujoco_quad_env import MujocoQuadEnv
from controllers import make_pid, make_lqr
from terrain import PRESETS

CTRL_NAMES = ["PID", "LQR", "Koopman-MPC"]
CTRL_KEY = {"pid": "PID", "lqr": "LQR", "koopman": "Koopman-MPC"}
CTRL_RGBA = {"PID": (1.0, 0.55, 0.0, 1), "LQR": (0.6, 0.25, 1.0, 1), "Koopman-MPC": (1.0, 0.15, 0.15, 1)}
WIND_RAMP = 3.0                 # m/s per second: how fast the wind follows the slider
MAX_WIND = 16.0


def arg(name, default, cast=float):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


# ====================================================================== wind
class WindModel:
    def __init__(self, env):
        self.env = env
        self.target = 0.0          # m/s, set by the slider
        self.speed = 0.0           # m/s, ramps toward target
        self.dir_deg = 0.0         # direction the wind blows TOWARD (0 = +x, 90 = +y)
        self.terrain_effect = True
        self.turb_on = True
        self.gust = 0.0
        self.rng = np.random.default_rng(1)
        self._mean_key, self._mean_g = None, 0.0

    def mean_ground(self):
        """Mean terrain height (m), cached per terrain: wind speeds up above it and is sheltered below it."""
        m = self.env.model
        key = (self.env.terrain_name, self.env.terrain_seed)
        if key != self._mean_key:
            n = int(m.hfield_nrow[0]); adr = int(m.hfield_adr[0])
            self._mean_g = float(m.hfield_data[adr:adr + n * n].mean() * m.hfield_size[0, 2])
            self._mean_key = key
        return self._mean_g

    @property
    def p(self):
        return PRESETS[self.env.terrain_name]

    def reset(self):
        self.speed = 0.0
        self.gust = 0.0
        self.rng = np.random.default_rng(1)

    def update(self, dt):
        self.speed += float(np.clip(self.target - self.speed, -WIND_RAMP * dt, WIND_RAMP * dt))
        tau = 2.0                                                    # gust correlation time (s)
        self.gust += -self.gust / tau * dt + self.p["turb"] * np.sqrt(2 * dt / tau) * self.rng.standard_normal()
        self.gust = float(np.clip(self.gust, -0.8, 0.8))

    def slope(self, x, y, d=0.1):
        g = self.env.ground_height
        return (g(x + d, y) - g(x - d, y)) / (2 * d), (g(x, y + d) - g(x, y - d)) / (2 * d)

    def at(self, x, y, z, local=True):
        """Wind vector (m/s) at world position (x, y, z)."""
        env, p = self.env, self.p
        th = np.radians(self.dir_deg)
        u = np.array([np.cos(th), np.sin(th)])
        g = env.ground_height(x, y)
        gmax = float(env.model.hfield_size[0, 2])
        prof = su = 1.0
        gust = self.gust if self.turb_on else 0.0
        if self.terrain_effect:
            h = max(z - g, 0.3)
            prof = (h / 2.0) ** p["alpha"]
            su = max(1.0 + p["speedup"] * (g - self.mean_ground()) / gmax, 0.5) if gmax > 0.05 else 1.0
            if local:
                sx, sy = self.slope(x, y)
                gust *= 1.0 + 3.0 * np.hypot(sx, sy)                 # rough, steep ground -> gustier
        mag = self.speed * prof * su * max(1.0 + gust, 0.0)
        wz = 0.0
        if self.terrain_effect and mag > 0:
            sx, sy = self.slope(x, y)
            wz = float(np.clip(0.5 * (sx * u[0] + sy * u[1]) * mag, -0.5 * mag, 0.5 * mag))   # orographic lift
        return np.array([mag * u[0], mag * u[1], wz])


# ====================================================================== simulation core
class Sim:
    RATE = 0.4          # max altitude-reference rate, m/s

    def __init__(self, env=None, ctrls=None, terrain="rolling", seed=0, ctrl="Koopman-MPC", agl=1.5, amp=2.5):
        self.env = env or MujocoQuadEnv(terrain=terrain, terrain_seed=seed)
        self.ctrls = ctrls if ctrls is not None else {}
        self.wind = WindModel(self.env)
        self.agl, self.params = agl, (amp, amp, 0.30, 0.25, 0.0, 1.0)
        self.ctrl_name, self.payload = ctrl, 0.0
        self.events, self.trail = [], collections.deque(maxlen=400)
        self.reset()

    def controller(self):
        if self.ctrl_name not in self.ctrls:
            if self.ctrl_name == "PID":
                self.ctrls["PID"] = make_pid(self.env)
            elif self.ctrl_name == "LQR":
                self.ctrls["LQR"] = make_lqr(self.env)
            else:
                from koopman_mpc import KoopmanMPC
                self.ctrls["Koopman-MPC"] = KoopmanMPC(self.env)
        return self.ctrls[self.ctrl_name]

    def xy(self, t):
        ax, ay, wx, wy, px, py = self.params
        return ax * np.sin(wx * t + px), ay * np.sin(wy * t + py)

    def ground_ahead(self, t):
        return max(self.env.ground_height(*self.xy(t + d)) for d in (0.0, 0.4, 0.8, 1.2, 1.6))

    def path_points(self):
        return [np.array([*self.xy(t), self.env.ground_height(*self.xy(t)) + self.agl]) for t in np.arange(0, 25.2, 0.5)]

    def reset(self):
        env = self.env
        x0, y0 = self.xy(0.0)
        r0 = np.array([x0, y0, self.ground_ahead(0.0) + self.agl])
        self.s = env.reset(r0)
        env.set_payload(self.payload)
        self.z_ref, self.k, self.crashed, self.crash_t = r0[2], 0, False, None
        self.dist = self.ref_dist = 0.0
        self.last_p, self.last_r = r0.copy(), r0.copy()
        self.hist, self.events = [], []
        self.trail.clear()
        self.w_local = np.zeros(3)
        self.ref = r0.copy()
        self.wind.reset()
        self.controller().reset()
        env.set_ref_marker(r0)

    def log(self, msg):
        self.events.append((self.k * self.env.dt, msg))

    def set_payload(self, frac):
        self.payload = float(frac)
        self.env.set_payload(self.payload)

    def reference(self, t):
        z_goal = self.ground_ahead(t) + self.agl
        self.z_ref += float(np.clip(z_goal - self.z_ref, -self.RATE * self.env.dt, self.RATE * self.env.dt))
        return np.array([*self.xy(t), self.z_ref])

    def step(self):
        env = self.env
        t = self.k * env.dt
        ref = self.reference(t)
        self.wind.update(env.dt)
        p = self.s[:3]
        self.w_local = self.wind.at(p[0], p[1], p[2])
        if self.wind.speed > 0.01:
            env.apply_wind(self.w_local)
        else:
            env.data.xfrc_applied[env.body_id, 0:3] = 0.0
        self.s = env.step(self.controller().rpm(self.s, ref))
        env.set_ref_marker(ref)
        self.ref = ref
        p = self.s[:3]
        self.dist += float(np.linalg.norm(p - self.last_p)); self.ref_dist += float(np.linalg.norm(ref - self.last_r))
        self.last_p, self.last_r = p.copy(), ref.copy()
        tilt = float(np.degrees(np.arccos(np.clip(self.s[14], -1, 1))))
        err = float(np.linalg.norm(p - ref))
        self.hist.append([t, err, env.agl(), ref[2] - p[2], float(np.hypot(*self.w_local[:2])), self.w_local[2],
                          self.dist, self.payload * 100, tilt, *p, *ref])
        if self.k % 2 == 0:
            self.trail.append(p.copy())
        self.k += 1
        if env.touching_terrain() or env.flipped():
            self.crashed, self.crash_t = True, t
            self.log("CRASHED (terrain contact or flip)")

    def status(self):
        if self.crashed:
            return "CRASHED"
        h = self.hist[-96:]
        if len(h) < 48:
            return "OK"
        a = np.array(h)
        if a[:, 3].mean() > 0.25:
            return "ALTITUDE DECAY"
        if a[:, 1].mean() > 0.35:
            return "OFF PATH"
        return "OK"


# ====================================================================== drawing helpers (viewer overlay)
def add_geom(scn, kind, p, q, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, kind, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, kind, width, np.asarray(p, float), np.asarray(q, float))
    scn.ngeom += 1


def speed_color(v):
    c = min(v / MAX_WIND, 1.0)
    return (c, 0.25 + 0.5 * (1 - abs(2 * c - 1)), 1.0 - c, 0.85)


def draw_overlay(scn, sim, path_pts, show_field=True, show_trail=True, show_path=True, clear=True):
    if clear:
        scn.ngeom = 0
    ARROW, CAP = mujoco.mjtGeom.mjGEOM_ARROW, mujoco.mjtGeom.mjGEOM_CAPSULE
    env, w = sim.env, sim.wind
    if show_path:
        for a, b in zip(path_pts[:-1], path_pts[1:]):
            add_geom(scn, CAP, a, b, 0.003, (0.2, 0.9, 0.3, 0.45))
    if show_trail and len(sim.trail) > 1:
        col = CTRL_RGBA[sim.ctrl_name][:3] + (0.9,)
        pts = list(sim.trail)
        for a, b in zip(pts[:-1], pts[1:]):
            add_geom(scn, CAP, a, b, 0.004, col)
    if show_field and w.speed > 0.05:                               # wind field arrows over the whole terrain
        for x in np.linspace(-5, 5, 9):
            for y in np.linspace(-5, 5, 9):
                z = env.ground_height(x, y) + sim.agl
                v = w.at(x, y, z, local=False)
                sp = float(np.linalg.norm(v))
                if sp > 0.05:
                    add_geom(scn, ARROW, [x, y, z], np.array([x, y, z]) + 0.09 * v, 0.012, speed_color(sp))
    p = sim.s[:3]
    if w.speed > 0.05:                                              # wind actually hitting the drone
        sp = float(np.linalg.norm(sim.w_local))
        add_geom(scn, ARROW, p + [0, 0, 0.08], p + [0, 0, 0.08] + 0.10 * sim.w_local, 0.008, (0.0, 0.9, 1.0, 1.0))


def hud_text(sim):
    w = sim.w_local
    left = ("Controller\nTerrain\nTime\nDistance flown\nTracking error\nHeight above ground\n"
            "Wind at drone\nWind vertical\nPayload\nStatus")
    right = (f"{sim.ctrl_name}\n{sim.env.terrain_name} (seed {sim.env.terrain_seed})\n{sim.k * sim.env.dt:7.2f} s\n"
             f"{sim.dist:7.2f} m\n{(sim.hist[-1][1] if sim.hist else 0):7.3f} m\n{sim.env.agl():7.2f} m\n"
             f"{np.hypot(w[0], w[1]):6.1f} m/s\n{w[2]:+6.1f} m/s\n{sim.payload:+7.0%}\n{sim.status()}")
    return left, right


# ====================================================================== interactive app (Tk panel + MuJoCo viewer)
class App:
    def __init__(self):
        import tkinter as tk
        from tkinter import ttk
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import mujoco.viewer
        self.tk, self.ttk = tk, ttk
        terrain = arg("--terrain", "rolling", str)
        self.sim = Sim(terrain=terrain, seed=arg("--seed", 0, int), ctrl=CTRL_KEY[arg("--ctrl", "koopman", str).lower()],
                       agl=arg("--agl", 1.5), amp=arg("--amp", 2.5))
        self.path_pts = self.sim.path_points()
        self.paused, self.viewer, self.ev_shown = False, None, 0
        self.next_t = time.perf_counter()

        root = self.root = tk.Tk()
        root.title("CF2X drone: terrain, wind and payload control")
        left = ttk.Frame(root, padding=8); left.grid(row=0, column=0, sticky="n")
        right = ttk.Frame(root, padding=8); right.grid(row=0, column=1, sticky="n")

        # --- controller + terrain
        box = ttk.LabelFrame(left, text="Controller"); box.pack(fill="x", pady=3)
        self.v_ctrl = tk.StringVar(value=self.sim.ctrl_name)
        for n in CTRL_NAMES:
            ttk.Radiobutton(box, text=n, value=n, variable=self.v_ctrl, command=self.on_ctrl).pack(side="left", padx=6)
        box = ttk.LabelFrame(left, text="Terrain"); box.pack(fill="x", pady=3)
        self.v_terrain = tk.StringVar(value=terrain)
        ttk.Combobox(box, textvariable=self.v_terrain, values=list(PRESETS), width=10, state="readonly").pack(side="left", padx=4)
        ttk.Label(box, text="seed").pack(side="left")
        self.v_seed = tk.IntVar(value=self.sim.env.terrain_seed)
        ttk.Spinbox(box, from_=0, to=99, textvariable=self.v_seed, width=4).pack(side="left", padx=4)
        ttk.Button(box, text="Apply terrain", command=self.on_terrain).pack(side="left", padx=4)

        # --- disturbances
        box = ttk.LabelFrame(left, text="Disturbances (live)"); box.pack(fill="x", pady=3)
        def slider(label, lo, hi, res, var, cmd, key):
            tk.Label(box, text=label, anchor="w").pack(fill="x")
            s = tk.Scale(box, from_=lo, to=hi, resolution=res, orient="horizontal", length=330, variable=var, command=cmd)
            s.pack(); s.bind("<ButtonRelease-1>", lambda e: self.on_release(key)); return s
        self.v_wind, self.v_dir = tk.DoubleVar(value=arg("--wind", 0.0)), tk.DoubleVar(value=arg("--dir", 0.0))
        self.v_pay, self.v_cam = tk.DoubleVar(value=arg("--payload-pct", 0.0)), tk.DoubleVar(value=1.2)
        self.sim.wind.target, self.sim.wind.dir_deg = self.v_wind.get(), self.v_dir.get()
        self.sim.set_payload(self.v_pay.get() / 100.0)
        slider("Wind speed at 2 m height (m/s)", 0, MAX_WIND, 0.5, self.v_wind, self.on_wind, "wind")
        slider("Wind blows toward (deg: 0 = +x, 90 = +y)", 0, 359, 5, self.v_dir, self.on_dir, "dir")
        slider("Extra payload (% of drone mass)", 0, 100, 5, self.v_pay, self.on_pay, "payload")
        slider("Camera distance (m)", 0.4, 6, 0.1, self.v_cam, self.on_cam, "cam")
        self.v_terr_wind, self.v_turb = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.v_field, self.v_trail = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        for txt, var, cmd in [("Terrain changes the wind (speed-up, updraft)", self.v_terr_wind, self.on_flags),
                              ("Turbulence / gusts", self.v_turb, self.on_flags),
                              ("Show wind-field arrows", self.v_field, None), ("Show flight trail", self.v_trail, None)]:
            ttk.Checkbutton(box, text=txt, variable=var, command=cmd).pack(anchor="w")

        box = ttk.Frame(left); box.pack(fill="x", pady=4)
        self.b_pause = ttk.Button(box, text="Pause", command=self.on_pause); self.b_pause.pack(side="left", padx=2)
        ttk.Button(box, text="Reset", command=self.on_reset).pack(side="left", padx=2)
        ttk.Button(box, text="Open viewer", command=self.open_viewer).pack(side="left", padx=2)
        ttk.Button(box, text="Save run", command=self.on_save).pack(side="left", padx=2)

        # --- live readout + event log
        box = ttk.LabelFrame(right, text="Live readout"); box.pack(fill="x", pady=3)
        self.readout = tk.StringVar()
        tk.Label(box, textvariable=self.readout, justify="left", font=("Consolas", 10), anchor="w").pack(fill="x")
        self.v_status = tk.StringVar()
        self.l_status = tk.Label(box, textvariable=self.v_status, font=("Consolas", 14, "bold")); self.l_status.pack(fill="x")
        box = ttk.LabelFrame(right, text="Event log (timestamps)"); box.pack(fill="x", pady=3)
        self.log = tk.Text(box, height=6, width=62, font=("Consolas", 9)); self.log.pack()
        self.fig = Figure(figsize=(6.2, 5.2), dpi=90)
        self.axs = self.fig.subplots(3, 1, sharex=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right); self.canvas.get_tk_widget().pack(pady=3)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.on_flags()
        self.open_viewer()
        self.last_ui = self.last_plot = 0.0
        root.lift(); root.attributes("-topmost", True)              # bring the control panel in front of the 3-D viewer
        root.after(2500, lambda: root.attributes("-topmost", False))
        root.after(5, self.tick)

    # ---- viewer
    def open_viewer(self):
        if self.viewer is not None and self.viewer.is_running():
            return
        import mujoco.viewer
        env = self.sim.env
        self.viewer = mujoco.viewer.launch_passive(env.model, env.data, show_left_ui=False, show_right_ui=False)
        cam = self.viewer.cam
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid, cam.distance, cam.elevation, cam.azimuth = env.body_id, self.v_cam.get(), -22, 135
        self.recolor()

    def recolor(self):
        m = self.sim.env.model
        for name in ("battery",):
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
            m.geom_rgba[gid] = CTRL_RGBA[self.sim.ctrl_name]

    def lock(self):
        class _N:
            def __enter__(s): return None
            def __exit__(s, *a): return False
        return self.viewer.lock() if self.viewer is not None else _N()

    # ---- callbacks
    def on_ctrl(self):
        self.sim.ctrl_name = self.v_ctrl.get()
        with self.lock():
            self.sim.reset(); self.recolor()
        self.sim.log(f"controller -> {self.sim.ctrl_name} (run restarted)"); self.clear_log()

    def on_terrain(self):
        with self.lock():
            self.sim.env.set_terrain(self.v_terrain.get(), int(self.v_seed.get()))
            self.path_pts = self.sim.path_points()
            self.sim.reset()
        if self.viewer is not None and self.viewer.is_running():
            if hasattr(self.viewer, "update_hfield"):
                self.viewer.update_hfield(0)             # push the new terrain to the renderer
            else:
                self.viewer.close(); self.open_viewer()
        self.sim.log(f"terrain -> {self.v_terrain.get()} seed {int(self.v_seed.get())} (run restarted)"); self.clear_log()

    def on_wind(self, v): self.sim.wind.target = float(v)
    def on_dir(self, v): self.sim.wind.dir_deg = float(v)
    def on_pay(self, v):
        with self.lock():
            self.sim.set_payload(float(v) / 100.0)
    def on_cam(self, v):
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.cam.distance = float(v)
    def on_flags(self):
        self.sim.wind.terrain_effect, self.sim.wind.turb_on = self.v_terr_wind.get(), self.v_turb.get()
    def on_release(self, key):
        if key == "wind":
            self.sim.log(f"wind target {self.v_wind.get():.1f} m/s @ {self.v_dir.get():.0f} deg")
        elif key == "dir":
            self.sim.log(f"wind direction {self.v_dir.get():.0f} deg")
        elif key == "payload":
            self.sim.log(f"payload {self.v_pay.get():+.0f} %")
    def on_pause(self):
        self.paused = not self.paused
        self.b_pause.config(text="Resume" if self.paused else "Pause")
    def on_reset(self):
        with self.lock():
            self.sim.reset()
        self.clear_log()
    def clear_log(self):
        self.log.delete("1.0", "end"); self.ev_shown = 0
    def on_save(self):
        if not self.sim.hist:
            return
        tag = time.strftime("%H%M%S")
        base = f"drone_run_{self.sim.ctrl_name}_{self.sim.env.terrain_name}_{tag}"
        cols = ["t", "err", "agl", "alt_deficit", "wind_speed", "wind_vz", "distance", "payload_pct", "tilt_deg",
                "x", "y", "z", "ref_x", "ref_y", "ref_z"]
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(cols); w.writerows(self.sim.hist)
        self.fig.savefig(base + ".png", dpi=130)
        with open(base + "_events.txt", "w") as f:
            f.write("\n".join(f"[{t:7.2f} s] {m}" for t, m in self.sim.events))
        self.sim.log(f"saved {base}.csv/.png")

    def on_close(self):
        try:
            if self.viewer is not None:
                self.viewer.close()
                time.sleep(0.3)
        finally:
            self.root.destroy()

    # ---- main loop
    def tick(self):
        sim, dt = self.sim, self.sim.env.dt
        now = time.perf_counter()
        stepped = False
        if self.viewer is not None and not self.viewer.is_running():
            self.viewer = None
        if not self.paused and not sim.crashed:
            n = 0
            while now >= self.next_t and n < 3:
                with self.lock():
                    sim.step()
                n += 1; stepped = True; self.next_t += dt
            if now - self.next_t > 0.25:
                self.next_t = now
        else:
            self.next_t = now
        if self.viewer is not None and (stepped or self.paused):
            if sim.k % 3 == 0:
                with self.lock():
                    draw_overlay(self.viewer.user_scn, sim, self.path_pts, self.v_field.get(), self.v_trail.get())
                l, r = hud_text(sim)                     # set_texts takes the viewer lock itself -> call it OUTSIDE our lock
                self.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, l, r))
            self.viewer.sync()
        if now - self.last_ui > 0.15:
            self.last_ui = now
            self.update_ui()
        if now - self.last_plot > 0.6 and sim.hist:
            self.last_plot = now
            self.update_plot()
        self.root.after(2, self.tick)

    def update_ui(self):
        sim = self.sim
        h = sim.hist[-1] if sim.hist else [0] * 15
        w = sim.w_local
        self.readout.set(
            f"time            {sim.k * sim.env.dt:7.2f} s\n"
            f"distance flown  {sim.dist:7.2f} m   (reference {sim.ref_dist:.2f} m)\n"
            f"tracking error  {h[1]:7.3f} m\n"
            f"height AGL      {h[2]:7.2f} m   (clearance goal >= {sim.agl:.2f} m, deficit {h[3]:+.2f} m)\n"
            f"tilt            {h[8]:7.1f} deg\n"
            f"wind at drone   {np.hypot(w[0], w[1]):7.1f} m/s  vertical {w[2]:+.1f} m/s  (slider {sim.wind.speed:.1f})\n"
            f"payload         {sim.payload:+7.0%}")
        st = sim.status()
        self.v_status.set(f"STATUS: {st}")
        self.l_status.config(fg={"OK": "#1a8f2e", "OFF PATH": "#c77700"}.get(st, "#c00000"))
        for t, m in sim.events[self.ev_shown:]:
            self.log.insert("end", f"[{t:7.2f} s] {m}\n"); self.log.see("end")
        self.ev_shown = len(sim.events)

    def update_plot(self):
        sim = self.sim
        a = np.array(sim.hist[-1900:])
        t = a[:, 0]
        for ax in self.axs:
            ax.clear(); ax.grid(alpha=.3)
        self.axs[0].plot(t, a[:, 4], color="tab:blue", label="wind speed at drone"); self.axs[0].plot(t, a[:, 5], color="tab:cyan", label="vertical wind")
        self.axs[0].set_ylabel("m/s"); self.axs[0].legend(fontsize=7, loc="upper left")
        self.axs[1].plot(t, a[:, 1], color="tab:red", label="tracking error"); self.axs[1].plot(t, a[:, 3], color="tab:orange", label="altitude deficit")
        self.axs[1].set_ylabel("m"); self.axs[1].legend(fontsize=7, loc="upper left")
        self.axs[2].plot(t, a[:, 2], color="tab:green", label="height above ground"); self.axs[2].axhline(sim.agl, color="k", ls="--", lw=0.8, label="target")
        self.axs[2].set_ylabel("m"); self.axs[2].set_xlabel("time (s)"); self.axs[2].legend(fontsize=7, loc="upper left")
        for et, msg in sim.events:
            if t[0] <= et <= t[-1]:
                for ax in self.axs:
                    ax.axvline(et, color="gray", ls=":", lw=1)
        self.fig.tight_layout(); self.canvas.draw_idle()


# ====================================================================== headless survival / stress test
def run_case(sim, wind, payload, T=18.0, t_on=3.0):
    """Disturbances switch on at t_on. Survived = no crash AND (last 3 s) error < 0.35 m AND altitude deficit < 0.30 m."""
    sim.reset(); sim.set_payload(0.0); sim.wind.target = 0.0
    dt = sim.env.dt
    for k in range(int(T / dt)):
        if k == int(round(t_on / dt)):
            sim.wind.target = wind; sim.set_payload(payload)
        sim.step()
        if sim.crashed:
            break
    h = np.array(sim.hist); last = h[h[:, 0] >= h[-1, 0] - 3.0]
    end_err, end_def = float(last[:, 1].mean()), float(last[:, 3].mean())
    survived = bool((not sim.crashed) and end_err < 0.35 and end_def < 0.30 and len(h) > int(T / dt) - 2)
    reason = "ok" if survived else ("crashed" if sim.crashed else ("altitude decay" if end_def >= 0.30 else "blown off path"))
    return dict(survived=survived, reason=reason,
                crashed=sim.crashed, crash_t=sim.crash_t, end_err=float(last[:, 1].mean()), end_deficit=float(last[:, 3].mean()),
                max_deficit=float(h[:, 3].max()), min_agl=float(h[:, 2].min()), distance=float(h[-1, 6]),
                max_wind=float(h[:, 4].max()))


def stress(quick=False):
    terrains = [arg("--terrain", None, str)] if "--terrain" in sys.argv else (["rolling", "mountain"] if quick else list(PRESETS))
    winds = [0, 4, 8, 12, 16] if quick else [0, 2, 4, 6, 8, 10, 12, 14, 16]
    loads = [0, 40, 80, 100] if quick else [0, 20, 40, 60, 80, 100]
    seed = arg("--seed", 0, int)
    env = MujocoQuadEnv(terrain=terrains[0], terrain_seed=seed)
    sim = Sim(env=env, ctrls={}, ctrl="PID")
    rows, best = [], {}
    print(f"stress test | terrains {terrains} | wind steps {winds} m/s | payload steps {loads} %  (disturbance starts at t=3 s, run 18 s)")
    for terr in terrains:
        env.set_terrain(terr, seed); sim.reset()
        for cname in CTRL_NAMES:
            sim.ctrl_name = cname
            for kind, levels in (("wind", winds), ("payload", loads)):
                ok_level, last = 0, None
                for lv in levels:
                    r = run_case(sim, lv if kind == "wind" else 0.0, lv / 100.0 if kind == "payload" else 0.0)
                    rows.append([terr, cname, kind, lv, r["survived"], r["crashed"], r["crash_t"], round(r["end_err"], 3),
                                 round(r["end_deficit"], 3), round(r["max_deficit"], 3), round(r["min_agl"], 3), round(r["distance"], 2)])
                    if not r["survived"]:
                        last = r; break
                    ok_level = lv
                best[(terr, cname, kind)] = (ok_level, last)
            print(f"  {terr:9s} {cname:12s} max wind {best[(terr, cname, 'wind')][0]:>3} m/s | max payload {best[(terr, cname, 'payload')][0]:>3} %")
    with open("stress_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["terrain", "controller", "disturbance", "level (m/s or %)", "survived", "crashed", "crash_time_s", "end_error_m",
                    "end_altitude_deficit_m", "max_altitude_deficit_m", "min_agl_m", "distance_m"])
        w.writerows(rows)
    print("\nHow each controller fails at the first level it cannot hold:")
    for (terr, cname, kind), (lvl, last) in best.items():
        if last is not None:
            why = f"crashed at {last['crash_t']:.1f} s" if last["crashed"] else \
                  f"{last['reason']} (altitude deficit {last['end_deficit']:.2f} m, tracking error {last['end_err']:.2f} m)"
            print(f"  {terr:9s} {cname:12s} {kind:8s} fails above {lvl} -> {why}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"PID": "tab:orange", "LQR": "tab:purple", "Koopman-MPC": "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(terrains))
    for j, (kind, unit) in enumerate([("wind", "m/s"), ("payload", "% extra mass")]):
        for i, cname in enumerate(CTRL_NAMES):
            vals = [best[(t, cname, kind)][0] for t in terrains]
            b = ax[j].bar(x + (i - 1) * 0.27, vals, 0.27, color=colors[cname], label=cname)
            ax[j].bar_label(b, fontsize=8)
        ax[j].set_xticks(x); ax[j].set_xticklabels(terrains); ax[j].set_ylabel(f"max sustained ({unit})")
        ax[j].set_title(f"Largest {kind} the controller can hold without crashing or losing altitude"); ax[j].legend(loc="upper left", fontsize=8); ax[j].grid(alpha=.3, axis="y"); ax[j].set_ylim(0, max(max(best[(t, c, kind)][0] for t in terrains for c in CTRL_NAMES) * 1.25, 1))
    fig.tight_layout(); fig.savefig("stress_summary.png", dpi=130)
    print("\nsaved stress_results.csv and stress_summary.png")


# ====================================================================== headless demo (no GUI)
def demo():
    terr, seed = arg("--terrain", "rolling", str), arg("--seed", 0, int)
    wind, pay, T = arg("--wind", 0.0), arg("--payload", 0.0), arg("--T", 18.0)
    env = MujocoQuadEnv(terrain=terr, terrain_seed=seed)
    ctrls = {}
    print(f"terrain {terr} (seed {seed}) | wind {wind} m/s from t=3 s | payload {pay:+.0%} from t=3 s | {T:g} s")
    print(f"{'controller':<13}{'status':<16}{'RMSE (m)':>9}{'max err':>9}{'min AGL':>9}{'distance':>10}{'max wind seen':>15}")
    for cname in (CTRL_NAMES if "--ctrl" not in sys.argv else [CTRL_KEY[arg("--ctrl", "koopman", str).lower()]]):
        sim = Sim(env=env, ctrls=ctrls, ctrl=cname)
        r = run_case(sim, wind, pay, T=T)
        h = np.array(sim.hist)
        print(f"{cname:<13}{sim.status():<16}{np.sqrt((h[:, 1] ** 2).mean()):>9.3f}{h[:, 1].max():>9.3f}{h[:, 2].min():>9.2f}{h[-1, 6]:>9.2f} m{r['max_wind']:>11.1f} m/s")


if __name__ == "__main__":
    if "--stress" in sys.argv:
        stress("--quick" in sys.argv)
    elif "--demo" in sys.argv:
        demo()
    else:
        App().root.mainloop()
        sys.stdout.flush()
        import os
        os._exit(0)                       # skip interpreter teardown (the viewer thread can abort it noisily)