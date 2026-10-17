"""MuJoCo stand-in for the gym-pybullet-drones CF2X environment used by the Koopman-MPC repo.

State layout (18) is the repo's Phase-2 layout:  [p(3), v(3), vec(R)(9, row-major), omega(3)]
with omega in the WORLD frame (as PyBullet's getBaseVelocity returns it).
Action: 4 motor RPMs.
"""
import os
import numpy as np
import mujoco
from terrain import make_terrain, terrain_height, build_preset, PRESETS


class MujocoQuadEnv:
    MASS, G = 0.027, 9.8
    KF, KM = 3.16e-10, 7.94e-12
    HOVER_RPM = float(np.sqrt(MASS * G / (4 * KF)))               # ~14468
    MAX_RPM = float(np.sqrt(2.25 * MASS * G / (4 * KF)))          # ~21703

    def __init__(self, xml_path=None, ctrl_hz=48, physics_hz=240, terrain_seed=0, flat=False, terrain="rolling"):
        xml_path = xml_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "cf2x_terrain.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 1.0 / physics_hz
        self.model.opt.gravity[:] = [0.0, 0.0, -self.G]            # keep in sync with the controllers' G (MuJoCo default is 9.81)
        self.model.vis.map.znear = 0.001                            # let the viewer camera zoom right up to the 11 cm drone
        self.data = mujoco.MjData(self.model)
        self.substeps = int(round(physics_hz / ctrl_hz))
        self.dt = self.substeps / physics_hz                        # control period
        m = self.model
        xml_h = float(m.hfield_size[0, 2])
        need = max(p["height"] for p in PRESETS.values())
        if xml_h < need:
            print(f"WARNING: XML hfield max height {xml_h} m < tallest preset {need} m. MuJoCo's collision bounds are built from the XML "
                  f"size, so set <hfield ... size=\"6 6 {need} 0.1\"> or hills may not collide.")
        self.terrain_name = "flat" if flat else terrain
        self.set_terrain(self.terrain_name, terrain_seed)          # set BEFORE any viewer opens
        self.body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
        self.terrain_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        # propeller hinge joints (visual only; turned kinematically in step())
        self.prop_q, self.prop_v = [], []
        for i in range(4):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"pj{i}")
            if j >= 0:
                self.prop_q.append(int(m.jnt_qposadr[j])); self.prop_v.append(int(m.jnt_dofadr[j]))
        self.prop_angle = np.zeros(4)
        ref_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ref")
        self.mocap_id = int(m.body_mocapid[ref_body]) if ref_body >= 0 else -1

    # ---- core API ----
    def reset(self, pos=(0.0, 0.0, 1.0)):
        mujoco.mj_resetData(self.model, self.data)
        self.set_payload(0.0)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = [1, 0, 0, 0]                          # MuJoCo quat order is (w, x, y, z)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        d = self.data
        R = d.xmat[self.body_id].reshape(3, 3).copy()
        w_world = R @ d.qvel[3:6]                                   # free-joint qvel[3:6] is in the BODY frame
        return np.concatenate([d.qpos[0:3], d.qvel[0:3], R.reshape(9), w_world])

    def step(self, rpm):
        rpm = np.clip(np.asarray(rpm, dtype=float), 0.0, self.MAX_RPM)
        self.data.ctrl[:] = self.KF * rpm ** 2                      # thrust per motor, newtons
        if self.prop_q:                                             # spin the visual propellers (slowed 60x so it is visible)
            self.prop_angle += np.array([-1, 1, -1, 1]) * rpm * (2 * np.pi / 60.0) * self.dt / 60.0
            for i, (q, v) in enumerate(zip(self.prop_q, self.prop_v)):
                self.data.qpos[q] = self.prop_angle[i]
                self.data.qvel[v] = 0.0
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    # ---- terrain selection ----
    def set_terrain(self, name, seed=0):
        """Rebuild the heightfield in place. If a viewer is open, call viewer.update_hfield(0) afterwards."""
        m = self.model
        n = int(m.hfield_nrow[0])
        h, height = build_preset(name, n, seed, float(m.hfield_size[0, 0]))
        m.hfield_size[0, 2] = height
        m.hfield_data[int(m.hfield_adr[0]):int(m.hfield_adr[0]) + n * n] = h.ravel()
        self.terrain_name, self.terrain_seed = name, seed

    # ---- disturbances (same idea as the repo: payload = extra mass, wind = horizontal force) ----
    def set_payload(self, frac):
        """Extra mass as a fraction of the nominal mass (0.25 = +25 %). Inertia is left unchanged."""
        self.model.body_mass[self.body_id] = self.MASS * (1.0 + frac)

    def set_wind(self, fx=0.0, fy=0.0):
        self.data.xfrc_applied[self.body_id, 0:2] = (fx, fy)

    def apply_wind(self, wind_vec, cda=(0.0033, 0.0033, 0.006), rho=1.2):
        """Aerodynamic drag from a wind vector (m/s): F = 0.5*rho*CdA*|v_rel|*v_rel, v_rel = wind - drone velocity.
        Returns the force (N). The wind speed comes from the terrain-aware WindModel in drone_sim.py."""
        rel = np.asarray(wind_vec, float) - self.data.qvel[0:3]
        F = 0.5 * rho * np.asarray(cda) * np.linalg.norm(rel) * rel
        self.data.xfrc_applied[self.body_id, 0:3] = F
        return F

    def set_ref_marker(self, pos):
        if self.mocap_id >= 0:
            self.data.mocap_pos[self.mocap_id] = pos

    # ---- terrain helpers ----
    def ground_height(self, x, y):
        return terrain_height(self.model, x, y)

    def agl(self):
        p = self.data.qpos[0:3]
        return float(p[2] - self.ground_height(p[0], p[1]))

    def touching_terrain(self):
        d = self.data
        return any(c.geom1 == self.terrain_geom or c.geom2 == self.terrain_geom for c in d.contact[:d.ncon])

    def flipped(self):
        return self.data.xmat[self.body_id][8] < 0.0                # R_zz < 0  -> upside down