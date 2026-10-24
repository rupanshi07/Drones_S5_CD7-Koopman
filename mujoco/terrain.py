"""Procedural heightfield generator + height lookup for the MuJoCo terrain.
Supports rich terrain presets and custom elevation maps.
"""
import numpy as np


def make_terrain(n=128, seed=0, half_size=6.0, n_waves=10, pad_radius=1.5, pad_blend=1.5):
    """Return an (n, n) array in [0, 1]. MuJoCo multiplies it by hfield size[2] (max height, m).
    Rows run along world y, columns along world x. A flat pad is kept around the origin (spawn point)."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(xs, xs)              # X varies along columns, Y along rows
    h = np.zeros((n, n))
    for _ in range(n_waves):
        k = rng.uniform(0.25, 1.0)          # spatial frequency, rad/m (wavelength 6-25 m)
        th, ph = rng.uniform(0, 2 * np.pi, 2)
        h += (1.0 / k) * np.sin(k * (np.cos(th) * X + np.sin(th) * Y) + ph)   # rougher = more small waves
    h = (h - h.min()) / (h.max() - h.min() + 1e-9)
    r = np.hypot(X, Y)
    t = np.clip((r - pad_radius) / pad_blend, 0.0, 1.0)
    mask = t * t * (3 - 2 * t)              # smoothstep: 0 on the pad, 1 outside
    return h * mask


def terrain_height(model, x, y, hf_id=0):
    """Bilinear terrain height (m) at world (x, y). Matches what the physics engine collides with."""
    nrow, ncol = int(model.hfield_nrow[hf_id]), int(model.hfield_ncol[hf_id])
    rx, ry, ez, _ = model.hfield_size[hf_id]
    adr = int(model.hfield_adr[hf_id])
    d = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol)
    u = np.clip((x + rx) / (2 * rx) * (ncol - 1), 0, ncol - 1 - 1e-9)
    v = np.clip((y + ry) / (2 * ry) * (nrow - 1), 0, nrow - 1 - 1e-9)
    i, j = int(v), int(u)
    fv, fu = v - i, u - j
    z = (d[i, j] * (1 - fu) * (1 - fv) + d[i, j + 1] * fu * (1 - fv)
         + d[i + 1, j] * (1 - fu) * fv + d[i + 1, j + 1] * fu * fv)
    return float(z * ez)


def max_slope_deg(model, hf_id=0):
    nrow, ncol = int(model.hfield_nrow[hf_id]), int(model.hfield_ncol[hf_id])
    rx, ry, ez, _ = model.hfield_size[hf_id]
    adr = int(model.hfield_adr[hf_id])
    d = model.hfield_data[adr:adr + nrow * ncol].reshape(nrow, ncol) * ez
    gy, gx = np.gradient(d, 2 * ry / (nrow - 1), 2 * rx / (ncol - 1))
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max())))


# ---- terrain presets: shape + wind properties ----
PRESETS = {
    "flat":      dict(height=0.01, n_waves=1,  kmin=0.25, kmax=1.0, pad=1.5, alpha=0.12, turb=0.05, speedup=0.0, desc="Completely flat flight arena with spawn pad"),
    "gentle":    dict(height=0.35, n_waves=6,  kmin=0.25, kmax=0.7, pad=1.2, alpha=0.14, turb=0.08, speedup=0.3, desc="Soft rolling hills with low gradient"),
    "rolling":   dict(height=0.70, n_waves=10, kmin=0.25, kmax=1.0, pad=1.5, alpha=0.16, turb=0.12, speedup=0.5, desc="Standard undulating hills and valleys"),
    "rugged":    dict(height=1.20, n_waves=14, kmin=0.40, kmax=1.4, pad=1.0, alpha=0.20, turb=0.18, speedup=0.8, desc="Sharp crests, rocky ridges, steep elevation changes"),
    "mountain":  dict(height=1.80, n_waves=12, kmin=0.30, kmax=1.0, pad=1.0, alpha=0.26, turb=0.25, speedup=1.0, desc="Alpine mountains with high peaks and deep passes"),
    "terraced":  dict(height=1.00, levels=4, pad=1.2, alpha=0.18, turb=0.10, speedup=0.4, desc="Stepped architectural terraces / platforms"),
    "canyon":    dict(height=1.40, corridor_w=1.8, pad=1.2, alpha=0.22, turb=0.16, speedup=0.7, desc="A valley corridor flanked by steep lateral canyon walls"),
    "obstacles": dict(height=0.80, n_mounds=8, pad=1.2, alpha=0.16, turb=0.12, speedup=0.5, desc="Dispersed conical mounds and obstacle hills"),
}


def build_preset(name, n=128, seed=0, half_size=6.0):
    """Return (h in [0,1], max_height_m) for a named preset."""
    name = str(name).lower()
    if name not in PRESETS:
        print(f"Unknown terrain preset '{name}', defaulting to 'rolling'. Available: {list(PRESETS.keys())}")
        name = "rolling"

    p = PRESETS[name]
    xs = np.linspace(-half_size, half_size, n)
    X, Y = np.meshgrid(xs, xs)
    rng_state = np.random.default_rng(seed)

    if name == "flat":
        return np.zeros((n, n)), p["height"]

    elif name == "terraced":
        # Multi-tiered plateau terraces
        h_raw = np.sin(0.4 * X) * np.cos(0.4 * Y) + 0.5 * np.sin(0.8 * X + 0.3 * Y)
        h_raw = (h_raw - h_raw.min()) / (h_raw.max() - h_raw.min() + 1e-9)
        levels = p.get("levels", 4)
        h = np.floor(h_raw * levels) / levels

    elif name == "canyon":
        # High ridges on Y boundaries, central pass along X
        y_factor = np.clip((np.abs(Y) - p["corridor_w"]) / (half_size - p["corridor_w"]), 0.0, 1.0)
        h_canyon = y_factor ** 1.8
        # Add subtle undulation along corridor
        h_waves = 0.15 * np.sin(0.6 * X)
        h = np.clip(h_canyon + h_waves, 0.0, 1.0)

    elif name == "obstacles":
        # Dispersed mounds
        h = np.zeros((n, n))
        for _ in range(p.get("n_mounds", 8)):
            mx = rng_state.uniform(-half_size * 0.7, half_size * 0.7)
            my = rng_state.uniform(-half_size * 0.7, half_size * 0.7)
            if np.hypot(mx, my) < 1.5:  # skip spawn zone
                continue
            mw = rng_state.uniform(1.0, 2.2)
            mh = rng_state.uniform(0.5, 1.0)
            h += mh * np.exp(-((X - mx)**2 + (Y - my)**2) / (2 * mw**2))
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    else:
        # Wave-based terrain (gentle, rolling, rugged, mountain)
        h = np.zeros((n, n))
        for _ in range(p["n_waves"]):
            k = rng_state.uniform(p["kmin"], p["kmax"])
            th, ph = rng_state.uniform(0, 2 * np.pi, 2)
            h += (1.0 / k) * np.sin(k * (np.cos(th) * X + np.sin(th) * Y) + ph)
        h = (h - h.min()) / (h.max() - h.min() + 1e-9)

    # Keep spawn point flat and safe
    pad = p.get("pad", 1.5)
    t = np.clip((np.hypot(X, Y) - pad) / 1.5, 0.0, 1.0)
    smooth_mask = t * t * (3 - 2 * t)
    return h * smooth_mask, p["height"]


def list_terrains():
    """Returns a dict of all terrain names and descriptions."""
    return {k: v.get("desc", "") for k, v in PRESETS.items()}