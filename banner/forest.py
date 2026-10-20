import importlib.util, json, math, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import foreground as procedural
import terrain as T
import render as R

HERE = Path(__file__).resolve().parent
ASSETS = Path(os.environ.get("BANNER_ASSETS", HERE.parent / ".assets"))
CAM_Z = 20.0


def lin(c):
    return (np.clip(np.asarray(c, float), 0, 1) ** 2.2).tolist()


def ground_grid(rng, half_fov):
    y0, y1, dy, dx = 25.0, 262.0, 2.0, 2.0
    ys = np.arange(y0, y1 + dy, dy)
    xmax = y1 * math.tan(half_fov) * 1.12
    xs = np.arange(-xmax, xmax + dx, dx)
    X, Y = np.meshgrid(xs, ys)
    z = np.zeros_like(X)
    for f, a in ((.006, 5.0), (.017, 2.2), (.045, .9), (.11, .35)):
        ph = rng.uniform(0, 6.3, 2)
        z += a * np.sin(X * f + ph[0]) * np.cos(Y * f * .8 + ph[1])
    z += np.clip((Y - 215) / 47, 0, 1) * 3.5 * (np.sin(X * .021 + 1.3) + 1)
    return dict(z=z.round(3).tolist(), x0=float(xs[0]), y0=float(y0), dx=dx, dy=dy), (xs, ys, z)


def ground_z(grid, x, y):
    xs, ys, z = grid
    i = np.clip(((y - ys[0]) / (ys[1] - ys[0])).astype(int), 0, len(ys) - 1)
    j = np.clip(((x - xs[0]) / (xs[1] - xs[0])).astype(int), 0, len(xs) - 1)
    return z[i, j]


def trees(rng, grid, half_fov, y0, y1, spacing, scale, edge_boost, centre_gap):
    ys = np.arange(y0, y1, spacing)
    pts = []
    for y in ys:
        xmax = y * math.tan(half_fov) * 1.08
        xs = np.arange(-xmax, xmax, spacing) + rng.uniform(-spacing / 2, spacing / 2)
        yy = y + rng.uniform(-spacing / 2, spacing / 2, len(xs))
        u = np.abs(xs) / xmax
        cluster = .5 + .5 * np.sin(xs * .031 + 1.7) * np.sin(xs * .0113 + yy * .02)
        p = np.clip(.18 + .8 * cluster + edge_boost * u ** 2 - centre_gap * np.clip(1 - u / .45, 0, 1), 0, 1)
        keep = rng.random(len(xs)) < p
        for x, y_, uu in zip(xs[keep], yy[keep], u[keep]):
            s = rng.uniform(*scale) * (1 + edge_boost * .6 * uu ** 2)
            pts.append([float(x), float(y_), float(ground_z(grid, np.array([x]), np.array([y_]))[0] - .4),
                        float(s), float(rng.uniform(0, 2 * math.pi)), int(rng.random() < .35)])
    return pts


def blender_python():
    if importlib.util.find_spec("bpy"):
        return sys.executable
    return os.environ.get("BANNER_BPY_PYTHON")


def render_blender(sc, L, W, H, S, margin, samples):
    py = blender_python()
    if not py or not (ASSETS / "fir_tree_01").exists():
        return None
    rng = np.random.default_rng(42)
    hfov_banner = math.radians(T.HFOV)
    hfov = 2 * math.atan(math.tan(hfov_banner / 2) * (W + 2 * margin) / W)
    half = hfov / 2
    gd, grid = ground_grid(rng, half)
    back = trees(rng, grid, half, 120, 258, 3.6, (.28, .55), 1.4, 0)
    front = trees(rng, grid, half, 38, 118, 7.0, (.45, .8), 1.6, .9)

    direct = float(np.clip(sc.sun_alt / 6, 0, 1)) ** .7 * (1 - L.cloud ** 1.3 * .93) * (0 if sc.weather.fog else 1)
    glow = np.asarray(L.glow, float)
    snow = sc.snowline < .3
    tint = {"spring": (.7, .9, .5), "summer": (.62, .8, .45), "autumn": (.75, .75, .5),
            "winter": (.9, .9, .85)}[sc.season]
    moon_on = sc.moon_alt > 0 and sc.sun_alt < -6
    cfg = dict(
        assets=str(ASSETS), samples=samples, res=[int((W + 2 * margin) * S), int(H * S)],
        exposure=0.0, hfov=math.degrees(hfov), shift_y=(R.Y_HORIZON - H / 2) / (W + 2 * margin), cam_z=CAM_Z,
        ground=gd, ground_tex="snow_02" if snow else "leafy_grass", ground_tex_scale=3.0 if snow else 2.0,
        ground_tint=(1, 1, 1) if snow else tint, back=back, front=front,
        zenith=lin(L.zenith), horizon=lin(L.horizon), ground_bounce=lin(np.asarray(tint) * .12),
        sky_strength=.26,
        sun_strength=5.5 * direct, sun_rgb=lin(glow / max(glow.max(), 1e-3)), sun_alt=max(sc.sun_alt, .5),
        sun_rel_az=sc.sun_az - T.HEADING, sun_angle=.6 + 4 * L.cloud,
        moon_strength=(.04 * sc.moon_frac * (1 - L.cloud * .8)) if moon_on else 0,
        moon_rgb=[.7, .8, 1.0], moon_alt=sc.moon_alt, moon_rel_az=sc.moon_az - T.HEADING, moon_angle=.5,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = os.environ.get("BANNER_FOREST_DIR", tmp)
        Path(tmp).mkdir(parents=True, exist_ok=True)
        cfg["out"] = tmp
        p = Path(tmp) / "cfg.json"
        p.write_text(json.dumps(cfg))
        r = subprocess.run([py, str(HERE / "forest_scene.py"), str(p)], capture_output=True, text=True)
        if r.returncode != 0 or not (Path(tmp) / "front.png").exists():
            print(r.stdout[-2000:], r.stderr[-2000:])
            return None
        out = []
        for name in ("back", "front"):
            a = np.asarray(Image.open(Path(tmp) / f"{name}.png")).astype(np.float32)
            out.append(a / (65535 if a.max() > 255 else 255))
    return out


def forest(sc, L, W, H, S, margin, samples=48):
    layers = render_blender(sc, L, W, H, S, margin, samples)
    if layers is None:
        print("blender forest unavailable, using procedural trees")
        return procedural.foreground(sc, L, W, H, S, margin)
    haze = np.asarray(L.horizon) * .6 + np.asarray(L.zenith) * .2
    rows = []
    for arr, k, shift in zip(layers, (.12 + .25 * L.gloom, .03 + .1 * L.gloom), (22, 36)):
        rgb = arr[..., :3] * (1 - k) + haze * k
        rows.append((np.concatenate([np.clip(rgb, 0, 1), arr[..., 3:]], -1), shift))
    a = layers[0][..., 3]
    top = np.argmax(a > .5, axis=0).astype(np.float32)
    top[a.max(0) < .5] = H * S
    return rows, top / S
