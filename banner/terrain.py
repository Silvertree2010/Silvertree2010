"""Real terrain: heightfield ray casting over a DEM of the Bernese Oberland, lit by the actual
sun position (with cast shadows), seasonal snow line, glaciers, forest, lakes and aerial
perspective."""
import json, math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, rotate

DATA = Path(__file__).resolve().parent / "data"
R_EARTH = 6_371_000
REFRACTION = 0.13

# Camera: aerial viewpoint above the lakes, looking south-east to the big peaks
CAM_LAT, CAM_LON, CAM_ALT = 46.690, 7.800, 1900.0
HEADING = 140.0          # degrees, 180 = south
HFOV = 68.0
D_MIN, D_MAX, STEPS = 40.0, 70_000.0, 1800


def _pix(lat, lon, meta):
    n = 2 ** meta["z"]
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x * 256 - meta["px0"], y * 256 - meta["py0"]


@lru_cache(1)
def dem():
    meta = json.loads((DATA / "dem.json").read_text())
    h = np.asarray(Image.open(DATA / "dem.png"), np.float32) / meta["scale"]
    px = 40_075_016.686 * math.cos(math.radians(CAM_LAT)) / (256 * 2 ** meta["z"])
    # sub-pixel detail so near slopes don't look faceted
    rng = np.random.default_rng(3)
    det = gaussian_filter(rng.standard_normal(h.shape).astype(np.float32), 1.0)
    rid = np.zeros_like(h)
    for sig, amp in ((1.0, 5), (2.2, 9), (4.5, 14)):     # ridged multi-scale relief
        n_ = gaussian_filter(rng.standard_normal(h.shape).astype(np.float32), sig)
        rid += (1 - np.abs(n_ / n_.std())) * amp
    gy, gx = np.gradient(h)
    steep = np.clip(np.hypot(gx, gy) / 18, 0, 1)          # rugged only where it's steep
    h = h + det * 3 + rid * steep
    return h, px, meta


def camera(W, Hh, horizon_y, view_w=None):
    """view_w: width that spans HFOV (defaults to W); a wider W renders extra margin."""
    h, px, meta = dem()
    cx, cy = _pix(CAM_LAT, CAM_LON, meta)
    f = ((view_w or W) / 2) / math.tan(math.radians(HFOV / 2))
    return dict(cx=cx, cy=cy, cz=CAM_ALT, f=f, W=W, H=Hh, yh=horizon_y, px=px)


def project(cam, alt, az):
    """Sky direction → screen (render pixels). Far outside the view → off-screen."""
    d = (az - HEADING + 540) % 360 - 180
    if abs(d) > 85:
        return (-1e4 if d < 0 else 1e4), cam["yh"]
    x = cam["W"] / 2 + cam["f"] * math.tan(math.radians(d))
    y = cam["yh"] - cam["f"] * math.tan(math.radians(alt)) / math.cos(math.radians(d))
    return x, y


# --------------------------------------------------------------------------- grid maps
def normals(h, px):
    gy, gx = np.gradient(h, px)
    n = np.stack([-gx, -gy, np.ones_like(h)], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def sun_vec(alt, az):
    a, e = math.radians(az), math.radians(alt)
    return np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)], np.float32)


def shadows(h, px, alt, az):
    """Cast shadows along the sun direction via a rotated running-envelope scan."""
    if alt <= 0.3:
        return np.zeros_like(h)
    # rotate so the sun sits to the left (−x), then scan left → right
    to_sun = math.degrees(math.atan2(-math.cos(math.radians(az)), math.sin(math.radians(az))))
    ang = to_sun - 180.0
    hr = rotate(h, ang, reshape=True, order=1, mode="constant", cval=-1e4)
    valid = rotate(np.ones_like(h), ang, reshape=True, order=0, mode="constant", cval=0) > .5
    drop = px * math.tan(math.radians(alt))
    env = np.empty_like(hr)
    env[:, 0] = hr[:, 0]
    for j in range(1, hr.shape[1]):
        env[:, j] = np.maximum(hr[:, j], env[:, j - 1] - drop)
    lit_r = ((env - hr) < 2.0).astype(np.float32)
    lit_r[~valid] = 1
    back = rotate(lit_r, -ang, reshape=True, order=1, mode="nearest")
    # crop back to the original grid (rotate with reshape grows the canvas)
    oy = (back.shape[0] - h.shape[0]) // 2
    ox = (back.shape[1] - h.shape[1]) // 2
    lit = back[oy:oy + h.shape[0], ox:ox + h.shape[1]]
    return gaussian_filter(lit, .8)


def albedo_maps(h, n, sc):
    """Snow, ice, rock, meadow, forest, water — all in one RGB albedo grid."""
    rng = np.random.default_rng(11)
    slope = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    big = gaussian_filter(rng.random(h.shape, dtype=np.float32), 6) - .5
    small = gaussian_filter(rng.random(h.shape, dtype=np.float32), 1.5) - .5
    nz = (big * 4 + small * 2).astype(np.float32)

    season = sc.season
    meadow = {"spring": (.25, .31, .16), "summer": (.21, .26, .14),
              "autumn": (.33, .31, .19), "winter": (.32, .31, .26)}[season]
    forest = (.07, .12, .07)
    rock = np.array((.25, .235, .22), np.float32)
    alb = np.empty(h.shape + (3,), np.float32)
    alb[:] = rock * (1 + nz[..., None] * .45)

    # meadows: gentle slopes below the rock band
    m_mask = np.clip((2350 + nz * 250 - h) / 250, 0, 1) * np.clip((40 - slope) / 10, 0, 1)
    alb = alb * (1 - m_mask[..., None]) + np.array(meadow, np.float32) * (1 + nz[..., None] * .4) * m_mask[..., None]
    # forest
    f_mask = (np.clip((1850 + nz * 200 - h) / 150, 0, 1) * np.clip((44 - slope) / 8, 0, 1)
              * np.clip((big * 2 + small * 1.5 + .62) * 2.5, 0, 1))
    fcol = np.array(forest, np.float32)
    alb = alb * (1 - f_mask[..., None]) + fcol * (1 + small[..., None] * 1.2) * f_mask[..., None]
    if season == "autumn" and sc.now.month >= 10:   # larches turn gold near the tree line
        larch = f_mask * np.clip((h - 1450) / 250, 0, 1) * np.clip(small * 8 + .2, 0, 1)
        alb = alb * (1 - larch[..., None]) + np.array((.55, .40, .12), np.float32) * larch[..., None]

    # snow: seasonal line in metres, steep faces hold none
    line = 600 + sc.snowline * 2600
    s_mask = np.clip((h - line + nz * 220) / 160, 0, 1) * np.clip((52 - slope) / 12, 0, 1)
    ice = np.clip((h - 2850 + nz * 200) / 120, 0, 1) * np.clip((32 - slope) / 8, 0, 1)
    snow = np.maximum(s_mask, ice)
    scol = np.array((.92, .94, .97), np.float32)
    alb = alb * (1 - snow[..., None]) + scol * (1 + small[..., None] * .08) * snow[..., None]

    fine = gaussian_filter(rng.random(h.shape, dtype=np.float32), .6) - .5
    alb *= (1 + fine * .5)[..., None]

    water = ((h < 570) & (slope < 2.5)).astype(np.float32)
    water = gaussian_filter(water, .7)
    return alb, snow, water


@lru_cache(1)
def micro():
    rng = np.random.default_rng(17)
    t = gaussian_filter(rng.random((512, 512), dtype=np.float32), .9, mode="wrap")
    t = (t - t.min()) / (t.max() - t.min())
    return t


# --------------------------------------------------------------------------- ray casting
def cast(cam, d_from=D_MIN, d_to=D_MAX):
    h, px, _ = dem()
    W, Hh, f = cam["W"], cam["H"], cam["f"]
    cols = np.arange(W, dtype=np.float32) + .5
    delta = np.arctan((cols - W / 2) / f)
    az = np.radians(HEADING) + delta
    dx, dy = np.sin(az), -np.cos(az)
    cosd = np.cos(delta)
    r = (d_to / d_from) ** (1 / STEPS)
    ds = d_from * r ** np.arange(STEPS, dtype=np.float64)

    M = np.full(W, Hh + 1.0, np.float32)
    Ms = np.empty((STEPS, W), np.float32)
    X = np.empty((STEPS, W), np.float32)
    Y = np.empty((STEPS, W), np.float32)
    for k, d in enumerate(ds):
        gx = cam["cx"] + dx * d / px
        gy = cam["cy"] + dy * d / px
        hk = map_coordinates(h, [gy, gx], order=1, mode="constant", cval=-1e4)
        heff = hk - d * d / (2 * R_EARTH) * (1 - REFRACTION)
        yp = cam["yh"] - f * (heff - cam["cz"]) / (d * cosd)
        M = np.minimum(M, yp)
        Ms[k], X[k], Y[k] = M, gx, gy

    # for each pixel row: first step whose envelope reaches that row
    rows = np.arange(Hh, dtype=np.float32) + .5
    K = np.empty((Hh, W), np.int32)
    for c in range(W):
        K[:, c] = np.searchsorted(-Ms[:, c], -rows, side="left")
    hit = K < STEPS
    Kc = np.minimum(K, STEPS - 1)
    cidx = np.broadcast_to(np.arange(W), (Hh, W))
    gx, gy = X[Kc, cidx], Y[Kc, cidx]
    dist = ds[Kc].astype(np.float32)
    return hit, gx, gy, dist, delta


def render_terrain(sc, L, cam, layers=((25_000, D_MAX), (6_000, 25_000), (D_MIN, 6_000))):
    """→ list of RGBA arrays (far → near) for parallax."""
    h, px, _ = dem()
    n = normals(h, px)
    alb, snow, water = albedo_maps(h, n, sc)
    sun_up = sc.sun_alt > 0
    lit = shadows(h, px, sc.sun_alt, sc.sun_az) if sun_up else np.zeros_like(h)
    sv = sun_vec(sc.sun_alt, sc.sun_az)
    lam = np.clip(n @ sv, 0, 1) * lit
    ao = 1 - np.clip((gaussian_filter(h, 8) - h) / 260, 0, .55)
    sky_vis = (.55 + .45 * n[..., 2]) * ao

    # light colours from the shared Light model
    w = sc.weather
    direct_k = (1 - L.cloud ** 1.3 * .9) * (0 if w.fog else 1) * float(np.clip(sc.sun_alt / 4, 0, 1))
    sun_col = np.clip(L.glow * 1.3, 0, 1.4) * direct_k * 1.6
    lum = float(L.amb @ np.array([.3, .59, .11], np.float32))
    sky_tint = L.zenith / (float(L.zenith.max()) + 1e-3)
    amb_col = (sky_tint * .65 + .35) * lum * (.55 + .25 * (1 - direct_k))
    if sc.moon_alt > 0 and sc.sun_alt < -6:
        mv = sun_vec(sc.moon_alt, sc.moon_az)
        moon_l = np.clip(n @ mv, 0, 1) * sc.moon_frac * .22 * (1 - L.cloud * .8)
    else:
        moon_l = 0

    shade = (lam[..., None] * sun_col + sky_vis[..., None] * amb_col
             + (moon_l[..., None] * np.array([.55, .65, .9], np.float32) if np.ndim(moon_l) else 0))
    col_grid = alb * shade
    # alpenglow on snow just before sunrise / after sunset
    ag = float(np.exp(-((sc.sun_alt - 1.0) / 3.5) ** 2)) * (1 - L.cloud * .85)
    if ag > .01:
        col_grid += (snow * ag * .4 * (.4 + .6 * np.clip((h - 2500) / 1200, 0, 1)))[..., None] * np.array([1., .5, .55], np.float32)

    out = []
    for d0, d1 in layers:
        hit, gx, gy, dist, delta = cast(cam, d0, d1)
        coords = [gy.ravel(), gx.ravel()]
        rgb = np.stack([map_coordinates(col_grid[..., i], coords, order=1, mode="nearest").reshape(hit.shape)
                        for i in range(3)], -1)
        wat = map_coordinates(water, coords, order=1, mode="nearest").reshape(hit.shape)
        tex = map_coordinates(micro(), [(gy.ravel() * 7) % 512, (gx.ravel() * 7) % 512], order=1,
                              mode="wrap").reshape(hit.shape)
        near = np.clip(1.15 - dist / 18000, 0, 1)
        rgb = rgb * (1 + (tex[..., None] - .5) * .9 * near[..., None])
        hh = map_coordinates(h, coords, order=1, mode="nearest").reshape(hit.shape)

        # lakes reflect the sky near the horizon
        refl = (L.horizon * .75 + L.zenith * .15) * (.7 + .3 * direct_k)
        rgb = rgb * (1 - wat[..., None] * .85) + refl * wat[..., None] * .85

        # aerial perspective: distance + height-dependent haze, sun-side forward scattering
        kind, inten = w.precip
        vis = 38_000 * (1 - L.gloom * .75)
        if w.fog:
            vis = 4_000
        dens = 1 + 1.6 * np.exp(-(hh - 500) / 1200)
        T = np.exp(-dist * dens / vis)[..., None]
        view_az = HEADING + np.degrees(delta)
        dd = np.radians((view_az - sc.sun_az + 540) % 360 - 180)
        fwd = np.exp(-np.abs(dd) / .55) * L.glow_k * float(np.clip(1 - sc.sun_alt / 35, 0, 1))
        away = np.clip(np.abs(dd) / 1.6, 0, 1)[None, :, None]
        base_h = L.horizon * (1 - away * .7) + (L.zenith * .75 + L.horizon * .25) * away * .7
        haze = base_h * (1 - L.gloom * .25) + L.glow * fwd[None, :, None] * .6
        rgb = rgb * T + haze * (1 - T)

        a = hit.astype(np.float32)
        out.append(np.concatenate([np.clip(rgb, 0, 1), a[..., None]], -1))
    return out
