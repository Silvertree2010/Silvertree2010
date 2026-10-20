"""Profile banner: an alpine panorama that follows the real time of day, sun and moon position,
local weather and season.

    python banner/generate.py --out dist                 # live
    python banner/generate.py --out dist --at 2026-10-20T18:30 --code 3 --cloud .4
"""
import argparse, base64, io, math, random
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import fx
import render as R
import terrain as T
from state import TZ, Weather, build_scene, fetch_weather

W, H, S = R.W, R.H, R.S
CAM = 40.0                       # camera move duration


def webp(arr, q=80):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=q, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def hexc(c):
    c = np.clip(c, 0, 1)
    return "#" + "".join(f"{int(v * 255):02x}" for v in c)


def image(uri, x, y, w, h, extra=""):
    return f'<image x="{fx.f(x)}" y="{fx.f(y)}" width="{fx.f(w)}" height="{fx.f(h)}" preserveAspectRatio="none" href="{uri}"{extra}/>'


def drifting(defs, body, key, arr, q, speed, sign):
    """Double-width image (tile x2) → seamless loop without an edge between two copies."""
    wide = np.concatenate([arr, arr], 1)
    x0 = 0 if sign < 0 else -W
    body.append(f'<g><image x="{x0}" y="-2" width="{2 * W}" height="{H + 4}" preserveAspectRatio="none" href="{webp(wide, q)}"/>'
                f'<animateTransform attributeName="transform" type="translate" from="0 0" to="{sign * W} 0" '
                f'dur="{W / speed:.0f}s" repeatCount="indefinite"/></g>')


def build(sc):
    L = R.Light(sc)
    w_ = sc.weather
    rnd = random.Random(sc.seed)
    kind, inten = w_.precip
    drift = fx.wind_vec(w_)
    defs, body = [], []

    # Sky
    body.append(image(webp(R.sky_layer(sc, L), 82), 0, 0, W, H))
    if L.night > .4 and L.cloud < .5 and not w_.fog:
        body.append(fx.meteors(rnd))
    rays = R.rays_layer(sc, L)
    if rays:
        arr, M = rays
        sx, sy = R.to_screen(sc.sun_alt, sc.sun_az)
        body.append(f'<g>{image(webp(arr, 70), -M, -M, W + 2 * M, H + 2 * M)}'
                    f'<animate attributeName="opacity" values="1;.55;1" dur="{CAM / 2}s" {fx.EASE} repeatCount="indefinite"/>'
                    f'<animateTransform attributeName="transform" type="rotate" values="-1.2 {sx:.0f} {sy:.0f};1.2 {sx:.0f} {sy:.0f};-1.2 {sx:.0f} {sy:.0f}" '
                    f'dur="{CAM}s" {fx.EASE} repeatCount="indefinite"/></g>')

    # Clouds drift endlessly with the wind
    cl = R.cloud_layer(sc, L)
    if cl is not None:
        speed = .25 + abs(drift) * 4                 # px/s
        sign = -1 if drift < 0 else 1
        drifting(defs, body, 'cl', cl, 78, speed, sign)

    # Real terrain (far → near) + a procedural tree line in front, with parallax
    margin = 40
    cam = T.camera(int((W + 2 * margin) * S), int(H * S), R.Y_HORIZON * S, view_w=W * S)
    terr = T.render_terrain(sc, L, cam)
    layers, ridges = [], []
    for arr, shift in zip(terr, (3, 8, 16)):
        layers.append((arr, -margin, W + 2 * margin, shift))
        top = np.argmax(arr[..., 3] > .5, axis=0).astype(np.float32)
        top[arr[..., 3].max(0) < .5] = H * S
        ridges.append((top / S, margin))
    # no artificial foreground: near-camera wildlife (bats, fireflies) hovers along the bottom edge
    ridges.append((np.full(int((W + 2 * margin) * S), H + 6, np.float32), margin))

    def ridge_y(x, layer=3):
        arr, off = ridges[layer]
        return float(arr[max(0, min(len(arr) - 1, int((x + off) * S)))])

    arr0, off0 = ridges[0]
    xs = np.arange(len(arr0)) / S - off0
    ok = (xs > 150) & (xs < 1130)
    pi = int(np.argmin(np.where(ok, arr0, 1e9)))
    peak = (float(xs[pi]), float(arr0[pi]))

    day = sc.sun_alt > -2
    dusk = -10 < sc.sun_alt <= 3
    night = sc.sun_alt <= -8
    calm = inten < .55 and not w_.thunder
    bird_col = hexc(L.amb * .18 + np.array([.02, .02, .025]))
    fauna = {i: [] for i in range(4)}
    if calm and not w_.fog:
        if sc.season == "winter" and day:
            fauna[1].append(fx.choughs(rnd, bird_col, peak))
        if sc.season == "spring" and sc.sun_alt > -4:
            fauna[3].append(fx.cranes(rnd, bird_col))
        if sc.season == "summer" and sc.sun_alt > 2:
            fauna[3].append(fx.swifts(rnd, bird_col))
        if sc.season == "autumn" and sc.sun_alt > -6:
            fauna[3].append(fx.murmuration(rnd, bird_col))
    fore = []
    if sc.season == "summer" and calm:
        if dusk and sc.sun_alt < 1:
            fore.append(fx.bats(rnd, "#08090b", ridge_y))
        if night and w_.temp > 10:
            fore.append(fx.fireflies(rnd, ridge_y))
    if sc.season == "autumn" and (sc.now.month >= 10 or w_.wind > 15) and calm:
        fore.append(fx.leaves(rnd, drift if abs(drift) > .2 else -.6))

    fog = R.fog_layer(sc, L)
    low = R.low_cloud_layer(sc, L)
    for i, (arr, x0, width, shift) in enumerate(layers):
        body += fauna[i]
        if i == 2 and low is not None:
            sign = -1 if drift < 0 else 1
            drifting(defs, body, 'lc', low, 74, .5 + abs(drift) * 5, sign)
        body.append(f'<g>{image(webp(arr, 84), x0, 0, width, H)}'
                    f'<animateTransform attributeName="transform" type="translate" values="0 0;-{shift} 0;0 0" '
                    f'dur="{CAM}s" {fx.EASE} repeatCount="indefinite"/></g>')
    if fog is not None:
        sign = -1 if drift < 0 else 1
        drifting(defs, body, 'fg', fog, 72, .6 + abs(drift) * 3, sign)
    body += fauna[3]
    body += fore

    # Precipitation + thunderstorms
    if kind == "rain":
        body.append(fx.rain(rnd, inten, drift, hexc(L.horizon * .6 + .35)))
    elif kind == "snow":
        body.append(fx.snow(rnd, inten, drift, hexc(np.maximum(L.amb, .35) * .95 + .05)))
    if w_.thunder:
        body.append(fx.lightning(rnd, lambda x: ridge_y(x, 0)))

    # Camera: grain + vignette
    grain = .04 + .05 * L.night
    defs.append('<filter id="gr" x="0" y="0" width="100%" height="100%"><feTurbulence type="fractalNoise" baseFrequency=".85" '
                'numOctaves="2" seed="4" stitchTiles="stitch"/><feColorMatrix values="0 0 0 0 .5 0 0 0 0 .5 0 0 0 0 .5 0 0 0 1.5 -.5"/></filter>')
    body.append(f'<rect width="{W}" height="{H}" filter="url(#gr)" opacity="{grain:.3f}"/>')
    defs.append('<radialGradient id="vg" cx="50%" cy="42%" r="72%"><stop offset=".5" stop-color="#000" stop-opacity="0"/>'
                f'<stop offset="1" stop-color="#000" stop-opacity="{.35 + .35 * L.night:.2f}"/></radialGradient>')
    body.append(f'<rect width="{W}" height="{H}" fill="url(#vg)"/>')
    # mask instead of clip-path: the group is composited first, then masked once,
    # so light layers underneath can't bleed through the anti-aliased corners
    defs.append(f'<mask id="k" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">'
                f'<rect width="{W}" height="{H}" rx="14" fill="#fff"/></mask>')

    title = f"{sc.now:%Y-%m-%d %H:%M} · {w_.label} · {sc.season}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'viewBox="0 0 {W} {H}" width="{W}" height="{H}"><title>{title}</title>'
           f'<defs>{"".join(defs)}</defs><g mask="url(#k)">' + "\n".join(body) + "</g></svg>")
    return svg, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dist")
    ap.add_argument("--name", default="banner.svg")
    ap.add_argument("--at", help="moment to render (ISO, local time) instead of now")
    ap.add_argument("--code", type=int, help="force a WMO weather code")
    ap.add_argument("--cloud", type=float)
    ap.add_argument("--temp", type=float)
    ap.add_argument("--wind", type=float)
    ap.add_argument("--wdir", type=float)
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()

    now = datetime.fromisoformat(a.at).replace(tzinfo=TZ) if a.at else datetime.now(TZ)
    w = Weather() if (a.offline or a.code is not None) else fetch_weather()
    for k, v in (("code", a.code), ("cloud", a.cloud), ("temp", a.temp), ("wind", a.wind), ("wind_dir", a.wdir)):
        if v is not None:
            setattr(w, k, v)
    sc = build_scene(now, w)
    svg, title = build(sc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / a.name).write_text(svg)
    print(f"{title}  →  {out / a.name} ({len(svg) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
