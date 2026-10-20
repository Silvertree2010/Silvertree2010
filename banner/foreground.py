import math, random

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

SS = 3


def _bezier(p0, p1, p2, n=7):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (k / (n - 1) for k in range(n))]


def _tree(dc, da, x, yb, h, rnd, col, snow, larch=False):
    trunk_w = max(1, int(h * .022))
    dc.line([(x, yb), (x, yb - h)], fill=tuple(int(v * .6) for v in col), width=trunk_w)
    da.line([(x, yb), (x, yb - h)], fill=255, width=trunk_w)
    n = max(10, min(90, int(h / (2.6 if larch else 1.8))))
    for i in range(n):
        t = (i + rnd.random()) / n
        y0 = yb - h + t * h * .96
        for side in (-1, 1):
            if larch and rnd.random() < .35:
                continue
            L = h * (.03 + .21 * t ** .9) * rnd.uniform(.65, 1.3)
            droop = L * (.18 + .45 * t) * rnd.uniform(.6, 1.3)
            p0 = (x, y0)
            p1 = (x + side * L * .55, y0 - L * .06)
            p2 = (x + side * L, y0 + droop)
            pts = _bezier(p0, p1, p2)
            c = tuple(int(max(0, min(255, v * rnd.uniform(.8, 1.2)))) for v in col)
            w = max(1, int(h * .010 * (1.2 - .6 * t)))
            dc.line(pts, fill=c, width=w)
            da.line(pts, fill=255, width=w)
            for k in range(1, len(pts)):
                px, py = pts[k]
                for _ in range(2 if larch else 3):
                    nl = h * rnd.uniform(.015, .045)
                    ang = math.radians(rnd.uniform(55, 125))
                    ex, ey = px + side * nl * .4 * rnd.random(), py + nl * math.sin(ang)
                    dc.line([(px, py), (ex + nl * math.cos(ang) * .5, ey)], fill=c, width=1)
                    da.line([(px, py), (ex + nl * math.cos(ang) * .5, ey)], fill=255, width=1)
            if snow and rnd.random() < .4:
                sp = [(px, py - w * .6) for px, py in pts[1:-2]]
                dc.line(sp, fill=(196, 204, 216), width=max(1, w - 1))
                da.line(sp, fill=255, width=max(1, w))
    dc.line([(x, yb - h), (x, yb - h * 1.04)], fill=col, width=max(1, trunk_w // 2))
    da.line([(x, yb - h), (x, yb - h * 1.04)], fill=255, width=max(1, trunk_w // 2))


def _ground(w, base, amp, rng):
    x = np.arange(w)
    g = np.zeros(w)
    for f, a in ((2, 1), (5, .5), (13, .25), (37, .1)):
        g += a * np.sin(x / w * math.pi * f + rng.uniform(0, 6.28))
    return base - amp * g / 1.85


def foreground(sc, L, W, H, S, margin):
    w, h = int((W + 2 * margin) * S), int(H * S)
    rnd = random.Random(42)
    rng = np.random.default_rng(42)
    season = sc.season
    snow = sc.snowline < .3
    larch_gold = season == "autumn" and sc.now.month >= 10

    lum = float(L.amb @ np.array([.3, .59, .11], np.float32))
    sky_tint = L.zenith / (float(L.zenith.max()) + 1e-3)
    amb = (sky_tint * .5 + .5) * lum
    direct = float(np.clip(sc.sun_alt / 6, 0, 1)) * (1 - L.cloud ** 1.3 * .9) * (0 if sc.weather.fog else 1)

    rows = []
    specs = [
        (283, 7, (11, 26), 760, .30, .3, 22),
        (301, 4, (7, 78), 260, .03, 0, 36),
    ]
    ground_line = None
    for gbase, gamp, (hmin, hmax), count, haze_k, blur, shift in specs:
        Wc, Hc = w * SS, h * SS
        rgb = Image.new("RGB", (Wc, Hc), (0, 0, 0))
        a = Image.new("L", (Wc, Hc), 0)
        dc, da = ImageDraw.Draw(rgb), ImageDraw.Draw(a)
        g = _ground(w, gbase * S, gamp * S, rng)
        gc = (34, 42, 26) if season != "winter" else (210, 216, 224)
        if season == "autumn":
            gc = (46, 44, 26)
        poly = [(0, Hc)] + [(x * SS, g[x] * SS) for x in range(w)] + [(Wc, Hc)]
        dc.polygon(poly, fill=gc); da.polygon(poly, fill=255)
        for x in range(0, w, 1):
            for _ in range(2):
                gx = (x + rnd.random()) * SS
                gy = g[x] * SS + 2
                gl = rnd.uniform(1.5, 4.5) * S * SS
                ang = math.radians(rnd.uniform(-25, 25))
                c = tuple(int(v * rnd.uniform(.8, 1.3)) for v in gc)
                dc.line([(gx, gy), (gx + gl * math.sin(ang), gy - gl * math.cos(ang))], fill=c, width=1)
                da.line([(gx, gy), (gx + gl * math.sin(ang), gy - gl * math.cos(ang))], fill=255, width=1)
        xs = sorted(rnd.uniform(0, w) for _ in range(count))
        for x in xs:
            u = x / w
            edge = max(0.0, abs(u - .5) * 2 - .25) / .75
            cluster = .5 + .5 * math.sin(u * 23 + 1.7) * math.sin(u * 7.3)
            if rnd.random() > .5 + .5 * max(edge, cluster * .7):
                continue
            th = (hmin + (hmax - hmin) * min(1, edge ** 1.6 * .8 + cluster * .12 + rnd.random() * .25)) * S * SS
            yb = g[min(w - 1, int(x))] * SS + rnd.uniform(1, 4) * SS
            is_larch = larch_gold and rnd.random() < .22
            if is_larch:
                col = (150, 110, 40)
            else:
                col = tuple(int(v * rnd.uniform(.8, 1.2)) for v in (17, 26, 20))
            _tree(dc, da, x * SS, yb, th, rnd, col, snow and not is_larch, larch=is_larch)

        rgb = rgb.resize((w, h), Image.LANCZOS)
        a = a.resize((w, h), Image.LANCZOS)
        alb = np.asarray(rgb, np.float32) / 255
        al = np.asarray(a, np.float32) / 255
        if blur:
            al = gaussian_filter(al, blur * .6)
            alb = np.stack([gaussian_filter(alb[..., i], blur * .6) for i in range(3)], -1)

        light = amb[None, None] * 1.0 + L.glow * direct * .55
        col = alb * light * 1.5
        occ = gaussian_filter(al, 2.5 * S)
        col = col * (1 - .5 * np.clip((occ - .55) / .45, 0, 1))[..., None]
        yy = np.arange(h, dtype=np.float32)[:, None]
        col = col * (.7 + .3 * np.clip((g[None, :] - yy) / (25 * S), 0, 1))[..., None]
        sx_dir = 1 if (sc.sun_az - 140 + 540) % 360 - 180 > 0 else -1
        shifted = np.roll(al, -sx_dir * max(1, int(1.2 * S)), 1)
        rim = np.clip(al - shifted, 0, 1)
        rim_k = float(np.clip(1 - abs(sc.sun_alt - 4) / 14, 0, 1)) * (1 - L.cloud * .8) * L.glow_k
        col = col + L.glow * (rim * rim_k * .9)[..., None]
        haze = L.horizon * .6 + L.zenith * .2
        col = col * (1 - haze_k) + haze * haze_k * (.6 + .4 * lum)
        rows.append((np.concatenate([np.clip(col, 0, 1), al[..., None]], -1), shift))
        ground_line = g / S
    return rows, ground_line
