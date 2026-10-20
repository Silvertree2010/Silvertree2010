import math, random
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

W, H = 1280, 300
S = 1.25
Y_HORIZON = 205
TAU = 2 * math.pi

def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255

def lerp(a, b, t):
    return a + (b - a) * t

def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)

PALETTE = [
    (-90, "#010208", "#03060d", "#0b1428", .00, "#5a6fa0", .035),
    (-18, "#02040b", "#060b18", "#142040", .05, "#5a6fa0", .04),
    (-12, "#040915", "#0d1830", "#233560", .15, "#6f84b8", .07),
    (-8,  "#071025", "#1c2a4e", "#6a4a6a", .25, "#7f8fc0", .11),
    (-4,  "#0d1a38", "#4a4a72", "#d0705a", .45, "#a08cb0", .18),
    (-1,  "#16295a", "#b06a5a", "#ff8c50", .75, "#e0a080", .28),
    (2,   "#24447e", "#e89a62", "#ffa860", .90, "#ffc090", .45),
    (6,   "#3460a0", "#f0bc88", "#ffc888", .70, "#ffd8b0", .62),
    (12,  "#3d72b8", "#d8d4c4", "#ffe4c0", .50, "#fff0dc", .80),
    (25,  "#3a74c0", "#b4cfe4", "#fff4e4", .40, "#ffffff", .92),
    (60,  "#2c68bc", "#a2c4e2", "#ffffff", .35, "#ffffff", 1.0),
]


class Light:

    def __init__(self, sc):
        a = max(-90, min(60, sc.sun_alt))
        for lo, hi in zip(PALETTE, PALETTE[1:]):
            if lo[0] <= a <= hi[0]:
                t = float(smooth((a - lo[0]) / (hi[0] - lo[0])))
                break
        self.zenith = lerp(rgb(lo[1]), rgb(hi[1]), t)
        self.horizon = lerp(rgb(lo[2]), rgb(hi[2]), t)
        self.glow = lerp(rgb(lo[3]), rgb(hi[3]), t)
        self.glow_k = lo[4] + (hi[4] - lo[4]) * t
        self.amb = lerp(rgb(lo[5]), rgb(hi[5]), t) * (lo[6] + (hi[6] - lo[6]) * t)

        w = sc.weather
        kind, inten = w.precip
        cloud = w.cloud
        gloom = cloud * .55 + (inten * .35 if kind else 0) + (.25 if w.thunder else 0) + (.2 if w.fog else 0)
        gloom = min(gloom, .92)
        self.cloud, self.gloom = cloud, gloom

        def grey(c, k=1.0):
            l = float(c @ np.array([.3, .59, .11], np.float32))
            return np.array([l, l, l * 1.03], np.float32) * k
        self.zenith = lerp(self.zenith, grey(self.zenith, .9), gloom * .9)
        self.horizon = lerp(self.horizon, grey(self.horizon, .85), gloom)
        self.glow_k *= (1 - cloud * .8)
        self.amb = lerp(self.amb, grey(self.amb), gloom * .8) * (1 - gloom * .45)

        if sc.moon_alt > 0 and sc.sun_alt < -4:
            mk = sc.moon_frac * min(1, sc.moon_alt / 20) * (1 - cloud * .8) * .16
            self.amb = self.amb + rgb("#8ea6d8") * mk
        self.night = float(np.clip((-sc.sun_alt - 4) / 10, 0, 1))
        self.haze = lerp(self.horizon, self.zenith, .25)


def to_screen(alt, az):
    import terrain
    cam = terrain.camera(int(W * S), int(H * S), Y_HORIZON * S)
    x, y = terrain.project(cam, alt, az)
    return x / S, y / S


def vnoise2(h, w, fh, fw, rng):
    g = rng.random((fh + 1, fw + 1)).astype(np.float32)
    return np.asarray(Image.fromarray(g, mode="F").resize((w, h), Image.BICUBIC))

def fbm2(h, w, fh, fw, rng, oct=5, gain=.5):
    out = np.zeros((h, w), np.float32); a, tot = 1, 0
    for o in range(oct):
        out += a * vnoise2(h, w, max(1, fh * 2 ** o), max(1, fw * 2 ** o), rng); tot += a; a *= gain
    return out / tot

def tileable_x(h, w, fh, fw, rng, oct=5):
    n = fbm2(h, w * 2, fh, fw * 2, rng, oct)
    t = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return n[:, w:] * (1 - t) + n[:, :w] * t

def noise1(n, f, rng):
    g = rng.random(f + 1)
    xs = np.linspace(0, f, n)
    i = np.floor(xs).astype(int).clip(0, f - 1); t = xs - i
    t = t * t * (3 - 2 * t)
    return g[i] * (1 - t) + g[i + 1] * t

def ridge_line(n, rng, rough):
    y = np.zeros(n); a, tot = 1, 0
    for o, f in enumerate([3, 7, 15, 31, 63, 127]):
        r = 1 - np.abs(2 * noise1(n, f, rng) - 1)
        y += a * r ** 1.6; tot += a; a *= .5 if o < 2 else .5 * rough
    return y / tot

def shift_down(a, n):
    return np.concatenate([np.repeat(a[:1], n, 0), a[:-n]], 0)


def blur(a, r):
    if r <= 0:
        return a
    return gaussian_filter(a.astype(np.float32), r)


def sky_layer(sc, L):
    w, h = int(W * S), int(H * S)
    rng = np.random.default_rng(sc.seed)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) / S, np.arange(h, dtype=np.float32) / S)
    t = np.clip(Y / Y_HORIZON, 0, 1) ** .85
    sky = lerp(L.zenith[None, None], L.horizon[None, None], t[..., None])

    sx, sy = to_screen(sc.sun_alt, sc.sun_az)
    dx, dy = X - sx, Y - sy
    g = (np.exp(-(dx / 620) ** 2) * np.exp(-((Y - Y_HORIZON) / 90) ** 2) * .9
         + np.exp(-(dx / 380) ** 2 - (dy / 140) ** 2) * .7
         + np.exp(-np.hypot(dx, dy * 1.4) / 45) * .8 * (1 - min(.65, max(0, sc.sun_alt) / 40)))
    sky = sky + L.glow * (g * L.glow_k)[..., None] * (1 - sky)

    if L.night > 0:
        lp = np.exp(-((Y - Y_HORIZON) / 60) ** 2) * np.exp(-((X - 380) / 520) ** 2)
        sky = sky + rgb("#c08050") * (lp * .05 * L.night * (1 + L.cloud * 2))[..., None]

    vis = L.night * (1 - L.cloud) ** 1.5 * (1 - (.6 if sc.weather.fog else 0))
    if sc.moon_alt > 0:
        vis_mw = vis * (1 - sc.moon_frac * .8)
    else:
        vis_mw = vis
    if vis > .02:
        srng = np.random.default_rng(1234)
        n = 420
        px = srng.random(n) * w
        py = (srng.random(n) ** 1.25) * Y_HORIZON * S
        mag = srng.random(n) ** 3
        star = np.zeros((h, w), np.float32)
        for x0, y0, m in zip(px, py, mag):
            star[int(y0), int(x0)] += .25 + 1.6 * m
        star = blur(star, .55) * 2.2
        fade = np.clip((Y_HORIZON - Y) / 60, 0, 1)
        sky = sky + (star * fade * vis)[..., None] * rgb("#e8eeff")
        u = (X * .35 + Y * .9 - 420) / 120
        band = np.exp(-u ** 2) * (fbm2(h, w, 6, 20, srng, 5) ** 2)
        sky = sky + (band * .09 * vis_mw * fade)[..., None] * rgb("#b9c3e6")

    if sc.moon_alt > -3:
        mx, my = to_screen(sc.moon_alt, sc.moon_az)
        r = 8.5
        ddx, ddy = (X - mx) / r, (Y - my) / r
        d2 = ddx ** 2 + ddy ** 2
        inside = np.clip((1 - np.sqrt(d2)) * r * S, 0, 1)
        xt = (1 - 2 * sc.moon_frac) * np.sqrt(np.clip(1 - ddy ** 2, 0, 1))
        lit = np.clip(((ddx - xt) if sc.moon_waxing else (-ddx - xt)) * r * S, 0, 1)
        maria = .82 + .18 * fbm2(h, w, int(h / 6), int(w / 6), rng, 3)
        day = float(np.clip(sc.sun_alt / 8, 0, 1))
        mk = (1 - day * .65) * (1 - L.cloud * .85)
        moon = inside * (lit * maria + .04 * (1 - lit))
        sky = lerp(sky, (sky + rgb("#f4f1e6") * moon[..., None]).clip(0, 1), mk)
        halo = np.exp(-np.hypot(X - mx, Y - my) / 38) * sc.moon_frac * .22 * (1 - day) * (1 - L.cloud * .5)
        sky = sky + rgb("#c9d4ee") * halo[..., None]

    if sc.sun_alt > -1.5:
        r = 10
        d = np.hypot(dx, dy)
        disc = np.clip((r - d) * S, 0, 1)
        low = float(np.clip(sc.sun_alt / 12, 0, 1))
        col = lerp(rgb("#ffb070"), rgb("#fffaf0"), low)
        k = (1 - L.cloud ** 1.4 * .97)
        sky = lerp(sky, col[None, None] * np.ones_like(sky), (disc * k)[..., None])
        sky = sky + col * (np.exp(-d / 26) * .6 * k)[..., None]

    sky = sky + (rng.random((h, w, 1), dtype=np.float32) - .5) * (3 / 255)
    return np.concatenate([sky.clip(0, 1), np.ones((h, w, 1), np.float32)], 2)


def rays_layer(sc, L):
    if not (-4 < sc.sun_alt < 22) or L.cloud > .9 or sc.weather.fog:
        return None
    M = 90
    w, h = int((W + 2 * M) * S), int((H + 2 * M) * S)
    rng = np.random.default_rng(sc.seed + 7)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32) / S - M, np.arange(h, dtype=np.float32) / S - M)
    sx, sy = to_screen(sc.sun_alt, sc.sun_az)
    dx, dy = X - sx, Y - sy
    ang = np.arctan2(dy, dx)
    rn = noise1(2048, 46, rng) * .6 + noise1(2048, 11, rng) * .4
    idx = ((ang + math.pi) / TAU * 2047).astype(int)
    a = np.clip((rn[idx] - .48) * 3.2, 0, 1) ** 1.5
    strength = .13 * (.4 + 1.2 * L.cloud * (1 - L.cloud) * 2) * (1 - abs(sc.sun_alt - 4) / 26)
    a = a * np.exp(-np.hypot(dx, dy) / 560) * strength
    col = np.broadcast_to(L.glow, (h, w, 3))
    return np.concatenate([col, a[..., None]], 2), M


def cloud_layer(sc, L):
    c = L.cloud
    if c < .04 and not sc.weather.fog:
        return None
    w, h = int(W * S), int(H * S)
    rng = np.random.default_rng(sc.seed + 3)
    Y = (np.arange(h, dtype=np.float32) / S)[:, None]
    kind, inten = sc.weather.precip

    n = tileable_x(h, w, 3, 5, rng, 6)
    top, bot = 20 - 60 * c, 150 + 70 * c
    band = smooth((Y - top) / 40) * smooth((bot - Y) / 50)
    thr = .62 - .42 * c
    dens = np.clip((n - thr) / (.22 + .1 * c), 0, 1) ** 1.2 * band
    if c > .7:
        sheet = smooth((c - .7) / .25) * smooth((bot + 10 - Y) / 60)
        dens = np.maximum(dens, sheet * (.75 + .25 * n))
    ci = tileable_x(h, w, 4, 3, rng, 5)
    ci = np.clip((ci - .55) * 2.5, 0, 1) * smooth((95 - Y) / 60) * .45 * min(1, c * 3)
    dens = np.maximum(dens, ci)

    up = np.clip(dens - shift_down(dens, int(4 * S)), -1, 1)
    shade = .65 + .55 * up - .25 * smooth((Y - 60) / 120) * dens
    dark_k = 1 - inten * .45 - (.25 if sc.weather.thunder else 0)
    base = (L.amb * .9 + L.horizon * .35) * dark_k
    lit = base[None, None] * shade[..., None] + L.glow * (L.glow_k * .35)
    if L.night > 0:
        lit = lit + rgb("#a06a48") * (.06 * L.night * smooth((Y - 80) / 120))[..., None]
    alpha = (dens * (.88 + .1 * c)).clip(0, 1)
    return np.concatenate([lit.clip(0, 1), alpha[..., None]], 2)


def low_cloud_layer(sc, L):
    k = float(np.clip((L.gloom - .3) / .5, 0, 1))
    if k == 0:
        return None
    w, h = int(W * S), int(H * S)
    rng = np.random.default_rng(sc.seed + 21)
    Y = (np.arange(h, dtype=np.float32) / S)[:, None]
    n = tileable_x(h, w, 3, 7, rng, 6)
    band = smooth((Y - 95) / 45) * smooth((228 - Y) / 30)
    dens = np.clip((n - (.62 - .25 * k)) / .25, 0, 1) * band * (.6 + .4 * k)
    up = np.clip(dens - shift_down(dens, int(4 * S)), -1, 1)
    base = (L.amb * .85 + L.horizon * .3) * (1 - sc.weather.precip[1] * .35)
    col = base[None, None] * (.75 + .45 * up)[..., None]
    return np.concatenate([col.clip(0, 1), dens[..., None]], 2)


def fog_layer(sc, L):
    w_ = sc.weather
    morning_mist = sc.season == "autumn" and -6 < sc.sun_alt < 12 and w_.cloud < .7
    k = 1.0 if w_.fog else (.45 if morning_mist else (.25 if w_.precip[1] > .5 else 0))
    if k == 0:
        return None
    w, h = int(W * S), int(H * S)
    rng = np.random.default_rng(sc.seed + 11)
    Y = (np.arange(h, dtype=np.float32) / S)[:, None]
    n = tileable_x(h, w, 2, 4, rng, 5)
    band = smooth((Y - (150 if w_.fog else 205)) / 50)
    a = np.clip(.25 + .9 * n - .35, 0, 1) * band * .85 * k
    col = lerp(L.haze, L.amb, .4) * 1.05
    return np.concatenate([np.broadcast_to(col, (h, w, 3)), a[..., None]], 2)


LAYERS = [
    (208, 72, 1.00, .60, "#7c7f86", 6),
    (230, 62, .80, .42, "#565c5c", 14),
    (254, 48, .55, .25, "#3a4a3e", 26),
    (294, 24, .32, .08, "#26352a", 34),
]

TREE_COL = {
    "spring": ["#1f3a24", "#27452a", "#1b3220"],
    "summer": ["#16291b", "#1b3020", "#132418"],
    "autumn": ["#1a2c1f", "#1d3022", "#b08a32", "#c39a3c", "#1a2c1f"],
    "winter": ["#1a2722", "#1e2b26", "#18241f"],
}


def conifer(x, yb, h, rnd):
    w = h * rnd.uniform(.3, .42); pts = [(x, yb - h)]
    tiers = rnd.randint(3, 5); right, left = [], []
    for k in range(1, tiers + 1):
        f = k / tiers; yy = yb - h * (1 - f) * .95
        ww = w * (.18 + .5 * f) * rnd.uniform(.85, 1.15)
        right += [(x + ww, yy), (x + ww * .45, yy - h * .04)]
        left += [(x - ww * .45, yy - h * .04), (x - ww, yy)]
    right[-1] = (x + w * .12, yb); left[0] = (x - w * .12, yb)
    return pts + right + [(x + w * .06, yb + 2), (x - w * .06, yb + 2)] + left[::-1]


def mountain_layers(sc, L, only=None):
    rng = np.random.default_rng(5)
    rnd = random.Random(5)
    h = int(H * S)
    sx, sy = to_screen(sc.sun_alt, sc.sun_az)
    sun_up = float(np.clip((sc.sun_alt + 1) / 6, 0, 1))
    direct = sun_up * (1 - L.cloud ** 1.3 * .9) * (0 if sc.weather.fog else 1)
    alpen = float(np.exp(-((sc.sun_alt - 1.5) / 4.5) ** 2)) * (1 - L.cloud * .85)
    kind, inten = sc.weather.precip
    murk = (1.1 if sc.weather.fog else 0) + inten * .7
    out, ridges = [], []
    for i, (base, amp, peak, aer, rock, shift) in enumerate(LAYERS):
        if only is not None and i not in only:
            continue
        lw = int((W + 2 * shift + 20) * S)
        x_off = shift + 10
        rl = ridge_line(lw, rng, .75 + .15 * i)
        rl = base * S - amp * S * (rl - rl.min()) / (rl.max() - rl.min() + 1e-6) * 1.1
        tex_n = fbm2(h, lw, max(2, int(h / 45)), int(lw / 45), rng, 5) - .5
        snow_n = fbm2(h, lw, max(2, int(h / 25)), int(lw / 25), rng, 4) - .5

        up = 2
        paint = Image.new("RGB", (lw * up, h * up), tuple(int(v * 255) for v in rgb(rock)))
        mask = Image.new("L", (lw * up, h * up), 0)
        dm, dp = ImageDraw.Draw(mask), ImageDraw.Draw(paint)
        poly = [(0, h * up)] + [(x * up, rl[x] * up) for x in range(lw)] + [(lw * up, h * up)]
        dm.polygon(poly, fill=255)
        trees = TREE_COL[sc.season]
        if i >= 2:
            density = .3 if i == 2 else 1
            x = 0.0
            while x < lw:
                cl = .5 + .5 * math.sin(x / (90 * S) + 1.3) * math.sin(x / (37 * S))
                if rnd.random() < (.35 + .6 * cl) * density:
                    hh = rnd.uniform(5, 15) * S * (.6 + .6 * cl) * (.6 if i == 2 else 1)
                    yb = rl[min(lw - 1, int(x))] + 3 * S
                    pts = [(px * up, py * up) for px, py in conifer(x, yb, hh, rnd)]
                    col = rgb(rnd.choice(trees))
                    if sc.season == "autumn" and sc.now.month == 9 and col[0] > .5:
                        col = rgb(trees[0])
                    dm.polygon(pts, fill=255)
                    dp.polygon(pts, fill=tuple(int(v * 255) for v in col))
                    if i == 3 and sc.snowline < .3 and rnd.random() < .6:
                        cap = [(px * up, py * up) for px, py in conifer(x, yb - hh * .5, hh * .48, random.Random(int(x)))]
                        dp.polygon(cap, fill=(118, 126, 138))
                x += rnd.uniform(2.2, 5.5) * S
        m = np.asarray(mask.resize((lw, h), Image.LANCZOS), np.float32) / 255
        col = np.asarray(paint.resize((lw, h), Image.LANCZOS), np.float32) / 255

        yy = np.arange(h, dtype=np.float32)[:, None]
        below = np.clip((yy - rl[None, :]) / S, 0, None)
        hgt = np.clip((base * S - yy) / (amp * S * 1.1), 0, 1)
        xs = np.arange(lw, dtype=np.float32) / S - x_off

        if i < 2:
            forest = rgb("#2d3b31") if sc.season != "winter" else rgb("#3a4440")
            col = lerp(col, forest[None, None], smooth((.45 - hgt) / .35)[..., None] * .7)
        col = col * (1 + tex_n[..., None] * (.4 + .2 * i))

        rel = 1 - np.abs(2 * fbm2(h, lw, max(2, int(h / 14)), int(lw / 22), rng, 4) - 1)
        body_h = blur(m, 16 * S) + .05 * rel * m
        gx = np.gradient(body_h, axis=1) * S
        sdir = np.tanh((sx - xs) / 220)[None, :]
        lit = .5 - .5 * np.tanh(gx * 55) * sdir + tex_n * .35
        dir_k = direct * float(np.clip(1.25 - sc.sun_alt / 45, .35, 1))
        shade = (1 - dir_k * .5) + dir_k * 1.05 * lit

        snow = smooth((hgt * peak + snow_n * .22 - sc.snowline) / .05) * (1 - below / 400).clip(0, 1)
        snow *= np.clip(1 - np.abs(gx) * 22, .4, 1)
        snow_col = lerp(np.ones(3, np.float32) * .93, L.glow, alpen * .55)
        col = lerp(col, snow_col[None, None], snow[..., None] * (.9, .85, .55, .4)[i])

        light = L.amb[None, None] * shade[..., None]
        light = light + L.glow * (alpen * .6 * hgt * (1 - i * .2))[..., None]
        img = col * light

        edge = np.clip(m - shift_down(m, max(1, int(1.4 * S))), 0, 1)
        edge = blur(edge, .6)
        back = np.exp(-((xs - sx) / 240) ** 2) * float(np.clip(1 - abs(sc.sun_alt - 2) / 10, 0, 1)) * L.glow_k
        img = img + L.glow * (edge * back[None, :] * (.9 - .2 * i))[..., None]

        haze = lerp(L.horizon, L.haze, .5) * (1 - L.gloom * .3)
        aer_k = np.clip(aer * (1 + murk), 0, .96)
        valley = smooth(below / 70) ** 1.4 * (.5 - .1 * i) * (1 + murk * .5)
        img = lerp(img, haze[None, None], np.clip(aer_k + valley, 0, .97)[..., None])

        out.append((np.concatenate([img.clip(0, 1), m[..., None]], 2), lw, shift))
        ridges.append((rl / S, x_off))
    return out, ridges
