import math, random

W, H = 1280, 300
TAU = 2 * math.pi
EASE = 'calcMode="spline" keySplines=".45 0 .55 1;.45 0 .55 1"'


def f(x):
    return f"{x:.1f}".rstrip("0").rstrip(".")


def catmull(points, closed=True):
    n = len(points)
    pts = points + points[:3] if closed else [points[0]] + points + [points[-1]]
    d = [f"M{f(pts[1][0])},{f(pts[1][1])}"]
    rng_ = range(1, n + 1) if closed else range(1, n)
    for i in rng_:
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        d.append(f"C{f(c1[0])},{f(c1[1])} {f(c2[0])},{f(c2[1])} {f(p2[0])},{f(p2[1])}")
    return "".join(d) + ("Z" if closed else "")


WING_UP = "M-3.2,-.7 Q-1.6,-2.3 0,0 Q1.6,-2.3 3.2,-.7"
WING_DN = "M-3.2,1.3 Q-1.6,.7 0,0 Q1.6,.7 3.2,1.3"


def bird(color, scale, flap, begin, glide=False, sw=.95):
    vals = f"{WING_UP};{WING_DN};{WING_UP}" + (f";{WING_UP};{WING_UP}" if glide else "")
    kt = ' keyTimes="0;.18;.36;.7;1"' if glide else ""
    return (f'<path d="{WING_UP}" transform="scale({f(scale)})" fill="none" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">'
            f'<animate attributeName="d" values="{vals}"{kt} dur="{f(flap)}s" begin="-{f(begin)}s" '
            f'repeatCount="indefinite"/></path>')


def cranes(rnd, color):
    out = []
    for k, (y, dur, off, n, s) in enumerate([(78, 70, 0, 11, 1.25), (118, 95, 48, 7, .95)]):
        g = []
        for i in range(n):
            side = 1 if i % 2 else -1
            rank = (i + 1) // 2
            dx, dy = rank * 11 * s, side * rank * 6.5 * s + rnd.uniform(-1, 1)
            g.append(f'<g transform="translate({f(dx)},{f(dy)})">'
                     f'{bird(color, s, rnd.uniform(.9, 1.1), rnd.uniform(0, 1), glide=True)}'
                     f'<animateTransform attributeName="transform" type="translate" '
                     f'values="{f(dx)},{f(dy)};{f(dx + rnd.uniform(-2, 2))},{f(dy + rnd.uniform(-2, 2))};{f(dx)},{f(dy)}" '
                     f'dur="{f(rnd.uniform(4, 7))}s" repeatCount="indefinite"/></g>')
        out.append(f'<g><g>{"".join(g)}<animateTransform attributeName="transform" type="translate" '
                   f'values="0,0;0,-8;0,0" dur="9s" {EASE} repeatCount="indefinite"/></g>'
                   f'<animateTransform attributeName="transform" type="translate" from="{W + 160},{y}" '
                   f'to="-200,{y - 25}" dur="{dur}s" begin="-{off}s" repeatCount="indefinite"/></g>')
    return "".join(out)


SWIFT = "M-4.5,.2 Q-2.2,-1.4 0,-.4 Q2.2,-1.4 4.5,.2 Q2,-.3 .4,1 L-.4,1 Q-2,-.3 -4.5,.2Z"


def swifts(rnd, color):
    out = []
    for _ in range(9):
        cx, cy = rnd.uniform(250, 1050), rnd.uniform(50, 140)
        pts = [(cx + rnd.uniform(-260, 260), cy + rnd.uniform(-50, 50)) for _ in range(6)]
        out.append(f'<path d="{SWIFT}" fill="{color}">'
                   f'<animateMotion dur="{f(rnd.uniform(6, 11))}s" begin="-{f(rnd.uniform(0, 10))}s" '
                   f'repeatCount="indefinite" rotate="auto" path="{catmull(pts)}"/></path>')
    return "".join(out)


BAT = "M-3.4,0 L-2.4,-1.6 L-1.4,-.4 L-.5,-1.2 L0,-.6 L.5,-1.2 L1.4,-.4 L2.4,-1.6 L3.4,0 L1.2,.4 L0,1.1 L-1.2,.4Z"


def bats(rnd, color, ridge_y):
    out = []
    for _ in range(6):
        cx = rnd.uniform(150, 1130)
        base = ridge_y(cx)
        pts = [(cx + rnd.uniform(-110, 110), base - rnd.uniform(15, 75)) for _ in range(9)]
        out.append(f'<g><path d="{BAT}" fill="{color}">'
                   f'<animateTransform attributeName="transform" type="scale" values="1,1;1,.25;1,1" '
                   f'dur="{f(rnd.uniform(.13, .18))}s" repeatCount="indefinite"/></path>'
                   f'<animateMotion dur="{f(rnd.uniform(7, 12))}s" begin="-{f(rnd.uniform(0, 10))}s" '
                   f'repeatCount="indefinite" path="{catmull(pts)}"/></g>')
    return "".join(out)


def fireflies(rnd, ridge_y):
    out = ['<radialGradient id="ff"><stop offset="0" stop-color="#e9ff8f" stop-opacity=".9"/>'
           '<stop offset=".25" stop-color="#c8f560" stop-opacity=".35"/>'
           '<stop offset="1" stop-color="#c8f560" stop-opacity="0"/></radialGradient>']
    for _ in range(34):
        x = rnd.uniform(30, 1250)
        y = ridge_y(x) + rnd.uniform(-22, 14)
        r = rnd.uniform(5, 14)
        loop = [(x + r * math.cos(a) + rnd.uniform(-3, 3), y + r * .6 * math.sin(a) + rnd.uniform(-3, 3))
                for a in [k * TAU / 5 for k in range(5)]]
        dur = rnd.uniform(3.5, 7)
        on = rnd.uniform(.3, .6)
        out.append(f'<g opacity="0"><circle r="6" fill="url(#ff)"/><circle r=".9" fill="#f4ffc4"/>'
                   f'<animate attributeName="opacity" values="0;0;1;.9;0;0" '
                   f'keyTimes="0;{f(on)};{f(on + .08)};{f(on + .2)};{f(on + .32)};1" '
                   f'dur="{f(dur)}s" begin="-{f(rnd.uniform(0, dur))}s" repeatCount="indefinite"/>'
                   f'<animateMotion dur="{f(rnd.uniform(6, 12))}s" repeatCount="indefinite" path="{catmull(loop)}"/></g>')
    return "".join(out)


def choughs(rnd, color, peak):
    px, py = peak
    out = []
    for i in range(7):
        rx, ry = rnd.uniform(40, 95), rnd.uniform(10, 24)
        cy = py - rnd.uniform(18, 50)
        pts = [(px + rx * math.cos(a), cy + ry * math.sin(a)) for a in [k * TAU / 8 for k in range(8)]]
        dur = rnd.uniform(9, 15)
        out.append(f'<g>{bird(color, rnd.uniform(.75, 1), .55, rnd.uniform(0, 1), glide=True, sw=1.1)}'
                   f'<animateMotion dur="{f(dur)}s" begin="-{f(rnd.uniform(0, dur))}s" '
                   f'repeatCount="indefinite" path="{catmull(pts)}"/></g>')
    return "".join(out)


def murmuration(rnd, color, n=300, dur=32.0):
    steps = 72

    def centre(t):
        return 640 + 320 * math.sin(t) + 38 * math.sin(3 * t + .7), 100 + 28 * math.sin(2 * t + .5) + 8 * math.sin(5 * t + 1)

    def local(t, u, v, k, p):
        th = .55 * math.sin(2 * t + 1) + .3 * math.sin(3 * t)
        sx = 1.15 * (80 + 60 * math.sin(3 * t + .4))
        sy = 1.15 * (20 + 14 * math.sin(2 * t + 2))
        uu = u + .25 * math.sin(3 * u + 5 * t) + .08 * math.sin(7 * v + 3 * t)
        vv = v + .8 * math.sin(math.pi * u + 4 * t) + .15 * math.sin(5 * u - 6 * t)
        x, y = sx * uu, sy * vv
        c, s = math.cos(th), math.sin(th)
        return c * x - s * y + 1.8 * math.sin(k * t + p), s * x + c * y + 1.3 * math.cos((k + 1) * t + p)

    cen = " L".join(f"{x:.0f},{y:.0f}" for x, y in (centre(TAU * s / 144) for s in range(145)))
    g = [f'<path id="mc" d="M{cen}" fill="none"/>'
         f'<g><animateMotion dur="{dur}s" repeatCount="indefinite"><mpath href="#mc"/></animateMotion>']
    for _ in range(n):
        while True:
            u, v = rnd.uniform(-1, 1), rnd.uniform(-1, 1)
            if u * u + v * v <= 1:
                break
        u, v = u * (.5 + .5 * abs(u)), v * (.55 + .45 * abs(v))
        k, p = rnd.choice([4, 5, 6, 7, 8]), rnd.uniform(0, TAU)
        pts = " L".join(f"{x:.0f},{y:.0f}" for x, y in (local(TAU * s / steps, u, v, k, p) for s in range(steps + 1)))
        g.append(f'<circle r="{rnd.uniform(.55, 1.1):.1f}" fill="{color}" opacity="{rnd.uniform(.8, 1):.2f}">'
                 f'<animateMotion dur="{dur}s" repeatCount="indefinite" path="M{pts}"/></circle>')
    g.append("</g>")
    return "".join(g)


def meteors(rnd):
    out = ['<linearGradient id="mt" x1="0" x2="1"><stop offset="0" stop-color="#fff" stop-opacity="0"/>'
           '<stop offset="1" stop-color="#fff" stop-opacity=".9"/></linearGradient>']
    for dur in (17, 29, 41):
        x, y = rnd.uniform(150, 1100), rnd.uniform(15, 90)
        ang = rnd.uniform(15, 40) * rnd.choice([1, -1])
        t0 = rnd.uniform(.1, .8)
        out.append(f'<g opacity="0" transform="translate({f(x)},{f(y)}) rotate({f(ang)})">'
                   f'<rect x="-70" y="-.6" width="70" height="1.2" rx=".6" fill="url(#mt)">'
                   f'<animate attributeName="x" values="-70;-70;60;60" keyTimes="0;{f(t0)};{f(t0 + .025)};1" '
                   f'dur="{dur}s" repeatCount="indefinite"/></rect>'
                   f'<animate attributeName="opacity" values="0;0;1;0;0" keyTimes="0;{f(t0)};{f(t0 + .01)};{f(t0 + .025)};1" '
                   f'dur="{dur}s" repeatCount="indefinite"/></g>')
    return "".join(out)


def wind_vec(w):
    toward = math.radians((w.wind_dir + 180) % 360)
    east = math.sin(toward)
    return -east * w.wind / 25


def rain(rnd, inten, drift, color):
    out = []
    n = int(70 + 230 * inten)
    tilt = max(-.7, min(.7, drift * .35))
    for i in range(n):
        layer = rnd.random()
        l = 6 + 12 * layer
        dur = 1.0 - .55 * layer + rnd.uniform(-.05, .05)
        op = .12 + .3 * layer
        x0 = rnd.uniform(-60, W + 60)
        dx = tilt * (H + 40)
        nx, ny = tilt / math.hypot(tilt, 1), 1 / math.hypot(tilt, 1)
        out.append(f'<line x1="0" y1="0" x2="{f(nx * l)}" y2="{f(ny * l)}" stroke="{color}" '
                   f'stroke-width="{f(.6 + .5 * layer)}" opacity="{f(op)}" stroke-linecap="round">'
                   f'<animateTransform attributeName="transform" type="translate" '
                   f'from="{f(x0 - dx / 2)},-20" to="{f(x0 + dx / 2)},{H + 20}" dur="{f(dur)}s" '
                   f'begin="-{f(rnd.uniform(0, dur))}s" repeatCount="indefinite"/></line>')
    return "".join(out)


def snow(rnd, inten, drift, color):
    out = []
    n = int(60 + 220 * inten)
    for _ in range(n):
        layer = rnd.random()
        r = .5 + 1.6 * layer ** 2
        dur = 16 - 9 * layer + rnd.uniform(-1, 1)
        x0 = rnd.uniform(-80, W + 80)
        sway, ph = rnd.uniform(6, 18), rnd.uniform(0, TAU)
        pts = [(x0 + drift * 40 * k / 8 + sway * math.sin(k * 1.4 + ph), -10 + (H + 20) * k / 8) for k in range(9)]
        out.append(f'<circle r="{f(r)}" fill="{color}" opacity="{f(.35 + .6 * layer)}">'
                   f'<animateMotion dur="{f(dur)}s" begin="-{f(rnd.uniform(0, dur))}s" repeatCount="indefinite" '
                   f'path="{catmull(pts, closed=False)}"/></circle>')
    return "".join(out)


def lightning(rnd, ridge_y):
    out = []
    for dur in (11, 17.5):
        t0 = rnd.uniform(.15, .8)
        kt = f"0;{f(t0)};{f(t0 + .006)};{f(t0 + .014)};{f(t0 + .02)};{f(t0 + .032)};1"
        out.append(f'<rect width="{W}" height="{H}" fill="#e8eeff" opacity="0">'
                   f'<animate attributeName="opacity" values="0;0;.55;.05;.35;0;0" keyTimes="{kt}" '
                   f'dur="{dur}s" repeatCount="indefinite"/></rect>')
        x, y = rnd.uniform(200, 1080), rnd.uniform(40, 70)
        end = ridge_y(x) - 4
        pts = [(x, y)]
        while pts[-1][1] < end:
            px, py = pts[-1]
            pts.append((px + rnd.uniform(-14, 14), min(end, py + rnd.uniform(9, 22))))
        d = "M" + " L".join(f"{f(a)},{f(b)}" for a, b in pts)
        out.append(f'<g opacity="0"><path d="{d}" fill="none" stroke="#c9d6ff" stroke-width="5" opacity=".25"/>'
                   f'<path d="{d}" fill="none" stroke="#fff" stroke-width="1.3"/>'
                   f'<animate attributeName="opacity" values="0;0;1;.2;1;0;0" keyTimes="{kt}" '
                   f'dur="{dur}s" repeatCount="indefinite"/></g>')
    return "".join(out)
