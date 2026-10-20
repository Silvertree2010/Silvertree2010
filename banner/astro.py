"""Sun and moon position – no dependencies, ~0.5° accuracy (plenty for a banner)."""
import math
from datetime import datetime, timezone

RAD, DEG = math.pi / 180, 180 / math.pi


def _days_j2000(dt: datetime) -> float:
    dt = dt.astimezone(timezone.utc)
    jd = dt.timestamp() / 86400 + 2440587.5
    return jd - 2451543.5          # days since 1999-12-31 0:00 UT (Schlyter epoch)


def _rev(x):
    return x % 360


def _lst(d, lon):
    ws = 282.9404 + 4.70935e-5 * d
    ms = 356.0470 + 0.9856002585 * d
    ls = ws + ms
    gmst0 = _rev(ls + 180) / 15
    ut = (d % 1) * 24
    return (gmst0 + ut + lon / 15) % 24


def _altaz(ra, dec, d, lat, lon):
    ha = (_lst(d, lon) * 15 - ra) * RAD
    dec, lat = dec * RAD, lat * RAD
    alt = math.asin(math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha))
    az = math.atan2(math.sin(ha), math.cos(ha) * math.sin(lat) - math.tan(dec) * math.cos(lat))
    return alt * DEG, _rev(az * DEG + 180)    # azimuth: 0 = north, 90 = east


def _sun_ecl(d):
    w = 282.9404 + 4.70935e-5 * d
    e = 0.016709 - 1.151e-9 * d
    M = _rev(356.0470 + 0.9856002585 * d)
    E = M + DEG * e * math.sin(M * RAD) * (1 + e * math.cos(M * RAD))
    xv = math.cos(E * RAD) - e
    yv = math.sqrt(1 - e * e) * math.sin(E * RAD)
    v = math.atan2(yv, xv) * DEG
    return _rev(v + w), M


def _ecl_to_eq(lon, lat, d):
    ob = (23.4393 - 3.563e-7 * d) * RAD
    lon, lat = lon * RAD, lat * RAD
    x = math.cos(lon) * math.cos(lat)
    y = math.sin(lon) * math.cos(lat)
    z = math.sin(lat)
    ye = y * math.cos(ob) - z * math.sin(ob)
    ze = y * math.sin(ob) + z * math.cos(ob)
    return _rev(math.atan2(ye, x) * DEG), math.atan2(ze, math.hypot(x, ye)) * DEG


def sun(dt, lat, lon):
    d = _days_j2000(dt)
    slon, _ = _sun_ecl(d)
    ra, dec = _ecl_to_eq(slon, 0, d)
    return _altaz(ra, dec, d, lat, lon)


def moon(dt, lat, lon):
    """-> (alt, az, illuminated fraction 0..1, waxing?)"""
    d = _days_j2000(dt)
    N = _rev(125.1228 - 0.0529538083 * d) * RAD
    i = 5.1454 * RAD
    w = _rev(318.0634 + 0.1643573223 * d) * RAD
    e = 0.054900
    M = _rev(115.3654 + 13.0649929509 * d)
    E = M + DEG * e * math.sin(M * RAD) * (1 + e * math.cos(M * RAD))
    for _ in range(4):
        E = E - (E - DEG * e * math.sin(E * RAD) - M) / (1 - e * math.cos(E * RAD))
    xv = 60.2666 * (math.cos(E * RAD) - e)
    yv = 60.2666 * math.sqrt(1 - e * e) * math.sin(E * RAD)
    v = math.atan2(yv, xv)
    r = math.hypot(xv, yv)
    xh = r * (math.cos(N) * math.cos(v + w) - math.sin(N) * math.sin(v + w) * math.cos(i))
    yh = r * (math.sin(N) * math.cos(v + w) + math.cos(N) * math.sin(v + w) * math.cos(i))
    zh = r * math.sin(v + w) * math.sin(i)
    mlon = math.atan2(yh, xh) * DEG
    mlat = math.atan2(zh, math.hypot(xh, yh)) * DEG

    slon, Ms = _sun_ecl(d)
    Mm = M
    Ls = slon
    Lm = _rev(N * DEG + w * DEG + M)
    D = Lm - Ls
    F = Lm - N * DEG
    s = lambda x: math.sin(x * RAD)
    mlon += (-1.274 * s(Mm - 2 * D) + 0.658 * s(2 * D) - 0.186 * s(Ms)
             - 0.059 * s(2 * Mm - 2 * D) - 0.057 * s(Mm - 2 * D + Ms)
             + 0.053 * s(Mm + 2 * D) + 0.046 * s(2 * D - Ms) + 0.041 * s(Mm - Ms)
             - 0.035 * s(D) - 0.031 * s(Mm + Ms))
    mlat += (-0.173 * s(F - 2 * D) - 0.055 * s(Mm - F - 2 * D)
             - 0.046 * s(Mm + F - 2 * D) + 0.033 * s(F + 2 * D))

    ra, dec = _ecl_to_eq(mlon, mlat, d)
    # topocentric parallax (the moon is close enough for it to matter, ~1°)
    alt, az = _altaz(ra, dec, d, lat, lon)
    alt -= DEG * math.asin(1 / r) * math.cos(alt * RAD)

    elong = math.acos(math.cos((slon - mlon) * RAD) * math.cos(mlat * RAD))
    frac = (1 - math.cos(elong)) / 2
    waxing = _rev(mlon - slon) < 180
    return alt, az, frac, waxing
