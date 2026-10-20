"""One-off: download Terrarium elevation tiles (AWS open data, Mapzen/Tilezen) and save a
16-bit heightmap + metadata. Run once; the result is committed so the workflow needs no network
for terrain.

    python banner/tools/fetch_dem.py
"""
import io, json, math, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Z = 12
# Bernese Oberland: from the lakes down to the Valais peaks
NORTH, SOUTH, WEST, EAST = 46.73, 46.22, 7.55, 8.42
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
OUT = Path(__file__).resolve().parent.parent / "data"


def tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def fetch(t):
    x, y = t
    with urllib.request.urlopen(URL.format(z=Z, x=x, y=y), timeout=30) as r:
        im = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"), np.float64)
    return t, im[..., 0] * 256 + im[..., 1] + im[..., 2] / 256 - 32768


def main():
    x0, y0 = tile_xy(NORTH, WEST, Z)
    x1, y1 = tile_xy(SOUTH, EAST, Z)
    tx = range(int(x0), int(x1) + 1)
    ty = range(int(y0), int(y1) + 1)
    tiles = [(x, y) for y in ty for x in tx]
    with ThreadPoolExecutor(8) as ex:
        res = dict(ex.map(fetch, tiles))
    mosaic = np.vstack([np.hstack([res[(x, y)] for x in tx]) for y in ty])
    # crop to the bbox in pixel space
    px0, py0 = int((x0 - tx[0]) * 256), int((y0 - ty[0]) * 256)
    px1, py1 = int((x1 - tx[0]) * 256), int((y1 - ty[0]) * 256)
    dem = np.clip(mosaic[py0:py1, px0:px1], 0, 16000)
    OUT.mkdir(exist_ok=True)
    Image.fromarray((dem * 4).round().astype(np.uint16)).save(OUT / "dem.png", optimize=True)
    meta = dict(z=Z, scale=4, px0=tx[0] * 256 + px0, py0=ty[0] * 256 + py0,
                width=dem.shape[1], height=dem.shape[0])
    (OUT / "dem.json").write_text(json.dumps(meta, indent=1))
    print(dem.shape, dem.min(), dem.max())


if __name__ == "__main__":
    main()
