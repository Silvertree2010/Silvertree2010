import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "assets")
API = "https://api.polyhaven.com/files/"

opener = urllib.request.build_opener()
opener.addheaders = [("User-Agent", "profile-banner/1.0")]
urllib.request.install_opener(opener)


def files(asset):
    return json.load(urllib.request.urlopen(API + asset))


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(url, path)


def model(asset, alpha_key):
    d = files(asset)
    g = d["gltf"]["1k"]["gltf"]
    jobs = [(g["url"], OUT / asset / Path(g["url"]).name)]
    jobs += [(v["url"], OUT / asset / rel) for rel, v in g.get("include", {}).items()]
    jobs.append((d[alpha_key]["1k"]["png"]["url"], OUT / asset / "textures" / "alpha.png"))
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda j: fetch(*j), jobs))


def texture(asset):
    d = files(asset)
    for key, name in (("Diffuse", "diff"), ("nor_gl", "nor")):
        fetch(d[key]["2k"]["jpg"]["url"], OUT / "tex" / asset / f"{name}.jpg")


if __name__ == "__main__":
    model("fir_tree_01", "twig_alpha")
    texture("leafy_grass")
    texture("snow_02")
    print("assets ready in", OUT)
