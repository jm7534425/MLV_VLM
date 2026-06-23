import os, json, time, re, shutil, urllib.request, urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = "https://datasets-server.huggingface.co"
OUT = "data/samples"
HDR = {"User-Agent": "Mozilla/5.0"}
PHOTO_DS = "DrRORAL/sketchy-dataset"      # photos: src OK, category name in key
SKETCH_DS = "JamieSJS/sketchy"            # sketches: src OK (query/test)

def get(url, tries=7):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                w = min(5 * (2 ** i), 60); print(f"   429 backoff {w}s"); time.sleep(w); continue
            if i == tries - 1: raise
            time.sleep(2)
        except Exception:
            if i == tries - 1: raise
            time.sleep(2)
    raise RuntimeError("retries")

def rows(ds, cfg, spl, off, length=100):
    time.sleep(0.9)
    url = f"{BASE}/rows?dataset={ds}&config={cfg}&split={spl}&offset={off}&length={length}"
    return json.loads(get(url).decode())

def synset(inst): return inst.split("_")[0]
def base_inst(s): return re.sub(r"-\d+$", "", s)

def save_img(src, path):
    try:
        with open(path, "wb") as f: f.write(get(src))
        return True
    except Exception:
        return False

def collect_photos(n_off=30):
    # DrRORAL photos live in offset [0,12500], alphabetical by category
    photos = defaultdict(list)   # synset -> [(inst, src)]
    name = {}                    # synset -> category name
    offs = [int(i * 12500 / n_off) for i in range(n_off)]
    for o in offs:
        for r in rows(PHOTO_DS, "default", "train", o, 60)["rows"]:
            k = r["row"]["__key__"]; cell = r["row"].get("jpg")
            if ".ipynb_checkpoints" in k or not (isinstance(cell, dict) and cell.get("src")): continue
            p = k.split("/")
            if len(p) < 5 or p[1] != "photo": continue
            cat, inst = p[3], p[-1]; sy = synset(inst)
            name[sy] = cat; photos[sy].append((inst, cell["src"]))
    print(f"  photos: {sum(len(v) for v in photos.values())} imgs across {len(photos)} categories")
    return photos, name

def collect_sketches(n_off=25):
    sk = defaultdict(list)   # synset -> [(sid, src)]
    for i in range(n_off):
        o = int(i * 452886 / n_off)
        for r in rows(SKETCH_DS, "query", "test", o, 100)["rows"]:
            sid = r["row"]["id"]; cell = r["row"].get("image")
            if not (isinstance(cell, dict) and cell.get("src")): continue
            sk[synset(sid)].append((sid, cell["src"]))
    print(f"  sketches: {sum(len(v) for v in sk.values())} imgs across {len(sk)} synsets")
    return sk

def main(n=100, cap=6):
    ps = os.path.join(OUT, "sketchy_photo"); skd = os.path.join(OUT, "sketchy_sketch")
    for d in (ps, skd):
        if os.path.isdir(d): shutil.rmtree(d)
        os.makedirs(d)
    photos, name = collect_photos()
    sketches = collect_sketches()
    common = sorted(set(photos) & set(sketches))
    print(f"  common categories: {len(common)} -> {[name[c] for c in common]}")
    # build pairs per category (prefer same instance, else any)
    by_cat = defaultdict(list)
    for sy in common:
        pmap = {base_inst(i): (i, s) for i, s in photos[sy]}
        used = set()
        for sid, ssrc in sketches[sy]:
            bi = base_inst(sid)
            if bi in pmap:                       # exact same-instance pair
                inst, psrc = pmap[bi]
                by_cat[sy].append((name[sy], inst, sid, psrc, ssrc)); used.add(inst)
        # fallback: category-level pairs (different instance)
        leftover_p = [(i, s) for i, s in photos[sy] if i not in used]
        for (inst, psrc), (sid, ssrc) in zip(leftover_p, sketches[sy]):
            by_cat[sy].append((name[sy], inst, sid, psrc, ssrc))
    # round-robin across categories, cap, then fill
    chosen, cnt = [], defaultdict(int)
    for c in (cap, 9999):
        while len(chosen) < n:
            added = False
            for sy in common:
                if cnt[sy] < c and by_cat[sy]:
                    chosen.append(by_cat[sy].pop()); cnt[sy] += 1; added = True
                    if len(chosen) >= n: break
            if not added: break
        if len(chosen) >= n: break
    # download
    tasks = []
    for cat, inst, sid, psrc, ssrc in chosen:
        tasks.append((psrc, os.path.join(ps, f"{cat}__{inst}.jpg")))
        tasks.append((ssrc, os.path.join(skd, f"{cat}__{sid}.jpg")))
    ok = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(lambda t: save_img(*t), tasks): ok += res
    print(f"\nDONE: {len(chosen)} pairs, files_ok={ok}/{len(tasks)}")
    print(f"categories covered: {len([k for k,v in cnt.items() if v])}")
    print("per-category:", {name[k]: v for k, v in sorted(cnt.items()) if v})

if __name__ == "__main__":
    main(100, cap=6)
