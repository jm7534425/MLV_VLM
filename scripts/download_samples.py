import os, json, time, re, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

BASE = "https://datasets-server.huggingface.co"
OUT = "data/samples"
HDR = {"User-Agent": "Mozilla/5.0"}

def get(url, tries=7):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = min(5 * (2 ** i), 60)
                print(f"   429 backoff {wait}s...")
                time.sleep(wait); continue
            if i == tries - 1: raise
            time.sleep(2)
        except Exception:
            if i == tries - 1: raise
            time.sleep(2)
    raise RuntimeError("max retries: " + url)

def rows(ds, cfg, spl, offset, length):
    time.sleep(0.8)  # politeness gap to avoid 429 on metadata calls
    url = f"{BASE}/rows?dataset={ds}&config={cfg}&split={spl}&offset={offset}&length={length}"
    return json.loads(get(url).decode())

def save_img(src, path):
    try:
        with open(path, "wb") as f:
            f.write(get(src))
        return True
    except Exception:
        return False

def download_all(tasks, workers=4):
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(lambda t: save_img(*t), tasks):
            ok += res
    return ok

def dl_simple(ds, cfg, spl, folder, n=100, col="image"):
    d = os.path.join(OUT, folder); os.makedirs(d, exist_ok=True)
    r = rows(ds, cfg, spl, 0, n)
    tasks = []
    for i, row in enumerate(r["rows"]):
        cell = row["row"][col]
        src = cell["src"] if isinstance(cell, dict) else cell
        tasks.append((src, os.path.join(d, f"{folder}_{i:03d}.jpg")))
    ok = download_all(tasks)
    print(f"[{folder}] downloaded {ok}/{len(tasks)}")
    return ok

def base_id(s): return re.sub(r"-\d+$", "", s)

def dl_sketchy_pairs(n=100):
    ds = "JamieSJS/sketchy"
    ps = os.path.join(OUT, "sketchy_photo"); os.makedirs(ps, exist_ok=True)
    sk = os.path.join(OUT, "sketchy_sketch"); os.makedirs(sk, exist_ok=True)
    # 1) collect a pool of sketches (query/test): base_id -> (sketch_id, src)
    need = {}
    for off in range(0, 600, 100):          # up to 6 pages
        r = rows(ds, "query", "test", off, 100)
        if not r["rows"]: break
        for row in r["rows"]:
            sid = row["row"]["id"]
            need.setdefault(base_id(sid), (sid, row["row"]["image"]["src"]))
        print(f"   sketches collected: {len(need)} base-ids (offset {off})")
        if len(need) >= n * 3: break
    # 2) page corpus (photos); keep any photo whose id has a matching sketch
    pairs = {}
    for off in range(0, 1500, 100):         # up to 15 pages, stop early when enough
        if len(pairs) >= n: break
        r = rows(ds, "corpus", "corpus", off, 100)
        if not r["rows"]: break
        for row in r["rows"]:
            pid = row["row"]["id"]
            if pid in need and pid not in pairs:
                pairs[pid] = (need[pid][0], row["row"]["image"]["src"], need[pid][1])
        print(f"   matched pairs: {len(pairs)} (corpus offset {off})")
    # 3) download matched pairs
    tasks = []
    for pid, (sid, psrc, ssrc) in list(pairs.items())[:n]:
        tasks.append((psrc, os.path.join(ps, f"{pid}.jpg")))
        tasks.append((ssrc, os.path.join(sk, f"{sid}.jpg")))
    ok = download_all(tasks)
    print(f"[sketchy pairs] matched={min(len(pairs),n)} files_ok={ok}/{len(tasks)}")
    return min(len(pairs), n)

if __name__ == "__main__":
    t0 = time.time()
    try: dl_sketchy_pairs(100)
    except Exception as e: print("sketchy FAILED:", e)
    try: dl_simple("zoheb/sketch-scene", "default", "train", "sketchyscene", 100)
    except Exception as e: print("sketchyscene FAILED:", e)
    try: dl_simple("rafaelpadilla/coco2017", "default", "val", "coco", 100)
    except Exception as e: print("coco FAILED:", e)
    print(f"done in {time.time()-t0:.0f}s")
