# -*- coding: utf-8 -*-
"""kiddraw CSV(imageData=base64 PNG) -> 카테고리별 샘플 이미지 벤치마크셋."""
import csv, os, glob, sys, base64, random, json, re
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
random.seed(42)

DIR = r"c:/Users/mini4/Desktop/사전교육/data/full_dataset_with_strokes"
OUT = r"c:/Users/mini4/Desktop/사전교육/data/bench_kiddraw"
N_PER = 25  # 카테고리당 샘플 수
os.makedirs(OUT, exist_ok=True)

manifest = []
cat_counts = {}
for f in sorted(glob.glob(DIR + "/*.csv")):
    rows = []
    with open(f, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            cat = (row.get("category") or "").strip()
            cat = re.sub(r"^an?\s+", "", cat).strip()  # "a dog"->"dog", "an apple"->"apple"
            idata = row.get("imageData") or ""
            if cat and len(idata) > 100:
                rows.append((cat, row.get("age", ""), idata))
    if not rows:
        continue
    cat = rows[0][0]
    random.shuffle(rows)
    sel = rows[:N_PER]
    cdir = os.path.join(OUT, cat.replace(" ", "_"))
    os.makedirs(cdir, exist_ok=True)
    for i, (c, age, idata) in enumerate(sel):
        try:
            png = base64.b64decode(idata)
        except Exception:
            continue
        fn = f"{cat.replace(' ', '_')}_{i:02d}_{age}.png"
        with open(os.path.join(cdir, fn), "wb") as w:
            w.write(png)
        manifest.append({"path": f"{cat.replace(' ', '_')}/{fn}", "category": cat, "age": age})
    cat_counts[cat] = len(sel)

with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as w:
    json.dump({"n_per": N_PER, "categories": sorted(cat_counts), "items": manifest}, w, ensure_ascii=False, indent=1)

print(f"categories: {len(cat_counts)} | total images: {len(manifest)}")
print("cats:", ", ".join(sorted(cat_counts)))
