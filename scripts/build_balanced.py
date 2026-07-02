# -*- coding: utf-8 -*-
"""균형 테스트셋: kiddraw(나이2~7, 어린단계) + AI-Hub(8~13, 상위단계). HTP 카테고리 우선."""
import csv, glob, os, base64, io, random, re
from collections import defaultdict
from PIL import Image
csv.field_size_limit(10**7)
random.seed(7)
OUT = "data/samples/balanced"; os.makedirs(OUT, exist_ok=True)
for f in glob.glob(OUT+"/*"): os.remove(f)

# --- kiddraw: 나이 2~7, person/house/tree, 나이당 2장 ---
def render(row, path):
    img = Image.open(io.BytesIO(base64.b64decode(row["imageData"])))
    bg = Image.new("RGB", img.size, (255,255,255))
    if img.mode=="RGBA": bg.paste(img, mask=img.split()[3])
    else: bg=img.convert("RGB")
    bg.save(path)

by_age = defaultdict(list)
for cf in glob.glob("data/full_dataset_with_strokes/*.csv"):
    cat = None
    for k in ["person","house","tree"]:
        if k in cf.lower(): cat=k
    if not cat: continue
    with open(cf, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            a = row.get("age","")
            m = re.match(r"age(\d+)", a)
            if m and 2 <= int(m.group(1)) <= 7 and row.get("imageData"):
                by_age[int(m.group(1))].append((cat,row))
kid = 0
for age in range(2,8):
    random.shuffle(by_age[age])
    for cat,row in by_age[age][:2]:
        render(row, f"{OUT}/kid_{cat}_age{age}_{kid}.jpg"); kid+=1
print("kiddraw rendered:", kid)

# --- AI-Hub: 나이 8~13, 나이당 2장 ---
ah = defaultdict(list)
for f in glob.glob("data/samples/ai_hub_child/*.jpg"):
    age = int(os.path.basename(f).split("__")[1].split("_")[1])
    ah[age].append(f)
import shutil
n=0
for age in range(8,14):
    random.shuffle(ah[age])
    for f in ah[age][:2]:
        shutil.copy(f, f"{OUT}/ah_age{age}_{n}.jpg"); n+=1
print("aihub copied:", n)
print("total:", len(glob.glob(OUT+"/*")))
