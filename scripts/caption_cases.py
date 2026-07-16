# -*- coding: utf-8 -*-
"""노트용 캡션 케이스 — 대표 그림 → 썸네일(docs/img) + build_caption 결과(json)."""
import sys, os, glob, json
sys.path.insert(0, "backend")
import caption
from PIL import Image

OUT = "docs/img"; os.makedirs(OUT, exist_ok=True)
CASES = {
    "house":  glob.glob("data/AI-hub아동그림/집/*.jpg")[0],
    "age3":   glob.glob("data/_kiddraw_examples/*person_age3*.png")[0],
    "age5duck": "data/samples/balanced/kid_house_age5_6.jpg",
}

def thumb(path, dst):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    im.thumbnail((300, 300)); im.save(dst, "JPEG", quality=88)

results = {}
for name, path in CASES.items():
    dst = f"{OUT}/case_cap_{name}.jpg"
    thumb(path, dst)
    obj, tk, ms, err = caption.build_caption(open(path, "rb").read())
    results[name] = {"img": f"img/case_cap_{name}.jpg", "ms": ms, "obj": obj, "err": err}
    print(f"{name}: {'OK' if obj else err} ({ms}ms)")

open("outputs/_capcases.json", "w", encoding="utf-8").write(
    json.dumps(results, ensure_ascii=False, indent=2))
print("saved outputs/_capcases.json")
