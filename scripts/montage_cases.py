# -*- coding: utf-8 -*-
"""19번 문서용 검색 케이스 몽타주: [쿼리] → [top-N 결과] 가로 스트립.
결과를 눈으로 보게 해서 성공/실패를 시각화."""
import os, sys, json
sys.path.insert(0, "backend")
import retrieval
from PIL import Image, ImageDraw, ImageFont

GAL = "backend/static/gallery"
OUT = "docs/img"; os.makedirs(OUT, exist_ok=True)
CELL, LAB, PAD, GAP = 150, 46, 16, 10
try:
    F = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 15)
    FB = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 16)
except Exception:
    F = FB = ImageFont.load_default()

def cell(img, title, sub, accent=None):
    c = Image.new("RGB", (CELL, CELL + LAB), "white")
    if img is not None:
        im = img.convert("RGB"); im.thumbnail((CELL - 8, CELL - 8))
        c.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, CELL - 1, CELL - 1], outline=accent or "#dddddd", width=3 if accent else 1)
    d.text((CELL // 2, CELL + 6), title, font=FB, fill=accent or "#222222", anchor="mt")
    if sub:
        d.text((CELL // 2, CELL + 26), sub, font=F, fill="#888888", anchor="mt")
    return c

def load_gid(gid):
    return Image.open(os.path.join(GAL, gid + ".jpg"))

def strip(query_cell, results, path):
    cells = [query_cell] + [cell(load_gid(r["gid"]), r["category_ko"] or r["category"], f'{r["score"]:.3f}') for r in results]
    W = PAD * 2 + len(cells) * CELL + (len(cells) - 1) * GAP
    H = PAD * 2 + CELL + LAB
    canvas = Image.new("RGB", (W, H), "#f7f8fb")
    x = PAD
    for c in cells:
        canvas.paste(c, (x, PAD)); x += CELL + GAP
    canvas.save(path); print("saved", path, canvas.size)

def txt_query_cell(text):
    c = Image.new("RGB", (CELL, CELL + LAB), "#eef1ff")
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, CELL - 1, CELL - 1], outline="#5b6ef5", width=3)
    d.text((CELL // 2, CELL // 2 - 10), "\U0001F50D", font=FB, fill="#5b6ef5", anchor="mm")
    d.text((CELL // 2, CELL // 2 + 18), text, font=FB, fill="#333", anchor="mm")
    d.text((CELL // 2, CELL + 6), "쿼리(텍스트)", font=F, fill="#5b6ef5", anchor="mt")
    return c

def img_query_cell(gid, label):
    return cell(load_gid(gid), "쿼리(그림)", label, accent="#5b6ef5")

def gid_of(cat, source=None, nth=0):
    man = json.load(open("backend/index/manifest.json", encoding="utf-8"))["items"]
    xs = [m for m in man if m["category"] == cat and (source is None or m["source"] == source)]
    return xs[nth]["gid"]

# 1) 성공(텍스트): 집 그림
strip(txt_query_cell("집 그림"), retrieval.retrieve_text("집 그림", 6), f"{OUT}/case_text_house.png")
# 2) 의미이웃 혼동: a cat -> 고양이+호랑이
strip(txt_query_cell("a cat"), retrieval.retrieve_text("a cat", 7), f"{OUT}/case_text_cat.png")
# 3) OOG: 피자 (하한 무시하고 뭐가 끌려오나 + 저점수 보이기)
oog = retrieval.search(retrieval.embed_text("피자 그림"), 6, min_score=0.0)
strip(txt_query_cell("피자 그림"), oog, f"{OUT}/case_oog_pizza.png")
# 4) img->img 성공: AI-Hub 집
hg = gid_of("house", "aihub", 2)
strip(img_query_cell(hg, "AI-Hub"), retrieval.retrieve_image(load_gid(hg).tobytes() and open(os.path.join(GAL, hg+'.jpg'),'rb').read(), 7)[1:], f"{OUT}/case_img_house.png")
# 5) img->img 실패: kiddraw 개 -> 토끼
dg = gid_of("dog", "kiddraw", 3)
strip(img_query_cell(dg, "kiddraw"), retrieval.retrieve_image(open(os.path.join(GAL, dg+'.jpg'),'rb').read(), 7)[1:], f"{OUT}/case_img_dog.png")
print("done")
