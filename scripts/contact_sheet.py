# -*- coding: utf-8 -*-
"""캡션 실험 12장 컨택트시트 (docs/23용). 각 셀: 그림 + age/tremor 라벨."""
import sys, os, glob, re
sys.path.insert(0, "backend")
import cv_features
from PIL import Image, ImageDraw, ImageFont

SAMPLES, OUT, N = "data/samples/balanced", "docs/img/caption_exp_samples.png", 12
CELL, LAB, PAD, GAP, COLS = 150, 40, 16, 10, 6
try:
    F = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 14)
    FB = ImageFont.truetype("C:/Windows/Fonts/malgunbd.ttf", 15)
except Exception:
    F = FB = ImageFont.load_default()

files = sorted(glob.glob(SAMPLES + "/*.jpg") + glob.glob(SAMPLES + "/*.png"))
files = files[::max(1, len(files) // N)][:N]

def cell(path):
    m = cv_features.measure(open(path, "rb").read())
    age = (re.search(r"age(\d+)", path) or [None, "?"])[1]
    trem = m["line_tremor"]
    shaky = trem > 0.55
    c = Image.new("RGB", (CELL, CELL + LAB), "white")
    im = Image.open(path).convert("RGB"); im.thumbnail((CELL - 8, CELL - 8))
    c.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, CELL - 1, CELL - 1], outline="#ccd", width=1)
    d.text((CELL // 2, CELL + 4), f"age {age}", font=FB, fill="#222", anchor="mt")
    d.text((CELL // 2, CELL + 22), f"tremor {trem:.2f} · {'떨림' if shaky else '안정'}",
           font=F, fill="#d0453a" if shaky else "#888", anchor="mt")
    return c

cells = [cell(f) for f in files]
rows = (len(cells) + COLS - 1) // COLS
W = PAD * 2 + COLS * CELL + (COLS - 1) * GAP
H = PAD * 2 + rows * (CELL + LAB) + (rows - 1) * GAP
canvas = Image.new("RGB", (W, H), "#f7f8fb")
for i, c in enumerate(cells):
    x = PAD + (i % COLS) * (CELL + GAP)
    y = PAD + (i // COLS) * (CELL + LAB + GAP)
    canvas.paste(c, (x, y))
os.makedirs("docs/img", exist_ok=True)
canvas.save(OUT)
print("saved", OUT, canvas.size)
