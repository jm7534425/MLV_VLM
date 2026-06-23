import os, random, shutil
random.seed(42)
SRC = "data/AI-hub아동그림"
DST = "data/samples/ai_hub_child"
os.makedirs(DST, exist_ok=True)
cats = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC,d))]
per = 100 // len(cats)  # 25 each
picked = []
for c in cats:
    files = [f for f in os.listdir(os.path.join(SRC,c)) if f.lower().endswith((".jpg",".jpeg",".png",".bmp"))]
    random.shuffle(files)
    sel = files[:per]
    for f in sel:
        shutil.copy(os.path.join(SRC,c,f), os.path.join(DST, f"{c}__{f}"))
        picked.append(f"{c}/{f}")
print(f"categories: {cats}")
print(f"per category: {per}, total copied: {len(picked)}")
print("sample:", picked[:5])
