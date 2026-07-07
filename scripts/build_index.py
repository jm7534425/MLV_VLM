# -*- coding: utf-8 -*-
"""갤러리(AI-Hub + kiddraw) -> 384px 썸네일 + SigLIP2 임베딩 -> FAISS 인덱스.
저장: backend/static/gallery/<gid>.jpg, backend/index/gallery.faiss, backend/index/manifest.json"""
import os, glob, io, base64, json, urllib.request, random
import numpy as np, faiss
from PIL import Image

random.seed(0)
ROOT = r"c:/Users/mini4/Desktop/사전교육"
AIHUB = os.path.join(ROOT, "data/AI-hub아동그림")
KIDD = os.path.join(ROOT, "data/bench_kiddraw")
GAL = os.path.join(ROOT, "backend/static/gallery")
IDXDIR = os.path.join(ROOT, "backend/index")
EMBED_URL = "http://localhost:8100/embed/image"
os.makedirs(GAL, exist_ok=True); os.makedirs(IDXDIR, exist_ok=True)

AIHUB_MAP = {"나무": ("tree", "나무"), "집": ("house", "집"),
             "남자사람": ("person", "남자사람"), "여자사람": ("person", "여자사람")}
KO = {"TV":"텔레비전","airplane":"비행기","apple":"사과","bear":"곰","bed":"침대","bee":"벌","bike":"자전거",
"bird":"새","boat":"배","book":"책","bottle":"병","bowl":"그릇","cactus":"선인장","camel":"낙타","car":"자동차",
"cat":"고양이","chair":"의자","clock":"시계","couch":"소파","cow":"소","cup":"컵","dog":"개","elephant":"코끼리",
"face":"얼굴","fish":"물고기","frog":"개구리","hand":"손","hat":"모자","horse":"말","house":"집","ice_cream":"아이스크림",
"key":"열쇠","lamp":"램프","mushroom":"버섯","octopus":"문어","person":"사람","phone":"전화기","piano":"피아노",
"rabbit":"토끼","scissors":"가위","sheep":"양","snail":"달팽이","spider":"거미","tiger":"호랑이","train":"기차",
"tree":"나무","watch":"손목시계","whale":"고래"}

def to_thumb(path):  # 흰배경 합성 + 384 축소 + RGB
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255,255,255,255))
    im = Image.alpha_composite(bg, im).convert("RGB")
    im.thumbnail((384,384))
    return im

# 1) 갤러리 수집
items = []  # (src_path, source, cat_en, cat_ko, age)
for kor, (en, disp) in AIHUB_MAP.items():
    for p in sorted(glob.glob(os.path.join(AIHUB, kor, "*.jpg"))):
        items.append((p, "aihub", en, disp, None))
for cdir in sorted(glob.glob(os.path.join(KIDD, "*"))):
    if not os.path.isdir(cdir): continue
    cat = os.path.basename(cdir)
    for p in sorted(glob.glob(os.path.join(cdir, "*.png"))):
        age = None
        b = os.path.basename(p)
        if "_age" in b: age = b.split("_age")[-1].split("_")[0].split(".")[0]
        items.append((p, "kiddraw", cat, KO.get(cat, cat), age))

print(f"gallery items: {len(items)}")

# 2) 썸네일 저장 + manifest
manifest = []
for i, (p, source, en, ko, age) in enumerate(items):
    gid = f"{i:05d}"
    out = os.path.join(GAL, f"{gid}.jpg")
    try:
        if not os.path.exists(out):          # 이미 있으면 스킵(재임베딩 시 시간 절약)
            to_thumb(p).save(out, "JPEG", quality=85)
    except Exception as e:
        print("skip", p, e); continue
    manifest.append({"gid": gid, "url": f"/gallery/{gid}.jpg",
                     "source": source, "category": en, "category_ko": ko, "age": age})
    if (i+1) % 400 == 0: print(f"  thumb {i+1}/{len(items)}")

# 3) 임베딩 (썸네일 base64 -> /embed/image, 배치)
def embed_batch(paths):
    b64 = [base64.b64encode(open(os.path.join(GAL, m+".jpg"),"rb").read()).decode() for m in paths]
    req = urllib.request.Request(EMBED_URL, data=json.dumps({"images": b64}).encode(),
                                 headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=300).read())["vecs"]

gids = [m["gid"] for m in manifest]
vecs = []
B = 32
for i in range(0, len(gids), B):
    vecs.extend(embed_batch(gids[i:i+B]))
    if (i+B) % 320 == 0: print(f"  embed {i+B}/{len(gids)}")
vecs = np.asarray(vecs, dtype=np.float32)
print("emb matrix:", vecs.shape)

# 4) 저장: 벡터는 npy(유니코드 경로 안전), 매니페스트 json.
#    faiss.write_index는 Windows 한글경로 미지원 → 로드 시 npy로 IndexFlatIP 인메모리 구성.
np.save(os.path.join(IDXDIR, "gallery_vecs.npy"), vecs)
json.dump({"dim": int(vecs.shape[1]), "count": len(manifest),
           "model": "google/siglip2-so400m-patch16-384", "items": manifest},
          open(os.path.join(IDXDIR, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"saved gallery_vecs.npy {vecs.shape} + manifest ({len(manifest)} items)")
