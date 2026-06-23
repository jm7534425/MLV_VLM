# ============================================================
#  1주차 시각화: 실패 케이스 / 검색 결과 / 모달리티 비교를 이미지 패널로
#  (앞 노트북에서 img_emb, txt_emb, SKETCHES, labels 가 있으면 재사용)
#  런타임: 앞 셀들 실행 후 이어서. 끊겼으면 CELL 0이 재계산.
#  출력: outputs PNG 5장 + 자동 다운로드
# ============================================================


# ====== CELL 0 : 준비 (변수 없으면 재로드/재계산) ======
import os, glob, re, json, textwrap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def gt_of(f): return os.path.basename(f).split("__")[0].lower()
def load(p): return Image.open(p).convert("RGB")

SK_DIR, PH_DIR = "data_samples/sketchy_sketch", "data_samples/sketchy_photo"
if "SKETCHES" not in globals():
    SKETCHES = sorted(f for f in glob.glob(SK_DIR + "/*") if f.lower().endswith((".jpg", ".png")))
    labels = [gt_of(f) for f in SKETCHES]
    CATS = sorted(set(labels))

# 검색용 임베딩이 없으면 SigLIP2로 재계산
if "img_emb" not in globals() or "txt_emb" not in globals():
    print("임베딩 재계산(SigLIP2)...")
    from transformers import AutoProcessor as SPProc, AutoModel
    SIG = "google/siglip2-base-patch16-224"
    sp = SPProc.from_pretrained(SIG)
    sm = AutoModel.from_pretrained(SIG, torch_dtype=torch.float16).to("cuda").eval()
    def _feat(o):
        if torch.is_tensor(o): return o
        for a in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(o, a, None)
            if v is not None: return v
    @torch.no_grad()
    def emb_imgs(paths, bs=16):
        out = []
        for i in range(0, len(paths), bs):
            ims = [load(p) for p in paths[i:i+bs]]
            inp = sp(images=ims, return_tensors="pt").to("cuda")
            e = _feat(sm.get_image_features(**inp)); e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.float().cpu())
        return torch.cat(out)
    @torch.no_grad()
    def emb_txts(ts):
        inp = sp(text=ts, padding="max_length", return_tensors="pt").to("cuda")
        e = _feat(sm.get_text_features(**inp)); return (e / e.norm(dim=-1, keepdim=True)).float().cpu()
    img_emb = emb_imgs(SKETCHES)
    txt_emb = emb_txts([f"a sketch of a {c}" for c in CATS])

os.makedirs("outputs", exist_ok=True)

def panel(ax, img, title, border=None):
    ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    if border:
        for s in ax.spines.values():
            s.set_edgecolor(border); s.set_linewidth(3)


# ====== CELL 1 : 실패 케이스 패널 (GT -> 모델 오답) ======
res = json.load(open("vlm_table_results.json"))["sketch"] if os.path.exists("vlm_table_results.json") else None
if res is None and "sketch_res" in globals(): res = sketch_res
fails = [r for r in res if not r["correct"]][:12]
n = len(fails); cols = 4; rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols*2.6, rows*2.8))
for ax, r in zip(axs.ravel(), fails):
    img = load(os.path.join(SK_DIR, r["file"]))
    panel(ax, img, f"GT: {r['gt']}\n→ pred: {r['pred']}", border="red")
for ax in axs.ravel()[n:]: ax.axis("off")
plt.suptitle("Object abstraction — Failure cases (sketch)", fontsize=13)
plt.tight_layout(); plt.savefig("outputs/fig_failures.png", dpi=130, bbox_inches="tight"); plt.show()


# ====== CELL 2 : text -> image 검색 결과 (쿼리별 top-5) ======
sim = txt_emb @ img_emb.T
queries = ["cat", "hat", "horse", "duck"]            # 보여줄 쿼리 카테고리
queries = [q for q in queries if q in CATS][:4]
fig, axs = plt.subplots(len(queries), 5, figsize=(5*2.3, len(queries)*2.5))
for qi, q in enumerate(queries):
    ci = CATS.index(q)
    top5 = sim[ci].topk(5).indices.tolist()
    for j, idx in enumerate(top5):
        ok = labels[idx] == q
        ttl = f"{labels[idx]}" + (" ✓" if ok else " ✗")
        panel(axs[qi][j], load(SKETCHES[idx]), ttl, border="green" if ok else "red")
    axs[qi][0].set_ylabel(f"query:\n'{q}'", fontsize=11, rotation=0, labelpad=30, va="center")
plt.suptitle("text → image retrieval (top-5, green=correct)", fontsize=13)
plt.tight_layout(); plt.savefig("outputs/fig_retrieval_t2i.png", dpi=130, bbox_inches="tight"); plt.show()


# ====== CELL 3 : image -> image 검색 (쿼리 + 최근접 4개) ======
isim = (img_emb @ img_emb.T).clone(); isim.fill_diagonal_(-1)
# 카테고리 다양하게 쿼리 4장 선택
seen, qidx = set(), []
for i, lb in enumerate(labels):
    if lb not in seen: seen.add(lb); qidx.append(i)
    if len(qidx) == 4: break
fig, axs = plt.subplots(len(qidx), 5, figsize=(5*2.3, len(qidx)*2.5))
for r, qi in enumerate(qidx):
    panel(axs[r][0], load(SKETCHES[qi]), f"QUERY\n{labels[qi]}", border="blue")
    nn = isim[qi].topk(4).indices.tolist()
    for j, idx in enumerate(nn):
        ok = labels[idx] == labels[qi]
        panel(axs[r][j+1], load(SKETCHES[idx]), f"{labels[idx]}" + (" ✓" if ok else " ✗"),
              border="green" if ok else "red")
plt.suptitle("image → image retrieval (query=blue, neighbors top-4)", fontsize=13)
plt.tight_layout(); plt.savefig("outputs/fig_retrieval_i2i.png", dpi=130, bbox_inches="tight"); plt.show()


# ====== CELL 4 : 모달리티 비교 (사진 vs 스케치, 같은 카테고리) ======
pairs = []
for c in CATS:
    ph = sorted(glob.glob(f"{PH_DIR}/{c}__*")); sk = sorted(glob.glob(f"{SK_DIR}/{c}__*"))
    if ph and sk: pairs.append((c, ph[0], sk[0]))
    if len(pairs) == 5: break
fig, axs = plt.subplots(2, len(pairs), figsize=(len(pairs)*2.4, 5))
for j, (c, ph, sk) in enumerate(pairs):
    panel(axs[0][j], load(ph), f"PHOTO\n{c}")
    panel(axs[1][j], load(sk), f"SKETCH\n{c}")
plt.suptitle("Modality: photo vs sketch (same category)", fontsize=13)
plt.tight_layout(); plt.savefig("outputs/fig_modality.png", dpi=130, bbox_inches="tight"); plt.show()


# ====== CELL 5 : 캡셔닝 예시 (이미지 + 생성 캡션) ======
caps = json.load(open("task1_captioning.json")) if os.path.exists("task1_captioning.json") else cap_results
pick = caps[:6]
fig, axs = plt.subplots(2, 3, figsize=(11, 7))
for ax, r in zip(axs.ravel(), pick):
    img = load(os.path.join(SK_DIR, r["file"]))
    cap = "\n".join(textwrap.wrap(r["caption"], 34))
    panel(ax, img, f"[{r['gt']}]")
    ax.set_xlabel(cap, fontsize=7)
plt.suptitle("Captioning examples", fontsize=13)
plt.tight_layout(); plt.savefig("outputs/fig_captioning.png", dpi=130, bbox_inches="tight"); plt.show()


# ====== CELL 6 : 다운로드 ======
from google.colab import files
for f in ["fig_failures.png", "fig_retrieval_t2i.png", "fig_retrieval_i2i.png",
          "fig_modality.png", "fig_captioning.png"]:
    files.download(f"outputs/{f}")
print("5장 저장/다운로드 완료")
