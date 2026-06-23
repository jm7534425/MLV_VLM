# ============================================================
#  1주차 산출물: VLM 4대 task 실습 결과
#  (Captioning / VQA / Classification / Retrieval)
#
#  ※ 앞 노트북(colab_fill_table.py)에서 model/processor/ask() 가 이미 로드된 상태를
#    이어서 사용. 새 런타임이면 CELL 0이 자동으로 다시 로드함.
#  런타임: GPU(T4)
# ============================================================


# ====== CELL 0 : (가드) 모델/데이터 준비 — 이미 있으면 그냥 통과 ======
import os, glob, re, json, torch

def gt_of(f):  # "cat__n0212...jpg" -> "cat"
    return os.path.basename(f).split("__")[0].lower()

if "ask" not in globals():
    print("ask() 없음 -> Qwen2.5-VL 재로드")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL)
    def ask(image_path, prompt, max_new=48):
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image_path}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(msgs)
        inp = processor(text=[text], images=imgs, videos=vids,
                        padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        tr = [o[len(i):] for i, o in zip(inp.input_ids, out)]
        return processor.batch_decode(tr, skip_special_tokens=True)[0].strip()

assert glob.glob("data_samples/sketchy_sketch/*"), "data_samples 없음 -> zip 업로드/해제 먼저"
SKETCHES = sorted(f for f in glob.glob("data_samples/sketchy_sketch/*")
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
CATS = sorted(set(gt_of(f) for f in SKETCHES))
print(f"sketches={len(SKETCHES)}, categories={CATS}")


# ====== CELL 1 : TASK 1 — Captioning (스케치 설명 생성) ======
CAP_PROMPT = ("Describe this hand-drawn sketch in one sentence. "
              "Mention: the main object, line quality, and whether it looks complete.")
cap_results = []
for i, f in enumerate(SKETCHES):
    cap = ask(f, CAP_PROMPT, max_new=60)
    cap_results.append({"file": os.path.basename(f), "gt": gt_of(f), "caption": cap})
    if i < 8:
        print(f"[{gt_of(f):8s}] {cap}")
    if (i + 1) % 25 == 0: print(f"  captioning {i+1}/{len(SKETCHES)}")
json.dump(cap_results, open("task1_captioning.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n캡션 {len(cap_results)}건 저장 -> task1_captioning.json")


# ====== CELL 2 : TASK 2 — VQA (질문-답) ======
VQA_QUESTIONS = [
    ("animal",  "Is the main object an animal? Answer only 'yes' or 'no'."),
    ("count",   "How many separate objects are drawn? Answer with a single integer."),
    ("color",   "Is this drawing in color or black-and-white? Answer 'color' or 'bw'."),
]
SAMPLE = SKETCHES[::max(1, len(SKETCHES)//20)][:20]   # 20장 샘플
vqa_results = []
ANIMALS = {"cat", "cow", "horse", "duck", "penguin", "pig", "lizard"}
for f in SAMPLE:
    row = {"file": os.path.basename(f), "gt": gt_of(f)}
    for key, q in VQA_QUESTIONS:
        row[key] = ask(f, q, max_new=12)
    vqa_results.append(row)
    exp_animal = "yes" if gt_of(f) in ANIMALS else "no"
    print(f"[{row['gt']:8s}] animal={row['animal']:>4s}(정답 {exp_animal}) | count={row['count']:>3s} | color={row['color']}")
json.dump(vqa_results, open("task2_vqa.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
# 'animal' 질문 정확도(우리가 카테고리로 정답을 알 수 있음)
acc_animal = sum(("yes" in r["animal"].lower()) == (r["gt"] in ANIMALS) for r in vqa_results) / len(vqa_results)
print(f"\nVQA 'animal' 정확도: {acc_animal:.2f}  (저장: task2_vqa.json)")


# ====== CELL 3 : TASK 3 — Classification (라벨로 정답 비교) ======
# QuickDraw 미보유 -> 라벨 보유한 Sketchy 스케치로 closed-set 분류.
# (QuickDraw 쓰려면 SKETCHES/CATS만 그 데이터로 교체하면 됨)
opt = ", ".join(CATS)
CLS_PROMPT = f"Classify the object in this sketch. Choose exactly ONE word from: {opt}. Answer with only that word."
cls_results, correct = [], 0
for i, f in enumerate(SKETCHES):
    pred = ask(f, CLS_PROMPT, max_new=8)
    pl = re.sub(r"[^a-z ]", " ", pred.lower())
    hit = gt_of(f) in pl
    correct += hit
    cls_results.append({"file": os.path.basename(f), "gt": gt_of(f), "pred": pred, "correct": hit})
    if (i + 1) % 25 == 0: print(f"  classification {i+1}/{len(SKETCHES)}")
cls_acc = correct / len(SKETCHES)
print(f"\nClassification accuracy(closed-set, sketch): {cls_acc:.3f}")
# 카테고리별
from collections import defaultdict
bc = defaultdict(lambda: [0, 0])
for r in cls_results: bc[r["gt"]][0] += r["correct"]; bc[r["gt"]][1] += 1
for c, (o, t) in sorted(bc.items()): print(f"  {c:8s}: {o}/{t} = {o/t:.2f}")
json.dump(cls_results, open("task3_classification.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)


# ====== CELL 4 : TASK 4 — Retrieval (SigLIP 2 임베딩) ======
!pip install -q -U "transformers>=4.49"
from transformers import AutoProcessor as SPProc, AutoModel
from PIL import Image
import numpy as np

SIG = "google/siglip2-base-patch16-224"
sp = SPProc.from_pretrained(SIG)
sm = AutoModel.from_pretrained(SIG, torch_dtype=torch.float16).to("cuda").eval()

def _feat(out):
    # 버전에 따라 텐서/출력객체 둘 다 대응 (SigLIP은 pooler_output == 정렬 임베딩)
    if torch.is_tensor(out):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    raise RuntimeError(f"임베딩 못 찾음: {type(out)}")

@torch.no_grad()
def embed_images(paths, bs=16):
    embs = []
    for i in range(0, len(paths), bs):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i+bs]]
        inp = sp(images=ims, return_tensors="pt").to("cuda")
        e = _feat(sm.get_image_features(**inp))
        embs.append((e / e.norm(dim=-1, keepdim=True)).float().cpu())
    return torch.cat(embs)

@torch.no_grad()
def embed_texts(texts):
    inp = sp(text=texts, padding="max_length", return_tensors="pt").to("cuda")
    e = _feat(sm.get_text_features(**inp))
    return (e / e.norm(dim=-1, keepdim=True)).float().cpu()

labels = np.array([gt_of(f) for f in SKETCHES])
img_emb = embed_images(SKETCHES)
txt_emb = embed_texts([f"a sketch of a {c}" for c in CATS])

# 두 방향 공통 평가: R@1, R@5, mAP (정답 여러 개인 카테고리 검색이므로 mAP가 정석)
def eval_retrieval(sim, relevant, ks=(1, 5)):
    sim = sim.cpu().numpy(); order = (-sim).argsort(axis=1)   # 유사도 내림차순 랭킹
    res = {f"R@{k}": 0.0 for k in ks}; ap_sum = 0.0; valid = 0
    for q in range(sim.shape[0]):
        r = relevant[q][order[q]]                            # 랭킹 순서대로 정답여부
        if r.sum() == 0: continue
        valid += 1
        for k in ks:
            res[f"R@{k}"] += 1.0 if r[:k].any() else 0.0     # top-k 안에 정답 존재(hit-rate)
        cum = np.cumsum(r); prec = cum / (np.arange(len(r)) + 1)
        ap_sum += (prec * r).sum() / r.sum()                 # average precision
    for k in ks: res[f"R@{k}"] /= valid
    res["mAP"] = ap_sum / valid
    return {k: round(float(v), 3) for k, v in res.items()}

# (a) text -> image : 쿼리=카테고리 텍스트, 정답=그 카테고리 이미지
rel_t2i = np.array([[labels[j] == CATS[qi] for j in range(len(labels))] for qi in range(len(CATS))])
t2i = eval_retrieval(txt_emb @ img_emb.T, rel_t2i)
print("text->image :", t2i)

# (b) image -> image : 쿼리=각 스케치, 정답=같은 카테고리(자기 제외)
isim = img_emb @ img_emb.T; isim.fill_diagonal_(-1e4)
rel_i2i = (labels[:, None] == labels[None, :]); np.fill_diagonal(rel_i2i, False)
i2i = eval_retrieval(isim, rel_i2i)
print("image->image:", i2i)

# 샘플: text 쿼리별 top-5가 뭐였는지 (정성 확인용)
sim = txt_emb @ img_emb.T
for ci, c in enumerate(CATS):
    top5 = sim[ci].topk(5).indices.tolist()
    print(f"  '{c:8s}' top5 ->", [labels[j] for j in top5])

json.dump({"text2img": t2i, "img2img": i2i},
          open("task4_retrieval.json", "w", encoding="utf-8"), indent=2)


# ====== CELL 5 : 4대 task 종합 요약 (산출물) ======
summary = {
    "task1_captioning": f"{len(cap_results)} captions (task1_captioning.json)",
    "task2_vqa_animal_acc": round(acc_animal, 3),
    "task3_classification_acc": round(cls_acc, 3),
    "task4_text2img": t2i,    # {R@1, R@5, mAP}
    "task4_img2img": i2i,     # {R@1, R@5, mAP}
}
print(json.dumps(summary, ensure_ascii=False, indent=2))
with open("vlm_tasks_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
from google.colab import files
for fn in ["task1_captioning.json", "task2_vqa.json", "task3_classification.json",
           "task4_retrieval.json", "vlm_tasks_summary.json"]:
    files.download(fn)
