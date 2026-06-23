# ============================================================
#  1주차 표 VLM 칸 채우기 (Google Colab, 무료 T4)
#  - Object abstraction = photo vs sketch 분류 정확도 격차
#  - failure case 수집 ("VLM 영향" 열 / 산출물)
#
#  사용법: 아래 블록을 Colab 셀에 하나씩 복사해 실행.
#  런타임 > 런타임 유형 변경 > GPU(T4) 선택 필수.
# ============================================================


# ====== CELL 1 : 패키지 설치 ======
# Qwen2.5-VL은 transformers>=4.49면 됨. (Qwen3-VL은 >=4.57, T4엔 8B 무거워 비권장)
!pip install -q "transformers>=4.49" accelerate qwen-vl-utils pillow


# ====== CELL 2 : 데이터 업로드 & 압축해제 ======
# 로컬의 data_samples.zip 을 선택해 업로드
from google.colab import files
up = files.upload()                 # data_samples.zip 선택
!unzip -o -q data_samples.zip
!echo "--- 폴더별 장수 ---"; for d in data_samples/*/; do echo "$d $(ls $d | wc -l)"; done
# (대안) 구글드라이브 사용 시:
# from google.colab import drive; drive.mount('/content/drive')
# !cp /content/drive/MyDrive/data_samples.zip . && unzip -o -q data_samples.zip


# ====== CELL 3 : 모델 로드 (T4에 맞는 3B) ======
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"     # T4 16GB에 안정적으로 적재
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, device_map="auto")
processor = AutoProcessor.from_pretrained(MODEL)
print("loaded:", MODEL)

def ask(image_path, prompt, max_new=24):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text",  "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(msgs)
    inputs = processor(text=[text], images=imgs, videos=vids,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# ====== CELL 4 : 분류 정확도 측정 (abstraction 격차) ======
import glob, os, re, json

# 파일명 "cat__n0212...jpg" -> gt="cat"
def gt_of(f): return os.path.basename(f).split("__")[0].lower()

# 정답 매칭(관대): 동의어/복수형 흡수
SYN = {"pear": ["pear"], "apple": ["apple"], "cat": ["cat", "kitten", "feline"],
       "cow": ["cow", "cattle", "bull", "ox"], "horse": ["horse", "pony"],
       "duck": ["duck", "duckling"], "penguin": ["penguin"], "pig": ["pig", "piglet", "hog"],
       "chair": ["chair", "seat", "stool"], "hat": ["hat", "cap"], "lizard": ["lizard", "reptile"]}
def is_correct(pred, gt):
    p = re.sub(r"[^a-z ]", " ", pred.lower())
    words = SYN.get(gt, [gt])
    return any(w in p for w in words) or gt in p

PROMPT = "What is the main object in this image? Answer with ONE common English noun only."

def run_group(folder):
    res = []
    files = sorted(f for f in glob.glob(folder + "/*")
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    for i, f in enumerate(files):
        try:
            ans = ask(f, PROMPT)
        except Exception as e:
            ans = f"ERR:{e}"
        ok = is_correct(ans, gt_of(f))
        res.append({"file": os.path.basename(f), "gt": gt_of(f), "pred": ans, "correct": ok})
        if (i + 1) % 20 == 0:
            print(f"  {folder}: {i+1}/{len(files)}")
    acc = sum(r["correct"] for r in res) / max(len(res), 1)
    return res, acc

print(">> photo 분류 중..."); photo_res, photo_acc = run_group("data_samples/sketchy_photo")
print(">> sketch 분류 중..."); sketch_res, sketch_acc = run_group("data_samples/sketchy_sketch")

print("\n===== Object abstraction (분류 정확도) =====")
print(f"photo  accuracy : {photo_acc:.3f}")
print(f"sketch accuracy : {sketch_acc:.3f}")
print(f"ABSTRACTION GAP : {photo_acc - sketch_acc:+.3f}  (사진 대비 스케치 정확도 하락폭)")


# ====== CELL 5 : failure case 수집 ("VLM 영향" 열 / 산출물) ======
fails = [r for r in sketch_res if not r["correct"]]
print(f"sketch 오인식: {len(fails)}/{len(sketch_res)}건")
print("\n--- 대표 failure 20건 (gt -> 모델 답) ---")
for r in fails[:20]:
    print(f"  [{r['gt']:8s}] -> '{r['pred']}'   ({r['file']})")

# 카테고리별 정확도(어떤 객체가 스케치에서 더 안 잡히나)
from collections import defaultdict
bycat = defaultdict(lambda: [0, 0])
for r in sketch_res:
    bycat[r["gt"]][0] += r["correct"]; bycat[r["gt"]][1] += 1
print("\n--- 카테고리별 sketch 정확도 ---")
for c, (ok, tot) in sorted(bycat.items()):
    print(f"  {c:8s}: {ok}/{tot} = {ok/tot:.2f}")


# ====== CELL 6 : (선택) captioning 샘플 — task 동작 확인 + 산출물 ======
CAP_PROMPT = "Describe this drawing in one sentence: object, line quality, completeness."
print("--- 캡션 샘플 5건 ---")
for f in sorted(glob.glob("data_samples/sketchy_sketch/*"))[:5]:
    print(f"[{gt_of(f)}] {ask(f, CAP_PROMPT, max_new=60)}")


# ====== CELL 7 : 결과 저장 & 다운로드 ======
summary = {
    "model": MODEL,
    "photo_acc": photo_acc, "sketch_acc": sketch_acc,
    "abstraction_gap": photo_acc - sketch_acc,
    "sketch_fail_rate": len(fails) / max(len(sketch_res), 1),
}
with open("vlm_table_results.json", "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "photo": photo_res, "sketch": sketch_res},
              f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
from google.colab import files as _f
_f.download("vlm_table_results.json")     # 로컬로 받아서 CV 결과와 합치기
