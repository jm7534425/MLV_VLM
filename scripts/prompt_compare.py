# -*- coding: utf-8 -*-
"""4주차 프롬프트 비교: ①단일호출 ②2단계(캡션→평가) ③CoT 체크리스트.
같은 그림셋에 3변형을 돌려 stage를 비교한다. (GT=나이 파생 약라벨)
"""
import json, re, base64, urllib.request, glob, os

API = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
STAGES = ["scribble_disordered","scribble_controlled","scribble_named",
          "preschematic","schematic","dawning_realism","pseudo_naturalistic"]

RUBRIC = """Lowenfeld 7단계로 아동 그림 발달단계를 판정한다.
[원칙] 성숙도(인식가능성·디테일·공간·비례·원근)로 판정. 사람=인물단서, 사물=인물단서 억지적용 금지.
[단계] scribble_disordered(무규칙 낙서) / scribble_controlled(제어된 반복선) / scribble_named(기초 기하형태·명명) /
preschematic(대상 인식되나 단순·기저선없이 떠있음; 사람=올챙이형) / schematic(기저선 등장·정밀 디테일) /
dawning_realism(겹침으로 깊이·개별 디테일) / pseudo_naturalistic(원근·명암·정확한 비례)"""

CAPTION_PROMPT = "이 그림에 보이는 것(객체·배치·신체부위·겹침·색)을 평이하게 묘사. 해석·전문용어 X."

def call(messages, max_tokens=600):
    p = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(API, data=json.dumps(p).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]

def img_msg(b64, text, system=None):
    u = {"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},{"type":"text","text":text}]}
    return ([{"role":"system","content":system}] if system else []) + [u]

def parse_stage(text):
    m = re.search(r'"stage"\s*:\s*"([a-z_]+)"', text)
    if m and m.group(1) in STAGES: return m.group(1)
    for s in STAGES:
        if s in text: return s
    return "?"

# --- 3 변형 ---
def v1_single(b64):
    sys = RUBRIC + '\n출력 JSON만: {"stage":"<코드>","evidence":"...","confidence":0.0~1.0}'
    return parse_stage(call(img_msg(b64, "발달단계 판정. JSON으로만.", sys)))

def v2_twostep(b64):
    cap = call(img_msg(b64, CAPTION_PROMPT, None), max_tokens=200)
    sys = RUBRIC + '\n출력 JSON만: {"stage":"<코드>","evidence":"...","confidence":0.0~1.0}'
    return parse_stage(call(img_msg(b64, f"[캡션]\n{cap}\n발달단계 판정. JSON으로만.", sys)))

def v3_cot(b64):
    sys = RUBRIC + """
먼저 observations를 항목별로 채운 뒤, 그것을 근거로 stage를 정하라.
출력 JSON만: {"observations":{"line_control":"...","symmetry":"...","baseline":"유/무","body_parts":"...","overlap_perspective":"...","color":"..."},"stage":"<코드>","evidence":"...","confidence":0.0~1.0}"""
    return parse_stage(call(img_msg(b64, "체크리스트 관찰 후 판정. JSON으로만.", sys)))

def expected(age):
    if age <= 3: return "scribble_named"
    if age <= 6: return "preschematic"
    if age <= 8: return "schematic"
    if age <= 11: return "dawning_realism"
    return "pseudo_naturalistic"

def idx(s): return STAGES.index(s) if s in STAGES else -9

# --- 테스트셋: kiddraw 4 + AI-Hub 4 ---
tests = []
for f in sorted(glob.glob("data/_kiddraw_examples/ex*.png")):
    age = int(re.search(r"age(\d+)", f).group(1))
    tests.append((f, age, "kiddraw"))
ah = {}
for f in glob.glob("data/samples/ai_hub_child/*.jpg"):
    cat = os.path.basename(f).split("__")[0]
    ah.setdefault(cat, f)
for f in list(ah.values())[:4]:
    age = int(os.path.basename(f).split("__")[1].split("_")[1])
    tests.append((f, age, "ai_hub"))

rows = ["# 프롬프트 변형 비교 (①단일 ②2단계 ③CoT체크리스트)\n",
        "GT=나이 파생 약라벨. 숫자는 |예측-기대| 단계거리(0=일치, 작을수록 좋음).\n",
        "| 그림 | 나이 | 기대(GT) | ①단일 | ②2단계 | ③CoT |",
        "|---|---|---|---|---|---|"]
err = {"v1":0,"v2":0,"v3":0}; n=0
for f, age, src in tests:
    b64 = base64.b64encode(open(f,"rb").read()).decode()
    exp = expected(age)
    v1, v2, v3 = v1_single(b64), v2_twostep(b64), v3_cot(b64)
    n += 1
    for k,v in [("v1",v1),("v2",v2),("v3",v3)]:
        if idx(v)>=0: err[k]+=abs(idx(v)-idx(exp))
    name = os.path.basename(f)[:22]
    def cell(v): return f"{v}({abs(idx(v)-idx(exp))})" if idx(v)>=0 else f"{v}(?)"
    rows.append(f"| {name} | {age} | {exp} | {cell(v1)} | {cell(v2)} | {cell(v3)} |")

rows.append(f"\n**평균 단계거리(MAE):** ①단일 {err['v1']/n:.2f} | ②2단계 {err['v2']/n:.2f} | ③CoT {err['v3']/n:.2f}  (작을수록 좋음)")
open("outputs/prompt_compare.md","w",encoding="utf-8").write("\n".join(rows))
print("done. MAE v1=%.2f v2=%.2f v3=%.2f"%(err['v1']/n,err['v2']/n,err['v3']/n))
