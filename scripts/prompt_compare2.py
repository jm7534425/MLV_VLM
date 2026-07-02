# -*- coding: utf-8 -*-
"""프롬프트 변형 비교 v2 — 균형 24장(나이2~13)에 ①단일 ②2단계 ③CoT 비교."""
import json, re, base64, urllib.request, glob, os
from collections import defaultdict

API="http://localhost:8000/v1/chat/completions"; MODEL="Qwen/Qwen3-VL-8B-Instruct"
STAGES=["scribble_disordered","scribble_controlled","scribble_named","preschematic","schematic","dawning_realism","pseudo_naturalistic"]
RUBRIC="""Lowenfeld 7단계로 아동 그림 발달단계를 판정한다.
[원칙] 성숙도(인식가능성·디테일·공간·비례·원근)로 판정. 사람=인물단서, 사물=인물단서 억지적용 금지.
[단계] scribble_disordered(무규칙 낙서) / scribble_controlled(제어된 반복선) / scribble_named(기초 기하형태·명명) /
preschematic(대상 인식되나 단순·기저선없이 떠있음; 사람=올챙이형) / schematic(기저선·정밀 디테일) /
dawning_realism(겹침으로 깊이·개별 디테일) / pseudo_naturalistic(원근·명암·정확한 비례)"""
CAP="이 그림에 보이는 것(객체·배치·신체부위·겹침·색)을 평이하게 묘사. 해석·전문용어 X."
JSON_OUT='\n출력 JSON만: {"stage":"<코드>","evidence":"...","confidence":0.0~1.0}'

def call(msgs, mt=600):
    p={"model":MODEL,"messages":msgs,"max_tokens":mt,"temperature":0}
    r=urllib.request.Request(API,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())["choices"][0]["message"]["content"]
def msg(b64,t,sys=None):
    u={"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},{"type":"text","text":t}]}
    return ([{"role":"system","content":sys}] if sys else [])+[u]
def pstage(t):
    m=re.search(r'"stage"\s*:\s*"([a-z_]+)"',t)
    if m and m.group(1) in STAGES: return m.group(1)
    for s in STAGES:
        if s in t: return s
    return "?"
def v1(b64): return pstage(call(msg(b64,"발달단계 판정. JSON으로만.",RUBRIC+JSON_OUT)))
def v2(b64):
    c=call(msg(b64,CAP,None),200)
    return pstage(call(msg(b64,f"[캡션]\n{c}\n발달단계 판정. JSON으로만.",RUBRIC+JSON_OUT)))
def v3(b64):
    sys=RUBRIC+'\n먼저 observations(line_control,symmetry,baseline,body_parts,overlap_perspective,color)를 채운 뒤 stage 결정.\n출력 JSON만: {"observations":{...},"stage":"<코드>","evidence":"...","confidence":0.0~1.0}'
    return pstage(call(msg(b64,"체크리스트 관찰 후 판정. JSON으로만.",sys)))

def expected(age):
    return ("scribble_named" if age<=3 else "preschematic" if age<=6 else
            "schematic" if age<=8 else "dawning_realism" if age<=11 else "pseudo_naturalistic")
def idx(s): return STAGES.index(s) if s in STAGES else -9

tests=[]
for f in sorted(glob.glob("data/samples/balanced/*.jpg")):
    age=int(re.search(r"age(\d+)",f).group(1)); tests.append((f,age))
tests.sort(key=lambda x:x[1])

rows=["# 프롬프트 변형 비교 v2 — 균형 24장 (나이 2~13)\n",
      "GT=나이 파생 약라벨. 셀=예측단계(|거리|). 거리 0=GT일치.\n",
      "| 그림(나이) | 기대(GT) | ①단일 | ②2단계 | ③CoT |","|---|---|---|---|---|"]
agg={"v1":[],"v2":[],"v3":[]}
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode()
    exp=expected(age); r1,r2,r3=v1(b64),v2(b64),v3(b64)
    for k,v in [("v1",r1),("v2",r2),("v3",r3)]:
        if idx(v)>=0: agg[k].append(abs(idx(v)-idx(exp)))
    def c(v): return f"{v}({abs(idx(v)-idx(exp))})" if idx(v)>=0 else f"{v}(?)"
    nm=os.path.basename(f).replace(".jpg","")[:20]
    rows.append(f"| {nm} ({age}) | {exp} | {c(r1)} | {c(r2)} | {c(r3)} |")

def stats(d):
    n=len(d); mae=sum(d)/n; exact=sum(x==0 for x in d)/n; adj=sum(x<=1 for x in d)/n
    return mae,exact,adj
rows.append("\n## 종합 (n=%d)\n"%len(tests))
rows.append("| 변형 | 평균거리(MAE)↓ | 정확일치↑ | ±1이내↑ |")
rows.append("|---|---|---|---|")
for k,nm in [("v1","① 단일"),("v2","② 2단계"),("v3","③ CoT")]:
    mae,ex,adj=stats(agg[k])
    rows.append(f"| {nm} | {mae:.2f} | {ex*100:.0f}% | {adj*100:.0f}% |")
open("outputs/prompt_compare_v2.md","w",encoding="utf-8").write("\n".join(rows))
for k,nm in [("v1","단일"),("v2","2단계"),("v3","CoT")]:
    mae,ex,adj=stats(agg[k]); print(f"{nm}: MAE {mae:.2f}, 일치 {ex*100:.0f}%, ±1 {adj*100:.0f}%")
print("saved outputs/prompt_compare_v2.md")
