# -*- coding: utf-8 -*-
"""앵커 추가 효과 검증 — 균형 24장에 '앵커 포함' 프롬프트(단일/2단계)로 재측정.
backend/pipeline.py의 실제 EVAL_SYSTEM·CAPTION_PROMPT를 그대로 import해서 사용.
앵커 전 결과(no-anchor): 단일 1.17 / 2단계 1.08 (prompt_compare_v2.md)
"""
import json, re, base64, urllib.request, glob, os, sys
sys.path.insert(0, "backend")
from pipeline import EVAL_SYSTEM, CAPTION_PROMPT  # 앵커 포함 실제 프롬프트

API="http://localhost:8000/v1/chat/completions"; MODEL="Qwen/Qwen3-VL-8B-Instruct"
STAGES=["scribble_disordered","scribble_controlled","scribble_named","preschematic","schematic","dawning_realism","pseudo_naturalistic"]

def call(msgs,mt=600):
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
def v_single(b64): return pstage(call(msg(b64,"발달단계 판정. JSON으로만.",EVAL_SYSTEM)))
def v_two(b64):
    c=call(msg(b64,CAPTION_PROMPT,None),200)
    return pstage(call(msg(b64,f"[캡션]\n{c}\n발달단계 판정. JSON으로만.",EVAL_SYSTEM)))

def expected(a): return ("scribble_named" if a<=3 else "preschematic" if a<=6 else "schematic" if a<=8 else "dawning_realism" if a<=11 else "pseudo_naturalistic")
def idx(s): return STAGES.index(s) if s in STAGES else -9

tests=[]
for f in sorted(glob.glob("data/samples/balanced/*.jpg")):
    tests.append((f,int(re.search(r"age(\d+)",f).group(1))))
tests.sort(key=lambda x:x[1])

rows=["# 앵커 추가 효과 — 균형 24장 재측정 (앵커 포함 프롬프트)\n",
      "비교: 앵커 전(no-anchor) 단일 1.17 / 2단계 1.08\n",
      "| 그림(나이) | 기대 | 단일(앵커) | 2단계(앵커) |","|---|---|---|---|"]
d1=[]; d2=[]
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode()
    exp=expected(age); s1=v_single(b64); s2=v_two(b64)
    if idx(s1)>=0: d1.append(abs(idx(s1)-idx(exp)))
    if idx(s2)>=0: d2.append(abs(idx(s2)-idx(exp)))
    def c(v): return f"{v}({abs(idx(v)-idx(exp))})" if idx(v)>=0 else f"{v}(?)"
    rows.append(f"| {os.path.basename(f).replace('.jpg','')[:20]} ({age}) | {exp} | {c(s1)} | {c(s2)} |")

def st(d): n=len(d); return sum(d)/n, sum(x==0 for x in d)/n, sum(x<=1 for x in d)/n
m1=st(d1); m2=st(d2)
rows+=["\n## 종합 (n=%d)\n"%len(tests),
       "| 프롬프트 | MAE↓ | 정확일치 | ±1이내 |","|---|---|---|---|",
       f"| 단일 (앵커 전) | 1.17 | 21% | 67% |",
       f"| **단일 (앵커 후)** | {m1[0]:.2f} | {m1[1]*100:.0f}% | {m1[2]*100:.0f}% |",
       f"| 2단계 (앵커 전) | 1.08 | 25% | 71% |",
       f"| **2단계 (앵커 후)** | {m2[0]:.2f} | {m2[1]*100:.0f}% | {m2[2]*100:.0f}% |"]
open("outputs/prompt_compare_anchor.md","w",encoding="utf-8").write("\n".join(rows))
print("앵커후 단일: MAE %.2f 일치 %.0f%% ±1 %.0f%%"%(m1[0],m1[1]*100,m1[2]*100))
print("앵커후 2단계: MAE %.2f 일치 %.0f%% ±1 %.0f%%"%(m2[0],m2[1]*100,m2[2]*100))
print("(앵커전: 단일 1.17/21/67, 2단계 1.08/25/71)")
