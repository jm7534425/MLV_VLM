# -*- coding: utf-8 -*-
"""Ablation: 캡션 단계 유무 (caption-first가 판정 정확도에 기여하나).
앵커·원칙·스키마 동일. ① 캡션O(2단계) vs ② 캡션X(이미지만). 균형 24장."""
import json, re, base64, urllib.request, glob, os, sys
from collections import defaultdict
sys.path.insert(0, "backend")
from pipeline import EVAL_SYSTEM, RESPONSE_FORMAT, STAGE_LIST, CAPTION_PROMPT

API="http://localhost:8000/v1/chat/completions"; MODEL="Qwen/Qwen3-VL-8B-Instruct"

def raw(msgs, mt=500, rf=None):
    p={"model":MODEL,"messages":msgs,"max_tokens":mt,"temperature":0}
    if rf: p["response_format"]=rf
    r=urllib.request.Request(API,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=120).read())["choices"][0]["message"]["content"]
def img(b64,t,sys=None):
    u={"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},{"type":"text","text":t}]}
    return ([{"role":"system","content":sys}] if sys else [])+[u]
def stage(t):
    try: return json.loads(t).get("stage","?")
    except: return "?"

def with_caption(b64):   # ① 캡션 O (현행 2단계)
    cap=raw(img(b64,CAPTION_PROMPT,None),200)
    return stage(raw(img(b64,f"[캡션]\n{cap}\n발달단계 판정. JSON으로만.",EVAL_SYSTEM),rf=RESPONSE_FORMAT))
def no_caption(b64):     # ② 캡션 X (이미지만 바로)
    return stage(raw(img(b64,"발달단계 판정. JSON으로만.",EVAL_SYSTEM),rf=RESPONSE_FORMAT))

def expected(a): return ("scribble_named" if a<=3 else "preschematic" if a<=6 else "schematic" if a<=8 else "dawning_realism" if a<=11 else "pseudo_naturalistic")
def idx(s): return STAGE_LIST.index(s) if s in STAGE_LIST else -9

tests=[(f,int(re.search(r"age(\d+)",f).group(1))) for f in sorted(glob.glob("data/samples/balanced/*.jpg"))]
rows=["# Ablation: 캡션 단계 유무 (caption-first 효과)\n","앵커·원칙·스키마 동일. |거리| 낮을수록 좋음.\n",
      "| 그림(나이) | 기대 | 캡션O | 캡션X |","|---|---|---|---|"]
dC=[]; dN=[]; flip=0
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode()
    exp=expected(age); sC=with_caption(b64); sN=no_caption(b64)
    eC,eN=abs(idx(sC)-idx(exp)),abs(idx(sN)-idx(exp))
    if idx(sC)>=0: dC.append(eC)
    if idx(sN)>=0: dN.append(eN)
    if sC!=sN: flip+=1
    rows.append(f"| {os.path.basename(f).replace('.jpg','')[:18]} ({age}) | {exp} | {sC}({eC}) | {sN}({eN}) |")
def mae(d): return sum(d)/len(d) if d else 0
rows+=["\n## 종합\n","| | 캡션 O(2단계·현행) | 캡션 X(이미지만) |","|---|---|---|",
       f"| MAE | {mae(dC):.2f} | {mae(dN):.2f} |",
       f"| 판정이 서로 다른 그림 | {flip}/{len(tests)} |"]
open("outputs/ablation_caption.md","w",encoding="utf-8").write("\n".join(rows))
print("캡션O MAE %.2f | 캡션X MAE %.2f | 판정불일치 %d/%d"%(mae(dC),mae(dN),flip,len(tests)))
