# -*- coding: utf-8 -*-
"""세부 ablation: '원칙 문구(사람/사물 구분)' 유무 비교. 균형 24장, 단일호출+스키마강제."""
import json, re, base64, urllib.request, glob, os, sys
sys.path.insert(0, "backend")
from pipeline import EVAL_SYSTEM, RESPONSE_FORMAT, STAGE_LIST

API="http://localhost:8000/v1/chat/completions"; MODEL="Qwen/Qwen3-VL-8B-Instruct"

# 원칙 블록 제거 버전
ABLATED = re.sub(r"\[원칙\].*?매핑\.\n", "", EVAL_SYSTEM, flags=re.DOTALL)
assert "인물 단서" not in ABLATED, "원칙 제거 실패"

def call(b64, sysprompt):
    p={"model":MODEL,"messages":[{"role":"system","content":sysprompt},
        {"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},
        {"type":"text","text":"발달단계 판정. JSON으로만."}]}],
       "max_tokens":500,"temperature":0,"response_format":RESPONSE_FORMAT}
    r=urllib.request.Request(API,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t=json.loads(urllib.request.urlopen(r,timeout=120).read())["choices"][0]["message"]["content"]
    try: return json.loads(t).get("stage","?")
    except: return "?"

def expected(a): return ("scribble_named" if a<=3 else "preschematic" if a<=6 else "schematic" if a<=8 else "dawning_realism" if a<=11 else "pseudo_naturalistic")
def idx(s): return STAGE_LIST.index(s) if s in STAGE_LIST else -9
def cat(f):
    b=os.path.basename(f)
    if "person" in b: return "person"
    if "house" in b or "tree" in b: return "object"
    return "aihub"

tests=[(f,int(re.search(r"age(\d+)",f).group(1))) for f in sorted(glob.glob("data/samples/balanced/*.jpg"))]
rows=["# Ablation: 원칙 문구(사람/사물 구분) 유무\n","단일호출+스키마강제. |거리| (낮을수록 좋음).\n",
      "| 그림(나이) | cat | 기대 | 원칙O | 원칙X |","|---|---|---|---|---|"]
from collections import defaultdict
dO=[]; dX=[]; catO=defaultdict(list); catX=defaultdict(list)
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode()
    exp=expected(age); c=cat(f)
    sO=call(b64,EVAL_SYSTEM); sX=call(b64,ABLATED)
    eO,eX=abs(idx(sO)-idx(exp)),abs(idx(sX)-idx(exp))
    if idx(sO)>=0: dO.append(eO); catO[c].append(eO)
    if idx(sX)>=0: dX.append(eX); catX[c].append(eX)
    rows.append(f"| {os.path.basename(f).replace('.jpg','')[:18]} ({age}) | {c} | {exp} | {sO}({eO}) | {sX}({eX}) |")

def mae(d): return sum(d)/len(d) if d else 0
rows+=["\n## 종합\n","| | 원칙 O(현행) | 원칙 X |","|---|---|---|",
       f"| 전체 MAE | {mae(dO):.2f} | {mae(dX):.2f} |",
       f"| person MAE | {mae(catO['person']):.2f} | {mae(catX['person']):.2f} |",
       f"| object(집/나무) MAE | {mae(catO['object']):.2f} | {mae(catX['object']):.2f} |",
       f"| aihub MAE | {mae(catO['aihub']):.2f} | {mae(catX['aihub']):.2f} |"]
open("outputs/ablation_principle.md","w",encoding="utf-8").write("\n".join(rows))
print("전체 MAE  원칙O %.2f | 원칙X %.2f"%(mae(dO),mae(dX)))
print("object    원칙O %.2f | 원칙X %.2f"%(mae(catO['object']),mae(catX['object'])))
print("person    원칙O %.2f | 원칙X %.2f"%(mae(catO['person']),mae(catX['person'])))
