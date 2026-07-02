# -*- coding: utf-8 -*-
"""모델 비교 — Qwen3-VL(8000) vs LLaVA-OneVision-7B(8001).
같은 프롬프트(EVAL_SYSTEM)+스키마강제로 균형 24장 평가. 정확도·지연·토큰 비교."""
import json, re, base64, urllib.request, glob, os, sys, time
sys.path.insert(0, "backend")
from pipeline import EVAL_SYSTEM, RESPONSE_FORMAT, STAGE_LIST, CAPTION_PROMPT

MODELS = {
    "Qwen3-VL-8B":   ("http://localhost:8000/v1/chat/completions", "Qwen/Qwen3-VL-8B-Instruct"),
    "LLaVA-OV-7B":   ("http://localhost:8001/v1/chat/completions", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
}

def call(url, model, msgs, mt=500, rf=None):
    p={"model":model,"messages":msgs,"max_tokens":mt,"temperature":0}
    if rf: p["response_format"]=rf
    r=urllib.request.Request(url,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    resp=json.loads(urllib.request.urlopen(r,timeout=180).read())
    ms=int((time.time()-t0)*1000)
    return resp["choices"][0]["message"]["content"], resp.get("usage",{}).get("total_tokens",0), ms
def img(b64,t,sys=None):
    u={"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},{"type":"text","text":t}]}
    return ([{"role":"system","content":sys}] if sys else [])+[u]

def eval_one(url, model, b64):
    cap,tk1,ms1 = call(url,model,img(b64,CAPTION_PROMPT,None),200)
    txt,tk2,ms2 = call(url,model,img(b64,f"[캡션]\n{cap}\n발달단계 판정. JSON으로만.",EVAL_SYSTEM),rf=RESPONSE_FORMAT)
    try: stage=json.loads(txt).get("stage","?"); ok=stage in STAGE_LIST
    except: stage,ok="?",False
    return stage, ok, tk1+tk2, ms1+ms2

def expected(a): return ("scribble_named" if a<=3 else "preschematic" if a<=6 else "schematic" if a<=8 else "dawning_realism" if a<=11 else "pseudo_naturalistic")
def idx(s): return STAGE_LIST.index(s) if s in STAGE_LIST else -9

tests=[(f,int(re.search(r"age(\d+)",f).group(1))) for f in sorted(glob.glob("data/samples/balanced/*.jpg"))]
res={m:{"dist":[],"tok":[],"ms":[],"valid":0} for m in MODELS}
detail=["| 그림(나이) | 기대 | "+" | ".join(MODELS)+" |","|"+"---|"*(2+len(MODELS))]
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode(); exp=expected(age)
    cells=[]
    for m,(url,mid) in MODELS.items():
        try:
            st,ok,tok,ms=eval_one(url,mid,b64)
            if idx(st)>=0: res[m]["dist"].append(abs(idx(st)-idx(exp)))
            res[m]["tok"].append(tok); res[m]["ms"].append(ms); res[m]["valid"]+=ok
            cells.append(f"{st}({abs(idx(st)-idx(exp)) if idx(st)>=0 else '?'})")
        except Exception as e:
            cells.append(f"ERR")
    detail.append(f"| {os.path.basename(f).replace('.jpg','')[:16]} ({age}) | {exp} | "+" | ".join(cells)+" |")

def avg(l): return sum(l)/len(l) if l else 0
out=["# 모델 비교 — Qwen3-VL vs LLaVA-OV-7B (균형 24장)\n","같은 프롬프트·스키마강제. MAE=나이GT 거리(낮을수록↑).\n",
     "## 종합\n","| 모델 | MAE↓ | 평균지연(ms)↓ | 평균토큰↓ | JSON유효 |","|---|---|---|---|---|"]
for m in MODELS:
    r=res[m]; out.append(f"| {m} | {avg(r['dist']):.2f} | {avg(r['ms']):.0f} | {avg(r['tok']):.0f} | {r['valid']}/{len(tests)} |")
out+=["\n## 그림별\n"]+detail
open("outputs/model_compare.md","w",encoding="utf-8").write("\n".join(out))
for m in MODELS:
    r=res[m]; print(f"{m}: MAE {avg(r['dist']):.2f} | {avg(r['ms']):.0f}ms | {avg(r['tok']):.0f}tok | valid {r['valid']}/{len(tests)}")
print("saved outputs/model_compare.md")
