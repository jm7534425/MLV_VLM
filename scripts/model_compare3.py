# -*- coding: utf-8 -*-
"""3-way 공정 비교 — Qwen3-VL / LLaVA-OV-7B / InternVL2.5-8B, 한국어+영어 프롬프트.
언어 편향·프롬프트 편향을 분리해서 본다. 균형 24장, 스키마 강제."""
import json, re, base64, urllib.request, glob, os, sys, time
sys.path.insert(0, "backend"); sys.path.insert(0, "scripts")
from pipeline import EVAL_SYSTEM, RESPONSE_FORMAT, STAGE_LIST, CAPTION_PROMPT
from prompt_compare_en import EVAL_SYSTEM_EN, CAPTION_PROMPT_EN

MODELS = {
    "Qwen3-VL-8B": ("http://localhost:8000/v1/chat/completions", "Qwen/Qwen3-VL-8B-Instruct"),
    "LLaVA-OV-7B": ("http://localhost:8001/v1/chat/completions", "llava-hf/llava-onevision-qwen2-7b-ov-hf"),
    "InternVL2.5-8B": ("http://localhost:8002/v1/chat/completions", "OpenGVLab/InternVL2_5-8B"),
}
LANGS = {
    "KO": (CAPTION_PROMPT, EVAL_SYSTEM, "발달단계 판정. JSON으로만."),
    "EN": (CAPTION_PROMPT_EN, EVAL_SYSTEM_EN, "Judge the developmental stage. JSON only."),
}

def call(url, model, msgs, mt=500, rf=None):
    p={"model":model,"messages":msgs,"max_tokens":mt,"temperature":0}
    if rf: p["response_format"]=rf
    r=urllib.request.Request(url,data=json.dumps(p).encode(),headers={"Content-Type":"application/json"})
    t0=time.time(); resp=json.loads(urllib.request.urlopen(r,timeout=180).read())
    return resp["choices"][0]["message"]["content"], resp.get("usage",{}).get("total_tokens",0), int((time.time()-t0)*1000)
def img(b64,t,sysp=None):
    u={"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}},{"type":"text","text":t}]}
    return ([{"role":"system","content":sysp}] if sysp else [])+[u]
def eval_one(url,mid,b64,cap_p,sys_p,ask):
    cap,tk1,ms1=call(url,mid,img(b64,cap_p,None),200)
    txt,tk2,ms2=call(url,mid,img(b64,f"[caption]\n{cap}\n{ask}",sys_p),rf=RESPONSE_FORMAT)
    try: st=json.loads(txt).get("stage","?"); ok=st in STAGE_LIST
    except: st,ok="?",False
    return st,ok,tk1+tk2,ms1+ms2

def expected(a): return ("scribble_named" if a<=3 else "preschematic" if a<=6 else "schematic" if a<=8 else "dawning_realism" if a<=11 else "pseudo_naturalistic")
def idx(s): return STAGE_LIST.index(s) if s in STAGE_LIST else -9

# 어떤 모델이 살아있는지 확인
alive={}
for m,(url,_) in MODELS.items():
    try:
        urllib.request.urlopen(url.replace("/chat/completions","/models"),timeout=5); alive[m]=True
    except: alive[m]=False; print(f"[skip] {m} 서버 없음")
active={m:v for m,v in MODELS.items() if alive.get(m)}

tests=[(f,int(re.search(r"age(\d+)",f).group(1))) for f in sorted(glob.glob("data/samples/balanced/*.jpg"))]
res={(m,lang):{"dist":[],"tok":[],"ms":[],"valid":0} for m in active for lang in LANGS}
for f,age in tests:
    b64=base64.b64encode(open(f,"rb").read()).decode(); exp=expected(age)
    for m,(url,mid) in active.items():
        for lang,(cp,sp,ask) in LANGS.items():
            try:
                st,ok,tok,ms=eval_one(url,mid,b64,cp,sp,ask)
                if idx(st)>=0: res[(m,lang)]["dist"].append(abs(idx(st)-idx(exp)))
                res[(m,lang)]["tok"].append(tok); res[(m,lang)]["ms"].append(ms); res[(m,lang)]["valid"]+=ok
            except Exception: pass

def avg(l): return sum(l)/len(l) if l else 0
out=["# 3-way 공정 비교 — 모델 × 언어 (균형 24장)\n","스키마 강제. MAE=나이GT 거리↓, JSON유효↑.\n",
     "| 모델 | 언어 | MAE↓ | 지연ms↓ | 토큰↓ | JSON유효 |","|---|---|---|---|---|---|"]
for m in active:
    for lang in LANGS:
        r=res[(m,lang)]
        out.append(f"| {m} | {lang} | {avg(r['dist']):.2f} | {avg(r['ms']):.0f} | {avg(r['tok']):.0f} | {r['valid']}/{len(tests)} |")
open("outputs/model_compare_3way.md","w",encoding="utf-8").write("\n".join(out))
for m in active:
    for lang in LANGS:
        r=res[(m,lang)]
        print(f"{m:16s} {lang}: MAE {avg(r['dist']):.2f} | {avg(r['ms']):.0f}ms | valid {r['valid']}/{len(tests)}")
print("saved outputs/model_compare_3way.md")
