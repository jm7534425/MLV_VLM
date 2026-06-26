# -*- coding: utf-8 -*-
import json, urllib.request, base64, glob, os, random
import sys
sys.path.insert(0, 'scripts')
# test_method_a의 SYSTEM 프롬프트 재사용 (모듈 import 시 루프 실행 방지 위해 직접 읽기)
src = open('scripts/test_method_a.py', encoding='utf-8').read()
SYSTEM = src.split('SYSTEM = """',1)[1].split('"""',1)[0]

API='http://localhost:8000/v1/chat/completions'; MODEL='Qwen/Qwen3-VL-8B-Instruct'
def judge(img):
    b64=base64.b64encode(open(img,'rb').read()).decode()
    p={'model':MODEL,'temperature':0,'max_tokens':500,'messages':[
        {'role':'system','content':SYSTEM},
        {'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/png;base64,{b64}'}},
        {'type':'text','text':'이 그림의 발달단계를 판정하고 JSON으로만 답하라.'}]}]}
    req=urllib.request.Request(API,data=json.dumps(p).encode('utf-8'),headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req,timeout=120).read())['choices'][0]['message']['content']

random.seed(1)
fs=glob.glob('data/samples/ai_hub_child/*.jpg')
by_cat={}
for f in fs: by_cat.setdefault(os.path.basename(f).split('__')[0],[]).append(f)
picks=[random.choice(v) for v in by_cat.values()]

out=["# 방법 A — AI-Hub(도메인 앵커) 입력 테스트\n"]
for f in picks:
    # 파일명: cat__cat_age_gender_id.jpg
    parts=os.path.basename(f).split('__')[1].split('_')
    meta=f"카테고리={parts[0]}, 나이={parts[1]}, 성별={parts[2]}"
    out.append(f"### {os.path.basename(f)}\n_{meta}_\n\n{judge(f)}\n")
open('outputs/method_a_aihub.md','w',encoding='utf-8').write("\n".join(out))
print("done:", len(picks))
