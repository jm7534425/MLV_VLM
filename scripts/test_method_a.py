# -*- coding: utf-8 -*-
import json, urllib.request, base64, glob, os

API = 'http://localhost:8000/v1/chat/completions'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'

SYSTEM = """너는 아동 그림의 발달단계를 판정하는 전문가다. Lowenfeld 7단계로 판정한다.

[중요 원칙]
- 단계는 그림의 '성숙도'(대상 인식가능성·디테일·공간표현·비례·원근)로 정한다.
- 사람 그림: 인물 단서(올챙이형/신체부위/비례)를 우선 적용.
- 나무·집·사물: 인물 단서(머리·몸통·신체부위)를 억지로 적용하지 마라. 대신 형태 인식가능성·디테일·공간(기저선/겹침)·원근으로 같은 축에 매핑.

[7단계 — 코드: 공통 성숙도 단서 (사람 특화)]
scribble_disordered: 통제 없는 무규칙 낙서, 대상 식별 불가.
scribble_controlled: 제어된 반복 선(곡선·지그재그), 아직 표상 의도 없음.
scribble_named: 기초 기하형태 등장, 우연한 형태에 명명, 매우 단순.
preschematic: 대상은 인식되나 단순·불안정, 기저선 없이 떠 있음, 과장 비율. (사람: 올챙이형, 머리+팔다리)
schematic: 기저선 등장, 고정 도식 + 정밀 디테일(집:창살·문손잡이·굴뚝 / 나무:기둥+가지+잎 / 사람:머리카락·손가락), 비례 균형 시작.
dawning_realism: 기저선→면, 겹침으로 깊이, 개별·풍부한 디테일, 색 변이 인식.
pseudo_naturalistic: 원근·명암(3D), 비례·구조 정확, 질감 묘사.

[절차]
1) caption: 보이는 것만 평이하게 묘사 (해석·전문용어 금지)
2) evidence: 위 단서와 대조한 근거 (사물엔 인물 단서 적용 금지)
3) quality_scores: 아래 5개 품질을 1~5점(1=매우 미흡, 5=우수)으로 채점.
   - object_recognizability: 객체를 알아볼 수 있는 정도
   - detail_level: 세부 묘사 수준
   - proportion: 비례 정확도
   - completeness: 완성도(누락 없음)
   - spatial_organization: 공간 구성(기저선·배치·깊이)

[출력 형식] JSON만 출력하라:
{"caption":"...", "stage":"<코드>", "evidence":"...", "confidence":0.0~1.0,
 "quality_scores":{"object_recognizability":1, "detail_level":1, "proportion":1, "completeness":1, "spatial_organization":1}}
stage는 반드시 다음 코드 문자열 중 하나: scribble_disordered, scribble_controlled, scribble_named, preschematic, schematic, dawning_realism, pseudo_naturalistic
"""

def judge(img_path):
    b64 = base64.b64encode(open(img_path,'rb').read()).decode()
    payload = {'model': MODEL, 'temperature': 0, 'max_tokens': 500,
        'messages': [
            {'role':'system','content':SYSTEM},
            {'role':'user','content':[
                {'type':'image_url','image_url':{'url':f'data:image/png;base64,{b64}'}},
                {'type':'text','text':'이 그림의 발달단계를 판정하고 JSON으로만 답하라.'}]}]}
    req = urllib.request.Request(API, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())['choices'][0]['message']['content']

results = []
for f in sorted(glob.glob('data/_kiddraw_examples/ex*.png')):
    raw = judge(f)
    results.append(f"### {os.path.basename(f)}\n{raw}\n")
open('outputs/method_a_full_schema.md','w',encoding='utf-8').write(
    "# 방법 A — 전체 출력 스키마 테스트 (stage + quality_scores, 카테고리 힌트 없음)\n\n" + "\n".join(results))
print("done:", len(results), "images")
