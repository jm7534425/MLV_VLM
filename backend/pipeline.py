# -*- coding: utf-8 -*-
"""파이프라인 — 방법 A(루브릭 CoT). scripts/test_method_a.py 로직 이식.
공통 1단계: caption(WHAT) → task별: evaluation(WHY) 등.
"""
import json, re
from vllm_client import vision_chat, resolve_model

PROMPT_VERSION = "method_a_v2"

# --- 캡션(WHAT): 그림 설명만, 평이하게 ---
CAPTION_PROMPT = ("이 아동 그림에 보이는 것(객체·배치·신체부위·겹침·색)을 "
                  "평이하게 한국어로 묘사하라. 해석이나 전문용어 없이 사실만.")

# --- evaluation(WHY): 7단계 루브릭 + 품질점수 ---
EVAL_SYSTEM = """너는 아동 그림의 발달단계를 판정하는 전문가다. Lowenfeld 7단계로 판정한다.

[원칙] 단계는 그림의 '성숙도'(인식가능성·디테일·공간·비례·원근)로 정한다.
- 사람 그림: 인물 단서(올챙이형/신체부위/비례) 우선.
- 나무·집·사물: 인물 단서를 억지 적용 말고 형태 인식가능성·디테일·공간(기저선/겹침)·원근으로 매핑.

[7단계 코드]
scribble_disordered: 통제 없는 무규칙 낙서, 식별 불가.
scribble_controlled: 제어된 반복 선, 표상 의도 없음.
scribble_named: 기초 기하형태 등장, 사후 명명, 매우 단순.
preschematic: 대상 인식되나 단순·불안정, 기저선 없이 떠 있음. (사람: 올챙이형)
schematic: 기저선 등장, 고정 도식 + 정밀 디테일, 비례 균형 시작.
dawning_realism: 기저선→면, 겹침으로 깊이, 개별 디테일, 색 변이.
pseudo_naturalistic: 원근·명암(3D), 비례·관절 정확, 질감.

[품질점수 1~5] object_recognizability, detail_level, proportion, completeness, spatial_organization

[출력] JSON만:
{"stage":"<코드>","evidence":"근거(간결히)","confidence":0.0~1.0,
 "quality_scores":{"object_recognizability":1,"detail_level":1,"proportion":1,"completeness":1,"spatial_organization":1}}
stage는 코드 문자열: scribble_disordered, scribble_controlled, scribble_named, preschematic, schematic, dawning_realism, pseudo_naturalistic
"""

STAGES = {"scribble_disordered", "scribble_controlled", "scribble_named",
          "preschematic", "schematic", "dawning_realism", "pseudo_naturalistic"}


def _parse_json(text):
    """모델 응답에서 JSON 추출(코드펜스·잡텍스트 대응)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no_json")
    return json.loads(m.group(0))


def make_caption(image_bytes, model_alias):
    model_id = resolve_model(model_alias)
    text, tokens, ms = vision_chat(image_bytes, CAPTION_PROMPT, model_id, max_tokens=200)
    return text.strip(), tokens, ms


def run_evaluation(image_bytes, model_alias, caption):
    """캡션을 참고로 발달단계+품질 판정. (caption은 호출자가 캐시에서 가져옴)"""
    model_id = resolve_model(model_alias)
    prompt = f"[캡션]\n{caption}\n\n이 그림의 발달단계를 판정하고 JSON으로만 답하라."
    text, tokens, ms = vision_chat(image_bytes, prompt, model_id,
                                   system=EVAL_SYSTEM, max_tokens=500)
    failure = None
    try:
        result = _parse_json(text)
        if result.get("stage") not in STAGES:
            failure = "parse_error"
        elif result.get("confidence", 1) < 0.4:
            failure = "low_confidence"
    except Exception:
        result, failure = None, "parse_error"
    return result, tokens, ms, failure
