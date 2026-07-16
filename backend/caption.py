# -*- coding: utf-8 -*-
"""구조화 캡션 (6주차 2단계) — VLM(Qwen)이 '의미' 필드만 JSON으로 생성.
색·잉크량·요소수 등 측정값은 cv_features(measured{})가 담당하므로 여기서 안 물어봄.
병합(measured{}+semantic{})은 3단계."""
import json
import cv_features
from vllm_client import vision_chat, resolve_model

CAPTION_PROMPT_VERSION = "caption_v3"   # v3: line_quality 폐기(간단 CV로 측정 불가)

# semantic{}: 순수 의미(object·scene·style·body_parts) + 해석형(spatial·completeness)
# ※ line_quality(선 떨림/안정)는 폐기 — VLM은 앵무새, 코드 metric은 디테일과 떨림을 구분 못 함(둘 다 실패).
CAPTION_SYSTEM = """너는 아동 그림을 사실적으로 묘사하는 캡셔너다. 보이는 것만, 평가·판정 없이 기술한다.
색·잉크량·요소 개수는 별도 코드가 측정하므로 언급하지 말고, 아래 항목에만 집중하라.

[필드]
objects: 식별되는 사물/대상 이름 목록 (예: 나무, 집, 고양이). 식별 불가면 빈 배열.
body_parts: 사람·동물이면 보이는 신체부위 (머리·몸통·팔·다리 등). 없으면 빈 배열.
scene: 전체 장면을 한 문장으로 (무엇이 어떻게 있는지).
style: 그림체를 짧게 (예: 단순 선화, 도식적, 사실적 시도).
spatial_layout: 요소 배치 (예: 기저선 위 나란히 / 허공에 떠 있음 / 겹쳐 깊이 표현).
completeness: 핵심 부위의 있음/누락 (예: 눈·코·입 다 있음 / 팔 누락).

[출력] JSON만:
{"objects":[],"body_parts":[],"scene":"","style":"","spatial_layout":"","completeness":""}"""

CAPTION_USER = "이 아동 그림을 위 스키마대로 묘사하라. JSON으로만 답하라."
# grounding: 코드 측정값을 자연어로 참고 제공 (spatial_layout 구체화에 도움 — Phase4 확인)
CAPTION_USER_GROUNDED = (CAPTION_USER +
    "\n[코드 측정 참고 — spatial_layout 판단에 활용] {hints}")

SEM_KEYS = ["objects", "body_parts", "scene", "style",
            "spatial_layout", "completeness"]        # line_quality 폐기(v3)

# response_format용 스키마 — 깨진 JSON·필드 누락 원천 차단 (4주차 패턴)
CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
        "body_parts": {"type": "array", "items": {"type": "string"}, "maxItems": 15},
        "scene": {"type": "string", "maxLength": 200},
        "style": {"type": "string", "maxLength": 100},
        "spatial_layout": {"type": "string", "maxLength": 150},
        "completeness": {"type": "string", "maxLength": 150},
    },
    "required": SEM_KEYS,
    "additionalProperties": False,
}
CAPTION_RF = {"type": "json_schema",
              "json_schema": {"name": "caption", "schema": CAPTION_SCHEMA}}


def make_structured_caption(image_bytes, model="qwen3-vl-8b", hints=None):
    """이미지 → semantic{} (6필드), tokens, latency_ms. 실패 시 (None, ..., err).
    hints: 코드 측정 자연어 힌트(grounding). 주면 프롬프트에 참고로 첨부.
    ※ 일부 vLLM은 json_schema 강제가 느슨 → 파싱 실패 시 온도 올려 1회 재시도."""
    user = CAPTION_USER_GROUNDED.format(hints=hints) if hints else CAPTION_USER
    mid = resolve_model(model)
    total_ms, tokens, last = 0, {}, "parse_error"
    for temp in (0.0, 0.4):
        text, tokens, ms = vision_chat(
            image_bytes, user, mid, system=CAPTION_SYSTEM,
            max_tokens=500, temperature=temp, response_format=CAPTION_RF)
        total_ms += ms
        try:
            sem = json.loads(text)
            for k in SEM_KEYS:
                sem.setdefault(k, [] if k in ("objects", "body_parts") else "")
            return sem, tokens, total_ms, None
        except Exception as e:
            last = f"parse_error: {e}"
    return None, tokens, total_ms, last


def cross_check(sem, measured):
    """VLM 의미 필드 ↔ 코드 측정값 충돌 감지 (환각 플래그)."""
    flags = []
    # 색: VLM scene이 유채색 언급했는데 코드는 흑백 → 플래그
    if not measured["is_color"] and any(c in sem.get("scene", "")
                                        for c in ("빨강", "파랑", "초록", "노랑", "알록달록", "컬러")):
        flags.append("color_mismatch: VLM=유채 언급 but 측정=흑백")
    return flags


def build_caption(image_bytes, model="qwen3-vl-8b"):
    """최종 구조화 캡션 = VLM semantic{} + 코드 measured{} + flags.
    grounding 기본 on (spatial_layout 개선, Phase4 확인)."""
    measured = cv_features.measure(image_bytes)
    hints = cv_features.to_hints(measured)          # grounding: 해석형 필드 참고
    sem, tokens, ms, err = make_structured_caption(image_bytes, model, hints=hints)
    if err:
        return None, tokens, ms, err
    # 리스트 필드 클린업: 공백 strip + 빈 항목 제거 (" fences" 같은 아티팩트)
    for k in ("objects", "body_parts"):
        sem[k] = [s.strip() for s in sem.get(k, []) if s and s.strip()]
    obj = {**{k: sem[k] for k in SEM_KEYS},          # VLM 필드
           "measured": measured,                     # 코드(측정 확정값)
           "flags": cross_check(sem, measured)}
    return obj, tokens, ms, None


def caption_to_text(obj):
    """구조화 캡션 → 평가/검색이 쓰는 WHAT 텍스트로 직렬화."""
    parts = []
    if obj.get("scene"): parts.append(obj["scene"])
    if obj.get("objects"): parts.append("객체: " + ", ".join(obj["objects"]))
    if obj.get("body_parts"): parts.append("신체부위: " + ", ".join(obj["body_parts"]))
    if obj.get("spatial_layout"): parts.append("배치: " + obj["spatial_layout"])
    if obj.get("completeness"): parts.append("완성도: " + obj["completeness"])
    return " / ".join(parts)


if __name__ == "__main__":
    import sys
    img = open(sys.argv[1], "rb").read()
    obj, tk, ms, err = build_caption(img)
    print(json.dumps(obj, ensure_ascii=False, indent=2) if obj else f"ERR {err}")
    print(f"[{ms}ms, tokens={tk}]")
