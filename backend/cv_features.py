# -*- coding: utf-8 -*-
"""CV feature 모듈 (6주차, CPU only) — 코드로 잴 수 있는 그림 속성을 측정.
캡셔닝 파이프라인의 measured{} 산출 + grounding용 자연어 힌트 생성.
지표 로직은 1주차 measure_characteristics.py 재활용/단일이미지화."""
import numpy as np
import cv2

MAXDIM = 768

# HSV hue(0~180) → 한국어 색 이름
_HUE_NAMES = [
    (10, "빨강"), (25, "주황"), (35, "노랑"), (85, "초록"),
    (100, "청록"), (130, "파랑"), (160, "보라"), (180, "빨강"),
]


def _load(image):
    """bytes 또는 경로 → BGR ndarray (한글경로 안전)."""
    if isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    h, w = img.shape[:2]
    s = MAXDIM / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def _palette(img, hsv, nonwhite):
    """유채색 여부 + 주요 색 이름. 검정/회색도 감지."""
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    chromatic = (S > 60) & (V > 60) & nonwhite          # 뚜렷한 유채 픽셀
    is_color = float(chromatic.mean()) > 0.01
    names = []
    if is_color:
        hues = hsv[:, :, 0][chromatic]
        for lo_name in _dominant_hues(hues):
            if lo_name not in names:
                names.append(lo_name)
    # 검정(진한 선)·회색 감지
    dark = (V < 60) & nonwhite
    if dark.mean() > 0.005:
        names.append("검정")
    if not names:
        names = ["흑백"]
    return is_color, names[:4]


def _dominant_hues(hues, min_frac=0.12):
    """hue 배열 → 비중 큰 색 이름들(많은 순)."""
    if len(hues) == 0:
        return []
    names = {}
    for h in hues:
        for hi, name in _HUE_NAMES:
            if h <= hi:
                names[name] = names.get(name, 0) + 1
                break
    tot = sum(names.values())
    ranked = sorted(names.items(), key=lambda kv: -kv[1])
    return [n for n, c in ranked if c / tot >= min_frac]


# [제거됨] _line_tremor: 윤곽 고주파 변화가 '손떨림'과 '정교한 디테일'을 구분 못 함
# (정교한 AI-Hub 그림이 휘갈긴 낙서보다 높게 나옴) → 타당하지 않아 line_quality째로 삭제.


def measure(image):
    """단일 이미지 → measured{} (환각 불가한 확정값)."""
    img = _load(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = gray.shape
    nonwhite = np.any(img < 230, axis=2)

    ink_ratio = float(nonwhite.mean())
    edge_density = float((cv2.Canny(gray, 80, 160) > 0).mean())
    n_comp = int(cv2.connectedComponents((nonwhite * 255).astype(np.uint8))[0] - 1)
    is_color, palette = _palette(img, hsv, nonwhite)

    ys, xs = np.where(nonwhite)
    if len(xs) > 10:
        spread_x = float(xs.std() / W); spread_y = float(ys.std() / H)
        vcenter = float(ys.mean() / H)                 # 잉크 세로 무게중심(1=하단)
        q = np.array([nonwhite[:H // 2, :W // 2].sum(), nonwhite[:H // 2, W // 2:].sum(),
                      nonwhite[H // 2:, :W // 2].sum(), nonwhite[H // 2:, W // 2:].sum()], float)
        p = q / max(q.sum(), 1); p = p[p > 0]
        quad_entropy = float(-(p * np.log2(p)).sum())
    else:
        spread_x = spread_y = vcenter = quad_entropy = 0.0

    return {
        "is_color": is_color,
        "palette": palette,
        "ink_ratio": round(ink_ratio, 4),
        "edge_density": round(edge_density, 4),
        "n_components": n_comp,
        "layout": {"spread_x": round(spread_x, 3), "spread_y": round(spread_y, 3),
                   "vcenter": round(vcenter, 3), "quad_entropy": round(quad_entropy, 3)},
    }


def to_hints(m):
    """measured{} → VLM grounding용 자연어 한 줄(숫자 X). 해석형 필드 실험용."""
    color = "유채색(" + "·".join(c for c in m["palette"] if c != "검정") + ")" \
        if m["is_color"] else "흑백"
    dens = "선이 희소함" if m["ink_ratio"] < 0.08 else \
           "선이 빽빽함" if m["ink_ratio"] > 0.25 else "선 밀도 보통"
    n = m["n_components"]
    count = "요소가 거의 하나로 이어짐" if n <= 2 else \
            f"분리된 요소 약 {n}개" if n <= 15 else "잔조각이 매우 많음"
    lay = m["layout"]
    pos = "화면 하단에 몰림(기저선 암시)" if lay["vcenter"] > 0.6 else \
          "화면 상단에 몰림" if lay["vcenter"] < 0.4 else "화면 중앙 분포"
    spread = "고르게 퍼짐" if lay["quad_entropy"] > 1.8 else "한쪽에 치우침"
    return f"{color}, {dens}, {count}, {pos}·{spread}."


if __name__ == "__main__":
    import sys, json
    m = measure(sys.argv[1])
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print("\n[hints]", to_hints(m))
