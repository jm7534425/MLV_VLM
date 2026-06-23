# -*- coding: utf-8 -*-
"""
drawing-specific visual characteristic 수치화 (CPU only)
8개 특성 proxy를 그룹별로 측정 -> 비교표(markdown) 생성.
"""
import os, glob, math
import numpy as np
import cv2

GROUPS = {
    "sketchy_photo":  "data/samples/sketchy_photo",   # 모달리티: 사진
    "sketchy_sketch": "data/samples/sketchy_sketch",  # 모달리티: 스케치 / 획: 성인
    "sketchyscene":   "data/samples/sketchyscene",    # 공간: 스케치 장면
    "coco":           "data/samples/coco",            # 공간: 사진 장면
    "ai_hub_child":   "data/samples/ai_hub_child",    # 획: 아동
}
MAXDIM = 768

def load(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)  # 한글경로 안전
    if img is None: return None
    h, w = img.shape[:2]
    s = MAXDIM / max(h, w)
    if s < 1: img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def metrics(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, W = gray.shape
    npx = H * W
    # --- 모달리티 proxies ---
    nonwhite = np.any(img < 230, axis=2)            # 비배경(흰색 아님)
    ink_ratio = nonwhite.mean()                      # sparse line (낮을수록 희소)
    bg_ratio = 1 - ink_ratio                         # 배경 없음(낮을수록 배경 적음)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = (edges > 0).mean()                # texture/edge
    sat = hsv[:, :, 1].mean() / 255.0                # 색(채도)
    # 연결요소 수(파편화=completeness)
    fg = (nonwhite * 255).astype(np.uint8)
    n_comp, _ = cv2.connectedComponents(fg)
    n_comp = n_comp - 1
    # --- 공간(spatial layout) proxies ---
    ys, xs = np.where(nonwhite)
    if len(xs) > 10:
        spread_x = xs.std() / W
        spread_y = ys.std() / H
        # 4분면 잉크 분포 엔트로피(고르게 퍼질수록 큼)
        q = np.zeros(4)
        q[0] = nonwhite[:H//2, :W//2].sum(); q[1] = nonwhite[:H//2, W//2:].sum()
        q[2] = nonwhite[H//2:, :W//2].sum(); q[3] = nonwhite[H//2:, W//2:].sum()
        p = q / max(q.sum(), 1); p = p[p > 0]
        quad_entropy = -(p * np.log2(p)).sum()
    else:
        spread_x = spread_y = quad_entropy = 0
    # --- 획 특성(stroke) proxies: arc-length 정규화 + 고주파 떨림 ---
    # 각 윤곽선을 길이로 N등분 리샘플 → 크기/해상도 무관하게 "떨림"만 비교
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    N = 256          # 리샘플 점수(고정) → 스케일 불변
    hf_list, amp_list = [], []
    for c in cnts:
        if len(c) < 60: continue
        pts = c[:, 0, :].astype(np.float64)
        seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
        L = np.concatenate([[0], np.cumsum(seg)])
        if L[-1] < 80: continue                          # 너무 작은 윤곽 제외
        uu = np.linspace(0, L[-1], N)
        rx = np.interp(uu, L, pts[:, 0]); ry = np.interp(uu, L, pts[:, 1])
        ang = np.unwrap(np.arctan2(np.gradient(ry), np.gradient(rx)))
        k = 13; sm = np.convolve(ang, np.ones(k) / k, mode="same")
        hf = ang - sm                                    # 저주파(전체모양) 제거 → 고주파 떨림만
        hf_list.append(hf.std())                         # 떨림 진폭
        amp_list.append(np.abs(np.diff(ang, 2)).mean())  # 방향 급변(미세 떨림)
    hf_tremor = float(np.mean(hf_list)) if hf_list else 0.0
    turn_amp = float(np.mean(amp_list)) if amp_list else 0.0
    return dict(ink_ratio=ink_ratio, bg_ratio=bg_ratio, edge_density=edge_density,
               sat=sat, n_comp=n_comp,
               spread_x=spread_x, spread_y=spread_y, quad_entropy=quad_entropy,
               hf_tremor=hf_tremor, turn_amp=turn_amp)

def aggregate(folder):
    files = [f for f in glob.glob(folder + "/*") if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    acc = {}
    n = 0
    for f in files:
        img = load(f)
        if img is None: continue
        m = metrics(img); n += 1
        for k, v in m.items(): acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}, n

def fmt(x):
    return f"{x:.4f}" if abs(x) < 10 else f"{x:.1f}"

def main():
    print("측정 중...")
    R = {}
    for g, folder in GROUPS.items():
        R[g], n = aggregate(folder)
        print(f"  {g}: n={n}")
    lines = []
    lines.append("# drawing-specific visual characteristic — 측정 결과 (CPU proxy)\n")
    lines.append(f"_그룹별 평균값. 이미지 max {MAXDIM}px 리사이즈 후 측정._\n")

    # A. 모달리티: 사진 vs 스케치
    lines.append("## A. 모달리티 (Sketchy photo vs sketch)\n")
    lines.append("| 특성 | proxy | photo | sketch | 해석 |")
    lines.append("|---|---|---|---|---|")
    p, s = R["sketchy_photo"], R["sketchy_sketch"]
    rowsA = [
        ("Sparse line", "ink_ratio", "사진이 훨씬 빽빽, 스케치는 희소"),
        ("배경 없음", "bg_ratio", "스케치는 흰 배경 비율 높음"),
        ("Texture 부족", "edge_density", "사진은 질감으로 edge 많음"),
        ("색 부족", "sat", "사진 채도 높음, 스케치 무채색"),
        ("Completeness", "n_comp", "스케치는 끊긴 획으로 조각 많음"),
    ]
    for name, key, interp in rowsA:
        lines.append(f"| {name} | {key} | {fmt(p[key])} | {fmt(s[key])} | {interp} |")

    # B. 공간: SketchyScene vs COCO
    lines.append("\n## B. 공간 (SketchyScene vs COCO)\n")
    lines.append("| 특성 | proxy | sketchyscene | coco | 해석 |")
    lines.append("|---|---|---|---|---|")
    ss, co = R["sketchyscene"], R["coco"]
    rowsB = [
        ("Spatial layout", "spread_x", "잉크 좌우 분산"),
        ("Spatial layout", "spread_y", "잉크 상하 분산"),
        ("Spatial layout", "quad_entropy", "4분면 분포 균일도"),
    ]
    for name, key, interp in rowsB:
        lines.append(f"| {name} | {key} | {fmt(ss[key])} | {fmt(co[key])} | {interp} |")

    # C. 획 특성: 성인(Sketchy sketch) vs 아동(AI-Hub)
    lines.append("\n## C. 획 특성 (성인 Sketchy vs 아동 AI-Hub)\n")
    lines.append("| 특성 | proxy | 성인(sketch) | 아동(child) | 해석 |")
    lines.append("|---|---|---|---|---|")
    ad, ch = R["sketchy_sketch"], R["ai_hub_child"]
    rowsC = [
        ("떨림 진폭", "hf_tremor", "고주파 떨림(전체모양 제거) — 클수록 손떨림"),
        ("방향 급변", "turn_amp", "미세 방향 변화 — 클수록 미숙/떨림"),
    ]
    for name, key, interp in rowsC:
        lines.append(f"| {name} | {key} | {fmt(ad[key])} | {fmt(ch[key])} | {interp} |")

    out = "\n".join(lines) + "\n"
    with open("outputs/characteristics_report.md", "w", encoding="utf-8") as f:
        f.write(out)
    try:
        print("\n" + out)
    except UnicodeEncodeError:
        print("\n[report saved; console cannot render some chars]")
    print("saved: outputs/characteristics_report.md")

if __name__ == "__main__":
    main()
