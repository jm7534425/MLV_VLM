# -*- coding: utf-8 -*-
"""
획 떨림 — 나이별 분석 (A안: AI-Hub 내부 비교로 매체/스캔/과제 교란 통제)
- 알고리즘 수정: arc-length 정규화 + reflect-padding 이동평균(경계편향 제거)
- 유의성: 순열검정(permutation test, scipy 불필요)
주의: 윤곽선(외곽) 기반이라 절대값은 떨림+두께 혼합. 단 같은 매체끼리 비교라 '나이 효과'는 공정.
"""
import os, glob, json
import numpy as np
import cv2

SRC = "data/AI-hub아동그림"
MAXDIM = 768
N = 256          # 윤곽 리샘플 점수(스케일 불변)
KSMOOTH = 13     # 저주파(전체모양) 제거용 창

def load_gray(path):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)  # 한글경로
    if img is None: return None
    h, w = img.shape[:2]; s = MAXDIM / max(h, w)
    if s < 1: img = cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
    return img

def stroke_metrics(gray):
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    cnts, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    hf_list, amp_list = [], []
    for c in cnts:
        if len(c) < 60: continue
        pts = c[:, 0, :].astype(np.float64)
        seg = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1))
        L = np.concatenate([[0], np.cumsum(seg)])
        if L[-1] < 80: continue
        uu = np.linspace(0, L[-1], N)
        rx = np.interp(uu, L, pts[:, 0]); ry = np.interp(uu, L, pts[:, 1])
        ang = np.unwrap(np.arctan2(np.gradient(ry), np.gradient(rx)))
        # reflect-padding 이동평균 → 경계 0-패딩 편향(버그③) 제거
        pad = np.pad(ang, KSMOOTH // 2, mode="reflect")
        sm = np.convolve(pad, np.ones(KSMOOTH) / KSMOOTH, mode="valid")
        hf_list.append((ang - sm).std())               # 떨림 진폭
        amp_list.append(np.abs(np.diff(ang, 2)).mean())  # 방향 급변
    if not hf_list: return None
    return float(np.mean(hf_list)), float(np.mean(amp_list))

def parse_age(path):
    p = os.path.basename(path).split("_")      # 나무_10_남_00498.jpg
    return int(p[1]) if len(p) >= 2 and p[1].isdigit() else None

def perm_corr_p(x, y, K=5000, seed=0):
    """순열검정: |corr| 가 관측만큼 큰 셔플 비율 = p"""
    r_obs = np.corrcoef(x, y)[0, 1]
    rng = np.random.default_rng(seed)
    cnt = sum(abs(np.corrcoef(rng.permutation(x), y)[0, 1]) >= abs(r_obs) for _ in range(K))
    return r_obs, (cnt + 1) / (K + 1)

def perm_diff_p(a, b, K=5000, seed=0):
    """두 그룹 평균차 순열검정"""
    obs = np.mean(b) - np.mean(a)
    pool = np.concatenate([a, b]); na = len(a)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(K):
        s = rng.permutation(pool)
        if abs(np.mean(s[na:]) - np.mean(s[:na])) >= abs(obs): cnt += 1
    return obs, (cnt + 1) / (K + 1)

def main():
    rows = []
    for f in glob.glob(SRC + "/*/*"):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")): continue
        age = parse_age(f)
        g = load_gray(f)
        if age is None or g is None: continue
        m = stroke_metrics(g)
        if m is None: continue
        rows.append({"age": age, "hf_tremor": m[0], "turn_amp": m[1],
                     "cat": os.path.basename(os.path.dirname(f))})
    age = np.array([r["age"] for r in rows])
    hf = np.array([r["hf_tremor"] for r in rows])
    ta = np.array([r["turn_amp"] for r in rows])
    print(f"분석 이미지: {len(rows)}")

    lines = ["# 획 떨림 — 나이별 분석 (AI-Hub 내부, 교란 통제)\n",
             f"분석 N={len(rows)}, 나이 {age.min()}~{age.max()}세. "
             "같은 연필·스캔·과제이므로 '나이 효과'만 비교됨.\n",
             "## 나이별 평균\n", "| 나이 | N | hf_tremor | turn_amp |", "|---|---|---|---|"]
    for a in sorted(set(age)):
        mask = age == a
        lines.append(f"| {a} | {mask.sum()} | {hf[mask].mean():.4f} | {ta[mask].mean():.4f} |")

    # 상관 + 순열검정 (가설: 나이↑ → 떨림↓ = 음의 상관)
    lines.append("\n## 나이-떨림 상관 (순열검정)\n| 지표 | Pearson r | p(perm) | 해석 |")
    lines.append("|---|---|---|---|")
    for name, y in [("hf_tremor", hf), ("turn_amp", ta)]:
        r, p = perm_corr_p(age.astype(float), y)
        sig = "유의(p<0.05)" if p < 0.05 else "유의하지 않음"
        direc = "나이↑→떨림↓" if r < 0 else "나이↑→떨림↑"
        lines.append(f"| {name} | {r:+.3f} | {p:.4f} | {direc}, {sig} |")

    # 어린(8~9) vs 큰(12~13) 버킷
    young = np.isin(age, [8, 9]); old = np.isin(age, [12, 13])
    lines.append(f"\n## 버킷 비교: 어린(8~9, N={young.sum()}) vs 큰(12~13, N={old.sum()})\n")
    lines.append("| 지표 | 어린 | 큰 | 차이 | p(perm) |")
    lines.append("|---|---|---|---|---|")
    for name, y in [("hf_tremor", hf), ("turn_amp", ta)]:
        diff, p = perm_diff_p(y[young], y[old])
        lines.append(f"| {name} | {y[young].mean():.4f} | {y[old].mean():.4f} | {diff:+.4f} | {p:.4f} |")

    out = "\n".join(lines) + "\n"
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stroke_age_report.md", "w", encoding="utf-8") as f:
        f.write(out)
    json.dump(rows, open("outputs/stroke_age_values.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    try: print(out)
    except UnicodeEncodeError: print("[report saved]")
    print("saved: outputs/stroke_age_report.md")

if __name__ == "__main__":
    main()
