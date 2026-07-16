# -*- coding: utf-8 -*-
"""6주차 Phase4 실험 — 캡션 해석형 필드 신뢰도 + grounding 효과.
(a) VLM line_quality ↔ 코드 tremor 일치율  (b) 필드 다양성(템플릿화)  (c) grounding 개선.
baseline(힌트X) vs grounded(CV 힌트) 비교."""
import sys, os, glob, re, json
sys.path.insert(0, "backend")
import cv_features
from cv_features import to_hints
from caption import make_structured_caption

SAMPLES = "data/samples/balanced"
N = 12


def code_line(tremor):
    return "떨림" if tremor > 0.55 else "안정"

def vlm_line(lq):
    if any(w in lq for w in ("떨", "불안", "삐뚤", "흔들", "거칠")): return "떨림"
    if any(w in lq for w in ("안정", "깔끔", "일정", "곧", "매끈")): return "안정"
    return "불명"


def run(image_bytes, measured, grounded):
    hints = to_hints(measured) if grounded else None
    sem, _, ms, err = make_structured_caption(image_bytes, hints=hints)
    return sem, ms, err


def metrics(rows, measured_list):
    """rows: [sem,...] 한 조건. 일치율·flag율·다양성 계산."""
    agree = det = flags = 0
    sp, lq, cp = set(), set(), set()
    for sem, m in zip(rows, measured_list):
        if not sem:
            continue
        v = vlm_line(sem["line_quality"]); c = code_line(m["line_tremor"])
        if v != "불명":
            det += 1; agree += (v == c)
            if v != c: flags += 1
        sp.add(sem["spatial_layout"]); lq.add(sem["line_quality"]); cp.add(sem["completeness"])
    n = sum(1 for s in rows if s)
    return {
        "line_agree": f"{agree}/{det}" + (f" ({agree/det:.0%})" if det else ""),
        "line_mismatch": flags,
        "diversity_spatial": f"{len(sp)}/{n}",
        "diversity_line": f"{len(lq)}/{n}",
        "diversity_completeness": f"{len(cp)}/{n}",
    }


def main():
    files = sorted(glob.glob(SAMPLES + "/*.jpg") + glob.glob(SAMPLES + "/*.png"))
    files = files[::max(1, len(files) // N)][:N]
    print(f"샘플 {len(files)}장")
    measured_list, base_rows, grnd_rows, detail = [], [], [], []
    for i, f in enumerate(files):
        b = open(f, "rb").read()
        m = cv_features.measure(b); measured_list.append(m)
        sb, msb, _ = run(b, m, False); base_rows.append(sb)
        sg, msg, _ = run(b, m, True); grnd_rows.append(sg)
        age = (re.search(r"age(\d+)", f) or [None, "?"])[1] if "age" in f else "?"
        detail.append((os.path.basename(f), age, round(m["line_tremor"], 2),
                       sb["line_quality"] if sb else "-", sg["line_quality"] if sg else "-"))
        print(f"  [{i+1}/{len(files)}] {os.path.basename(f)} tremor={m['line_tremor']:.2f}")

    B, G = metrics(base_rows, measured_list), metrics(grnd_rows, measured_list)
    out = ["# 캡션 해석형 필드 실험 (baseline vs grounding)\n",
           f"샘플 {len(files)}장 (나이 스펙트럼). line 라벨: 코드 tremor>0.55=떨림.\n",
           "| 지표 | baseline | +grounding |", "|---|---|---|",
           f"| line_quality ↔ 코드 일치율 | {B['line_agree']} | {G['line_agree']} |",
           f"| line 불일치(환각) 수 | {B['line_mismatch']} | {G['line_mismatch']} |",
           f"| spatial_layout 다양성(고유/전체) | {B['diversity_spatial']} | {G['diversity_spatial']} |",
           f"| line_quality 다양성 | {B['diversity_line']} | {G['diversity_line']} |",
           f"| completeness 다양성 | {B['diversity_completeness']} | {G['diversity_completeness']} |",
           "\n## 이미지별 line_quality (tremor / baseline / grounded)\n",
           "| 파일 | age | tremor | baseline | grounded |", "|---|---|---|---|---|"]
    for fn, age, tr, lb, lg in detail:
        out.append(f"| {fn} | {age} | {tr} | {lb} | {lg} |")
    os.makedirs("outputs", exist_ok=True)
    open("outputs/caption_experiment.md", "w", encoding="utf-8").write("\n".join(out))
    print("\nsaved outputs/caption_experiment.md")
    print(f"일치율 baseline {B['line_agree']} → grounded {G['line_agree']}")
    print(f"line 다양성 baseline {B['diversity_line']} → grounded {G['diversity_line']}")


if __name__ == "__main__":
    main()
