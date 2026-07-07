# -*- coding: utf-8 -*-
"""bench_out/emb_*.npz 들을 읽어 recall@k(정확히는 precision@k) 비교표 출력."""
import glob, os, numpy as np

OUT = "/home/intern/bench_out"
KS = [1, 5, 10]

def pAtk(qvecs, qcats, gvecs, gcats, exclude_self=False):
    """text->image: 각 쿼리 top-k 중 같은 카테고리 비율(P@k) 평균. R@1=top1 정답률."""
    sims = qvecs @ gvecs.T  # (Q,G) 정규화돼 있음=코사인
    res = {f"P@{k}": [] for k in KS}
    for i in range(len(qvecs)):
        order = np.argsort(-sims[i])
        if exclude_self:
            order = order[order != i]
        for k in KS:
            topk = order[:k]
            res[f"P@{k}"].append(np.mean(gcats[topk] == qcats[i]))
    return {k: float(np.mean(v)) for k, v in res.items()}

rows = []
for f in sorted(glob.glob(os.path.join(OUT, "emb_*.npz"))):
    key = os.path.basename(f)[4:-4]
    d = np.load(f, allow_pickle=True)
    gv, gc = d["gal_vecs"], d["gal_cats"]
    en, ko, qc = d["en_vecs"], d["ko_vecs"], d["q_cats"]
    t_en = pAtk(en, qc, gv, gc)
    t_ko = pAtk(ko, qc, gv, gc)
    i2i = pAtk(gv, gc, gv, gc, exclude_self=True)  # image->image
    rows.append((key, t_en, t_ko, i2i))

hdr = "| 모델 | text→img(EN) P@1/5/10 | text→img(KO) P@1/5/10 | img→img P@1/5/10 |"
print(hdr); print("|---|---|---|---|")
def fmt(r): return f"{r['P@1']:.2f}/{r['P@5']:.2f}/{r['P@10']:.2f}"
for key, en, ko, i2i in rows:
    print(f"| {key} | {fmt(en)} | {fmt(ko)} | {fmt(i2i)} |")
