# -*- coding: utf-8 -*-
"""Drawing Retrieval (5주차).
갤러리 FAISS 인덱스 + SigLIP2 임베딩 서버(:8100, SSH 터널)로 top-k 검색.
- text→image / image→image / caption→image
인덱스 빌드: scripts/build_index.py (오프라인)."""
import os, json, base64, urllib.request
import numpy as np
import faiss

IDXDIR = os.path.join(os.path.dirname(__file__), "index")
EMBED_URL = "http://localhost:8100"   # SigLIP2 임베딩 서버 (터널)
# SigLIP2 스케치 유사도는 압축돼 있어 절대 임계로 깔끔히 못 자름(유효 0.11 vs OOG 0.13 겹침).
# 명백히 무관한 꼬리(0.06~0.09)만 쳐내는 보수적 하한 → OOG 쿼리가 k개를 무조건 채우지 않게.
MIN_SCORE = 0.09

_index = None
_items = None
_meta = None


def _load():
    global _index, _items, _meta
    if _index is None:
        # 벡터는 npy로 저장(Windows 한글경로 faiss 저장 버그 회피) → 인메모리 IndexFlatIP.
        vecs = np.load(os.path.join(IDXDIR, "gallery_vecs.npy")).astype(np.float32)
        _index = faiss.IndexFlatIP(vecs.shape[1])
        _index.add(vecs)
        _meta = json.load(open(os.path.join(IDXDIR, "manifest.json"), encoding="utf-8"))
        _items = _meta["items"]
    return _index, _items


def _post(path, payload):
    req = urllib.request.Request(EMBED_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def embed_text(text):
    return np.asarray(_post("/embed/text", {"texts": [text]})["vecs"], dtype=np.float32)


def embed_image(img_bytes):
    b64 = base64.b64encode(img_bytes).decode()
    return np.asarray(_post("/embed/image", {"images": [b64]})["vecs"], dtype=np.float32)


def search(vec, k=8, min_score=MIN_SCORE):
    index, items = _load()
    D, I = index.search(vec, k)
    out = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or float(score) < min_score:   # 무관한 꼬리 제거
            continue
        m = items[idx]
        out.append({"gid": m["gid"], "url": m["url"], "source": m["source"],
                    "category": m["category"], "category_ko": m["category_ko"],
                    "age": m.get("age"), "score": round(float(score), 4)})
    return out


def retrieve_text(text, k=8):
    return search(embed_text(text), k)


def retrieve_image(img_bytes, k=8):
    return search(embed_image(img_bytes), k)
