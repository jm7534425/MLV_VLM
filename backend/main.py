# -*- coding: utf-8 -*-
"""FastAPI backend skeleton (3주차).
- POST /analyze (evaluation) : 실제 동작
- /retrieve /caption /feedback /edit : 자리만 (5~8주)
- static/index.html 서빙
실행: uvicorn main:app --reload --port 9000  (SSH 터널로 vLLM 8000 연결 전제)
"""
from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import caption_db
import pipeline
import retrieval
import caption as caption_mod
from vllm_client import resolve_model

app = FastAPI(title="Drawing Analysis API", version="0.1")


def envelope(task, caption, result, *, model, tokens, latency_ms,
             caption_cached=False, failure_type=None):
    return {
        "task": task,
        "caption": caption,
        "result": result,
        "meta": {
            "model": resolve_model(model),
            "prompt_version": pipeline.PROMPT_VERSION,
            "latency_ms": latency_ms,
            "tokens": tokens,
            "cost": None,
            "caption_cached": caption_cached,
            "failure_type": failure_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def not_implemented(task):
    return JSONResponse(status_code=501,
                        content={"task": task, "result": None,
                                 "meta": {"failure_type": "not_implemented"}})


# ===== 공유 캡션 (caption-first): 구조화 캡션을 1회 생성·캐시 → 모든 태스크가 재사용 =====
def get_or_build_caption(img, model):
    """구조화 캡션을 캐시에서 가져오거나 새로 만듦.
    반환: (cap_obj, tokens, latency_ms, cached, err)."""
    ver = caption_mod.CAPTION_PROMPT_VERSION
    key_model = resolve_model(model)
    hit = caption_db.get(img, key_model, ver)     # SQLite 영속 캐시
    if hit is not None:
        return hit, {}, 0, True, None
    obj, tokens, ms, err = caption_mod.build_caption(img, model)
    if err:
        return None, tokens, ms, False, err
    caption_db.put(img, key_model, ver, obj)
    return obj, tokens, ms, False, None


# ===== 1차: evaluation — 구조화 캡션(WHAT) 자동 생성·캐시 → 발달단계 판정(WHY) =====
@app.post("/analyze")
async def analyze(image: UploadFile = File(...),
                  model: str = Form("qwen3-vl-8b")):
    img = await image.read()

    # 캡션: 구조화 캡션을 캐시 조회 / 생성 (검색·캡션뷰와 공유)
    cap_obj, cap_tokens, cap_ms, cached, cap_err = get_or_build_caption(img, model)
    if cap_err:
        return JSONResponse(status_code=200, content={
            "task": "evaluation", "caption": None, "result": None,
            "meta": {"failure_type": cap_err}})

    # evaluation: 구조화 캡션의 WHAT 텍스트를 판정 근거로
    cap_text = caption_mod.caption_to_text(cap_obj)
    result, ev_tokens, ev_ms, failure = pipeline.run_evaluation(img, model, cap_text)

    tokens = {k: (cap_tokens.get(k) or 0) + (ev_tokens.get(k) or 0)
              for k in ("input", "output", "total")}
    return envelope("evaluation", cap_obj, result,       # caption = 구조화 객체
                    model=model, tokens=tokens, latency_ms=cap_ms + ev_ms,
                    caption_cached=cached, failure_type=failure)


# ===== 2차: retrieval (실제 동작, 5주) =====
@app.post("/retrieve")
async def retrieve(image: UploadFile = File(None),
                   query: str = Form(None),
                   mode: str = Form("auto"),   # text | image | caption | auto
                   model: str = Form("qwen3-vl-8b"),
                   k: int = Form(8)):
    t0 = datetime.now(timezone.utc)
    caption = None
    if mode == "auto":
        mode = "text" if query else "image"

    try:
        if mode == "text":
            results = retrieval.retrieve_text(query, k)
            q = {"type": "text", "value": query}
        elif mode == "caption":
            img = await image.read()
            cap_obj, _, _, _, cerr = get_or_build_caption(img, model)  # 공유 구조화 캡션
            if cerr:
                raise RuntimeError(cerr)
            caption = caption_mod.caption_to_text(cap_obj)
            results = retrieval.retrieve_text(caption, k)
            q = {"type": "caption", "value": caption}
        else:  # image
            img = await image.read()
            results = retrieval.retrieve_image(img, k)
            q = {"type": "image", "value": None}
    except Exception as e:
        # 임베딩서버(8100)·캡션 vLLM(8000) 미기동 등 → 깔끔한 JSON 에러로
        hint = "caption 모드는 캡션용 vLLM(8000)이 필요합니다" if mode == "caption" \
               else "임베딩 서버(8100 터널)를 확인하세요"
        return JSONResponse(status_code=200, content={
            "task": "retrieval", "query": {"type": mode, "value": query}, "results": [],
            "meta": {"failure_type": "backend_unavailable",
                     "error": f"{type(e).__name__}: {e}", "hint": hint}})

    ms = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
    return {"task": "retrieval", "query": q, "caption": caption, "results": results,
            "meta": {"embed_model": "google/siglip2-so400m-patch16-384",
                     "k": k, "latency_ms": ms,
                     "timestamp": datetime.now(timezone.utc).isoformat()}}

@app.post("/caption")    # 구조화 캡션(공유 캐시). UI 탭은 없앰 — 평가가 자동 생성. API용으로 유지.
async def caption_task(image: UploadFile = File(...),
                       model: str = Form("qwen3-vl-8b")):
    img = await image.read()
    try:
        obj, tokens, ms, _cached, err = get_or_build_caption(img, model)
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "task": "captioning", "result": None,
            "meta": {"failure_type": "backend_unavailable", "error": f"{type(e).__name__}: {e}"}})
    if err:
        return JSONResponse(status_code=200, content={
            "task": "captioning", "result": None, "meta": {"failure_type": err}})
    return {"task": "captioning", "result": obj,
            "meta": {"model": resolve_model(model),
                     "prompt_version": caption_mod.CAPTION_PROMPT_VERSION,
                     "latency_ms": ms, "tokens": tokens,
                     "timestamp": datetime.now(timezone.utc).isoformat()}}

@app.post("/feedback")   # 7주 (evaluation + persona)
async def feedback(): return not_implemented("feedback")

@app.post("/edit")       # 8주 (종합)
async def edit(): return not_implemented("editing")


@app.get("/health")
async def health(): return {"status": "ok"}


@app.get("/captions")    # Caption DB 현황 (개수 + 최근) — 검증·데모용
async def captions_stats(): return caption_db.stats()


# ===== 정적 프론트 (index.html) — 맨 마지막에 mount =====
app.mount("/", StaticFiles(directory="static", html=True), name="static")
