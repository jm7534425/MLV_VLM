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

import cache
import pipeline
import retrieval
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


# ===== 1차: evaluation (실제 동작) =====
@app.post("/analyze")
async def analyze(image: UploadFile = File(...),
                  model: str = Form("qwen3-vl-8b"),
                  caption: str = Form(None)):
    img = await image.read()

    # 캡션: 전달받았으면 재사용 / 캐시 조회 / 없으면 생성
    cached = False
    cap_tokens, cap_ms = {}, 0
    if not caption:
        caption = cache.get(img, resolve_model(model), pipeline.PROMPT_VERSION)
        if caption:
            cached = True
        else:
            caption, cap_tokens, cap_ms = pipeline.make_caption(img, model)
            cache.put(img, resolve_model(model), pipeline.PROMPT_VERSION, caption)
    else:
        cached = True

    # evaluation
    result, ev_tokens, ev_ms, failure = pipeline.run_evaluation(img, model, caption)

    tokens = {k: (cap_tokens.get(k) or 0) + (ev_tokens.get(k) or 0)
              for k in ("input", "output", "total")}
    return envelope("evaluation", caption, result,
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
            caption, _, _ = pipeline.make_caption(img, model)   # 캡션용 vLLM 필요
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

@app.post("/caption")    # 6주 (구조화 캡션)
async def caption_task(): return not_implemented("captioning")

@app.post("/feedback")   # 7주 (evaluation + persona)
async def feedback(): return not_implemented("feedback")

@app.post("/edit")       # 8주 (종합)
async def edit(): return not_implemented("editing")


@app.get("/health")
async def health(): return {"status": "ok"}


# ===== 정적 프론트 (index.html) — 맨 마지막에 mount =====
app.mount("/", StaticFiles(directory="static", html=True), name="static")
