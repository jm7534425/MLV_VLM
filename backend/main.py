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


# ===== 자리만 (후속 주차) =====
@app.post("/retrieve")   # 5주
async def retrieve(): return not_implemented("retrieval")

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
