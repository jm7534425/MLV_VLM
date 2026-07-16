# -*- coding: utf-8 -*-
"""vLLM(OpenAI 호환) 호출 클라이언트 — GPU 서버의 Qwen3-VL."""
import os, json, time, base64, urllib.request

# 기본 = 데스크탑 GPU서버(Tailscale serve HTTPS 프록시). 환경변수로 교체 가능(터널 등).
VLLM_URL = os.environ.get(
    "VLLM_URL", "https://node.tail841802.ts.net/v1/chat/completions")
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

# 프론트의 모델 별칭 → 실제 vLLM 모델 id (4주차 모델비교 대비)
MODEL_MAP = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    # "llava-ov-1.5": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",  # 4주차 추가
}


def resolve_model(alias: str) -> str:
    return MODEL_MAP.get(alias, DEFAULT_MODEL)


def img_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def chat(messages, model_id: str, max_tokens: int = 500, temperature: float = 0.0,
         response_format=None):
    """vLLM 호출 → (텍스트, usage dict, latency_ms).
    response_format: json_schema 강제 시 {"type":"json_schema","json_schema":{...}}."""
    payload = {"model": model_id, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if response_format:
        payload["response_format"] = response_format
    req = urllib.request.Request(
        VLLM_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    latency_ms = int((time.time() - t0) * 1000)
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    tokens = {"input": usage.get("prompt_tokens"),
              "output": usage.get("completion_tokens"),
              "total": usage.get("total_tokens")}
    return text, tokens, latency_ms


def vision_chat(image_bytes, prompt, model_id, system=None, **kw):
    """이미지 + 텍스트 1턴 호출."""
    content = [{"type": "image_url", "image_url": {"url": img_to_data_url(image_bytes)}},
               {"type": "text", "text": prompt}]
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": content}]
    return chat(messages, model_id, **kw)
