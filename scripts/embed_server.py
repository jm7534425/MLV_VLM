# -*- coding: utf-8 -*-
"""SigLIP2 임베딩 마이크로서비스 (GPU4). 이미지·텍스트를 같은 공간 벡터로.
POST /embed/text {texts:[...]}  /embed/image {images:[b64...]}  -> {vecs:[[...]], dim}"""
import base64, io, torch, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModel, AutoProcessor

MID = "google/siglip2-so400m-patch16-384"
DEV = "cuda"
model = AutoModel.from_pretrained(MID, dtype=torch.float16).to(DEV).eval()
proc = AutoProcessor.from_pretrained(MID)

def _pool(x):
    return x.pooler_output if hasattr(x, "pooler_output") else x
def _norm(a):
    a = a.astype(np.float32); return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
def _load(b64):  # kiddraw/AI-Hub PNG는 알파에 그림 → 흰배경 합성
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")

app = FastAPI()
class TextReq(BaseModel): texts: list[str]
class ImgReq(BaseModel): images: list[str]

@app.get("/health")
def health(): return {"ok": True, "model": MID}

@app.post("/embed/text")
def embed_text(r: TextReq):
    with torch.no_grad():
        inp = proc(text=r.texts, padding="max_length", truncation=True, return_tensors="pt").to(DEV)
        kw = {k: inp[k] for k in ("input_ids", "attention_mask") if k in inp}
        v = _pool(model.get_text_features(**kw)).float().cpu().numpy()
    return {"vecs": _norm(v).tolist(), "dim": v.shape[1]}

@app.post("/embed/image")
def embed_image(r: ImgReq):
    out = []
    with torch.no_grad():
        for i in range(0, len(r.images), 32):
            ims = [_load(b) for b in r.images[i:i+32]]
            inp = proc(images=ims, return_tensors="pt").to(DEV)
            pv = inp["pixel_values"]; pv = pv.half() if pv.dtype == torch.float32 else pv
            f = _pool(model.get_image_features(pixel_values=pv)).float().cpu().numpy()
            out.append(f)
    v = np.concatenate(out, 0)
    return {"vecs": _norm(v).tolist(), "dim": v.shape[1]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
