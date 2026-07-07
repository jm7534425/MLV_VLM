# -*- coding: utf-8 -*-
"""kiddraw 벤치마크 임베딩 추출. model_key: qwen | siglip2 | clip
저장: bench_out/emb_<key>.npz  (gal_vecs, gal_cats, en_vecs, ko_vecs, q_cats)"""
import sys, os, json, glob, base64, urllib.request, numpy as np

BENCH = "/home/intern/bench_kiddraw"
OUT = "/home/intern/bench_out"; os.makedirs(OUT, exist_ok=True)
KEY = sys.argv[1]  # qwen|siglip2|clip

man = json.load(open(os.path.join(BENCH, "manifest.json"), encoding="utf-8"))
items = man["items"]; cats = sorted(set(it["category"] for it in items))
KO = {"TV":"텔레비전","airplane":"비행기","apple":"사과","bear":"곰","bed":"침대","bee":"벌","bike":"자전거",
"bird":"새","boat":"배","book":"책","bottle":"병","bowl":"그릇","cactus":"선인장","camel":"낙타","car":"자동차",
"cat":"고양이","chair":"의자","clock":"시계","couch":"소파","cow":"소","cup":"컵","dog":"개","elephant":"코끼리",
"face":"얼굴","fish":"물고기","frog":"개구리","hand":"손","hat":"모자","horse":"말","house":"집","ice cream":"아이스크림",
"key":"열쇠","lamp":"램프","mushroom":"버섯","octopus":"문어","person":"사람","phone":"전화기","piano":"피아노",
"rabbit":"토끼","scissors":"가위","sheep":"양","snail":"달팽이","spider":"거미","tiger":"호랑이","train":"기차",
"tree":"나무","watch":"손목시계","whale":"고래"}
EN_Q = [f"a drawing of a {c}" for c in cats]
KO_Q = [f"{KO[c]} 그림" for c in cats]
paths = [os.path.join(BENCH, it["path"]) for it in items]
gal_cats = np.array([it["category"] for it in items])

def norm(a):
    a = np.asarray(a, dtype=np.float32); return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)

# ---------- Qwen: HTTP 임베딩 엔드포인트 ----------
def run_qwen():
    URL = "http://localhost:8003/v1/embeddings"; MODEL = "Qwen/Qwen3-VL-Embedding-8B"
    def post(p):
        r = urllib.request.Request(URL, data=json.dumps(p).encode(), headers={"Content-Type":"application/json"})
        return json.loads(urllib.request.urlopen(r, timeout=120).read())["data"][0]["embedding"]
    def txt(t): return post({"model":MODEL,"input":t})
    def im(path):
        b64 = base64.b64encode(open(path,"rb").read()).decode()
        return post({"model":MODEL,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{b64}"}}]}]})
    gv = []
    for i,p in enumerate(paths):
        gv.append(im(p))
        if (i+1)%200==0: print(f"  gal {i+1}/{len(paths)}", flush=True)
    en = [txt(q) for q in EN_Q]; ko = [txt(q) for q in KO_Q]
    return norm(gv), norm(en), norm(ko)

# ---------- SigLIP2 / CLIP: transformers ----------
def run_hf(key):
    import torch
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    def load(p):  # kiddraw PNG는 그림이 알파채널 → 흰배경 합성해야 보임
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255,255,255,255))
        return Image.alpha_composite(bg, im).convert("RGB")
    if key=="siglip2":
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-so400m-patch16-384"
        model = AutoModel.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
        proc = AutoProcessor.from_pretrained(mid)
        pooler = lambda x: x.pooler_output if hasattr(x,"pooler_output") else x
        img_fn = lambda pv: pooler(model.get_image_features(pixel_values=pv))
        txt_fn = lambda **t: pooler(model.get_text_features(**t))
        pad = "max_length"
    else:  # clip: get_image_features가 ModelOutput만 줘서 투영 수동 적용(정렬공간)
        from transformers import CLIPModel, CLIPProcessor
        mid = "openai/clip-vit-large-patch14"
        model = CLIPModel.from_pretrained(mid, dtype=torch.float16).to(dev).eval()
        proc = CLIPProcessor.from_pretrained(mid)
        img_fn = lambda pv: model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
        txt_fn = lambda **t: model.text_projection(model.text_model(**t).pooler_output)
        pad = True
    @torch.no_grad()
    def img_vecs(pth):
        out=[]
        for i in range(0,len(pth),32):
            ims=[load(p) for p in pth[i:i+32]]
            inp=proc(images=ims, return_tensors="pt").to(dev)
            pv=inp["pixel_values"]; pv=pv.half() if pv.dtype==torch.float32 else pv
            out.append(img_fn(pv).float().cpu().numpy())
            if (i+32)%256==0: print(f"  gal {i+32}/{len(pth)}", flush=True)
        return np.concatenate(out,0)
    @torch.no_grad()
    def txt_vecs(qs):
        inp=proc(text=qs, padding=pad, truncation=True, return_tensors="pt").to(dev)
        kw={k:inp[k] for k in ("input_ids","attention_mask") if k in inp}  # siglip은 attention_mask 없음
        return txt_fn(**kw).float().cpu().numpy()
    return norm(img_vecs(paths)), norm(txt_vecs(EN_Q)), norm(txt_vecs(KO_Q))

if KEY=="qwen": gv,en,ko = run_qwen()
else:           gv,en,ko = run_hf(KEY)

np.savez(os.path.join(OUT,f"emb_{KEY}.npz"), gal_vecs=gv, gal_cats=gal_cats,
         en_vecs=en, ko_vecs=ko, q_cats=np.array(cats))
print(f"[{KEY}] saved: gal={gv.shape} en={en.shape} ko={ko.shape}")
