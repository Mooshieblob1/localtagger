import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
import io
from clip_interrogator import Config, Interrogator
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
import pandas as pd
import requests
import gc

app = FastAPI(title="LAN Image Interrogator")

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- MODEL MANAGER ---

class ModelManager:
    def __init__(self):
        self.current_clip_model_name = None
        self.ci_interrogator = None
        self.wd14_session = None
        self.wd14_tags = None
        self.wd14_tag_names = None

    def unload_clip(self):
        if self.ci_interrogator is not None:
            print("Unloading CLIP model...")
            del self.ci_interrogator
            self.ci_interrogator = None
            self.current_clip_model_name = None
            gc.collect()
            torch.cuda.empty_cache()

    def load_clip(self, model_name: str):
        if self.current_clip_model_name == model_name and self.ci_interrogator is not None:
            return self.ci_interrogator

        # Unload if different model is loaded
        self.unload_clip()

        print(f"Loading CLIP model: {model_name} on {DEVICE}...")
        config = Config(clip_model_name=model_name, device=DEVICE)
        self.ci_interrogator = Interrogator(config)
        self.current_clip_model_name = model_name
        return self.ci_interrogator

    def load_wd14(self):
        if self.wd14_session is not None:
            return self.wd14_session, self.wd14_tag_names

        print("Loading WD14 Tagger...")
        WD14_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
        model_path = hf_hub_download(repo_id=WD14_REPO, filename="model.onnx")
        tags_path = hf_hub_download(repo_id=WD14_REPO, filename="selected_tags.csv")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' not in ort.get_available_providers():
            print("Warning: CUDAExecutionProvider not found for ONNX Runtime. Fallback to CPU.")
            providers = ['CPUExecutionProvider']

        self.wd14_session = ort.InferenceSession(model_path, providers=providers)
        tags_df = pd.read_csv(tags_path)
        self.wd14_tags = tags_df
        self.wd14_tag_names = tags_df['name'].tolist()
        return self.wd14_session, self.wd14_tag_names

model_manager = ModelManager()

# --- HELPER FUNCTIONS ---

def download_image(url: str) -> Image.Image:
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

def load_image_from_bytes(image_data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def prepare_image_wd14(image: Image.Image, target_size=448):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img = np.array(image)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    return img

def run_wd14(image: Image.Image, threshold: float):
    session, tag_names = model_manager.load_wd14()
    
    input_tensor = prepare_image_wd14(image)
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: input_tensor})[0][0]
    
    res_tags = {}
    for i, p in enumerate(probs):
        if p > threshold:
            res_tags[tag_names[i]] = float(p)
            
    sorted_tags = dict(sorted(res_tags.items(), key=lambda item: item[1], reverse=True))
    return {"tags": sorted_tags, "tag_string": ", ".join(sorted_tags.keys())}

def run_clip(model_name: str, image: Image.Image, mode: str):
    interrogator = model_manager.load_clip(model_name)
    if mode == "best":
        return interrogator.interrogate(image)
    else:
        return interrogator.generate_caption(image)

# --- ENDPOINTS ---

# VIT Endpoints
@app.post("/interrogate/vit")
async def interrogate_vit_post(file: UploadFile = File(...), mode: str = "fast"):
    image_data = await file.read()
    image = load_image_from_bytes(image_data)
    caption = run_clip("ViT-L-14/openai", image, mode)
    return {"caption": caption, "model": "ViT-L-14/openai"}

@app.get("/interrogate/vit")
async def interrogate_vit_get(url: str = Query(...), mode: str = "fast"):
    image = download_image(url)
    caption = run_clip("ViT-L-14/openai", image, mode)
    return {"caption": caption, "model": "ViT-L-14/openai"}

# EVA Endpoints
@app.post("/interrogate/eva")
async def interrogate_eva_post(file: UploadFile = File(...), mode: str = "fast"):
    image_data = await file.read()
    image = load_image_from_bytes(image_data)
    caption = run_clip("ViT-g-14/laion2b_s12b_b42k", image, mode)
    return {"caption": caption, "model": "ViT-g-14/laion2b_s12b_b42k"}

@app.get("/interrogate/eva")
async def interrogate_eva_get(url: str = Query(...), mode: str = "fast"):
    image = download_image(url)
    caption = run_clip("ViT-g-14/laion2b_s12b_b42k", image, mode)
    return {"caption": caption, "model": "ViT-g-14/laion2b_s12b_b42k"}

# PixAI (WD14) Endpoints
@app.post("/interrogate/pixai")
async def interrogate_pixai_post(file: UploadFile = File(...), threshold: float = 0.35):
    image_data = await file.read()
    image = load_image_from_bytes(image_data)
    return run_wd14(image, threshold)

@app.get("/interrogate/pixai")
async def interrogate_pixai_get(url: str = Query(...), threshold: float = 0.35):
    image = download_image(url)
    return run_wd14(image, threshold)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
