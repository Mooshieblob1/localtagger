from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from PIL import Image
import io
import onnxruntime as ort
from huggingface_hub import hf_hub_download
import numpy as np
import cv2
import pandas as pd
import requests
import re

app = FastAPI(title="LAN Image Interrogator")

# --- MODEL MANAGER ---

class ModelManager:
    def __init__(self):
        self.wd14_session = None
        self.wd14_tags = None
        self.wd14_tag_names = None

    def load_wd14(self):
        if self.wd14_session is not None:
            return self.wd14_session, self.wd14_tag_names

        print("Loading WD14 Tagger (EVA02-Large)...")
        WD14_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
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

def make_square(img, target_size):
    old_size = img.shape[:2]
    desired_size = max(old_size)
    desired_size = max(desired_size, target_size)

    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [255, 255, 255]
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def smart_resize(img, size):
    # Assumes the image has already gone through make_square
    if img.shape[0] > size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    elif img.shape[0] < size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    else:  # just do nothing
        pass
    return img

def prepare_image_wd14(image: Image.Image, target_size=448):
    # Handle alpha channel: paste on white background
    image = image.convert('RGBA')
    new_image = Image.new('RGBA', image.size, 'WHITE')
    new_image.paste(image, mask=image)
    image = new_image.convert('RGB')
    
    # Convert to numpy array (RGB)
    img = np.array(image)
    
    # Convert RGB to BGR (OpenCV format) as expected by the reference code
    img = img[:, :, ::-1]

    # Preprocessing: make square and resize
    img = make_square(img, target_size)
    img = smart_resize(img, target_size)
    
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)
    return img

RE_SPECIAL = re.compile(r'([\\()])')

def run_wd14(image: Image.Image, threshold: float, use_spaces: bool = False, use_escape: bool = True, include_ranks: bool = False, score_descend: bool = True):
    session, tag_names = model_manager.load_wd14()
    
    # Get input size from model if possible, otherwise default to 448
    try:
        _, height, _, _ = session.get_inputs()[0].shape
    except:
        height = 448

    input_tensor = prepare_image_wd14(image, target_size=height)
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: input_tensor})[0][0]
    
    res_tags = {}
    # Tags that are known to hallucinate due to color bleeding from clothing/backgrounds.
    # Enforce a stricter minimum confidence for these regardless of the user-provided threshold.
    high_hallucination_tags = {"blue_skin", "green_skin", "red_skin", "colored_skin", "pale_skin"}
    for i, p in enumerate(probs):
        tag_name = tag_names[i]
        # normalize tag name for robust matching (lowercase, spaces -> underscores)
        tag_norm = tag_name.lower().replace(" ", "_")

        if tag_norm in high_hallucination_tags:
            # Enforce strict minimum confidence of 0.85 for these tags
            if p >= 0.85:
                res_tags[tag_name] = float(p)
            # otherwise discard immediately (do not include)
            continue

        # Default behavior for other tags: use user-provided threshold
        if p > threshold:
            res_tags[tag_name] = float(p)
            
    # Formatting logic from reference code
    text_items = []
    tags_pairs = res_tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    
    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', '-')
        else:
            tag_outformat = tag_outformat.replace(' ', ', ')
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)

    if use_spaces:
        output_text = ' '.join(text_items)
    else:
        output_text = ', '.join(text_items)

    # Also return the raw tags dictionary (sorted)
    sorted_tags = dict(sorted(res_tags.items(), key=lambda item: item[1], reverse=True))
    
    return {"tags": sorted_tags, "tag_string": output_text}

# --- ENDPOINTS ---

# Main Interrogation Endpoints (EVA02-Large)
@app.post("/interrogate")
async def interrogate_post(
    file: UploadFile = File(...), 
    threshold: float = 0.35,
    use_spaces: bool = False,
    use_escape: bool = True,
    include_ranks: bool = False,
    score_descend: bool = True
):
    image_data = await file.read()
    image = load_image_from_bytes(image_data)
    return run_wd14(image, threshold, use_spaces, use_escape, include_ranks, score_descend)

@app.get("/interrogate")
async def interrogate_get(
    url: str = Query(...), 
    threshold: float = 0.35,
    use_spaces: bool = False,
    use_escape: bool = True,
    include_ranks: bool = False,
    score_descend: bool = True
):
    image = download_image(url)
    return run_wd14(image, threshold, use_spaces, use_escape, include_ranks, score_descend)


@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
