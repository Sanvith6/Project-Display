import base64
import requests
import re
import time
import json
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

# Config Values
VISION_URL = "https://api.ocr.space/parse/image"
MAX_RETRIES = 3
INITIAL_BACKOFF = 2
CROP_TOP_RATIO = 0.83
CROP_BOTTOM_RATIO = 0.96

def crop_scorecard(img: Image.Image):
    w, h = img.size
    top = int(h * CROP_TOP_RATIO)
    bottom = int(h * CROP_BOTTOM_RATIO)
    return img.crop((0, top, w, bottom))

def preprocess_for_ocr(img: Image.Image):
    img = img.convert("L")
    contrast = ImageEnhance.Contrast(img)
    img = contrast.enhance(2.8)
    sharp = ImageEnhance.Sharpness(img)
    img = sharp.enhance(2.0)
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def resize_for_ocr(img: Image.Image, max_width=1200):
    w, h = img.size
    if w > max_width:
        ratio = max_width / w
        return img.resize((max_width, int(h * ratio)), Image.Resampling.LANCZOS)
    return img

def image_to_base64_bytes(pil_img: Image.Image, quality=90):
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def call_vision_ocr(base64_image: str, api_key: str):
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        data = {
            "apikey": api_key,
            "language": "eng",
            "base64Image": f"data:image/jpeg;base64,{base64_image}", 
            "isOverlayRequired": False,
            "OCREngine": 2,
            "scale": True,
            "detectOrientation": True,
        }
        try:
            resp = requests.post(VISION_URL, data=data, timeout=60)
            if resp.status_code == 200:
                result = resp.json()
                parsed = result.get("ParsedResults", [])
                text = "\n".join([p.get("ParsedText", "") for p in parsed]).strip()
                return {"success": True, "text": text}
        except Exception as e:
            time.sleep(backoff)
            backoff *= 2
    return {"success": False, "text": None}

def analyze_frame(image_path, api_key):
    img = Image.open(image_path).convert("RGB")
    processed_img = crop_scorecard(img)
    processed_img = preprocess_for_ocr(processed_img)
    processed_img = resize_for_ocr(processed_img)
    
    # Save debug image to confirm preprocessing looks good
    processed_img.save("debug_verified_crop.jpg")
    print("Saved 'debug_verified_crop.jpg' - Check this if text is garbage.")

    b64 = image_to_base64_bytes(processed_img)
    return call_vision_ocr(b64, api_key=api_key)

if __name__ == "__main__":
    test_frames = [
        r"c:\project\Project-Display\frames\frame_000042.jpg",
        r"c:\project\Project-Display\frames\frame_000060.jpg",
        r"c:\project\Project-Display\frames\frame_000180.jpg"
    ]
    api_key = "K85194943688957"
    
    for tf in test_frames:
        print(f"\nTesting {tf}...")
        try:
            res = analyze_frame(tf, api_key=api_key)
            print(f"[{tf}] Result: {res['text']}")
        except Exception as e:
            print(f"[{tf}] Error: {e}")
