import base64
import requests
import re
import time
import json
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm
from pathlib import Path
from io import BytesIO

from config import (
    OCR_KEYS, VISION_URL,
    MAX_RETRIES, INITIAL_BACKOFF, FRAME_SUBSAMPLE, FRAME_RATE
)
from video_processing import get_sampled_frame_paths


# ================= CUSTOM OCR SETTINGS (Sony LIV) =================

# Hard-tuned crop for Sony LIV cricket scoreboard
CROP_TOP_RATIO = 0.83
CROP_BOTTOM_RATIO = 0.96


# ================= IMAGE PROCESSING =================

def crop_scorecard(img: Image.Image):
    """Crop only the bottom scoreboard strip."""
    w, h = img.size
    top = int(h * CROP_TOP_RATIO)
    bottom = int(h * CROP_BOTTOM_RATIO)
    return img.crop((0, top, w, bottom))


def preprocess_for_ocr(img: Image.Image):
    """
    Make text OCR-friendly:
    - Grayscale
    - Increase contrast
    - Sharpen
    - Reduce noise
    """
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
    """Convert PIL image to base64 (raw, no data:image prefix)."""
    buf = BytesIO()
    # Ensure RGB for JPEG saving
    # If the image is "L" (grayscale) from preprocess, saving as JPEG is still fine
    pil_img = pil_img.convert("RGB")
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ================= OCR.SPACE API =================

def call_vision_ocr(base64_image: str, api_key: str = None):
    """
    Calls the OCR.Space API using base64 image data.
    """
    if not api_key:
        return {"success": False, "text": None, "error": "No API key provided"}

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
        except Exception as e:
            if attempt == MAX_RETRIES:
                return {"success": False, "text": None, "error": str(e)}
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 403:
             return { "success": False, "text": None, "error": f"Rate limit (403) for key {api_key[:4]}..." }

        if resp.status_code != 200:
            if attempt == MAX_RETRIES:
                 return { "success": False, "text": None, "error": f"HTTP {resp.status_code}: {resp.text}" }
            time.sleep(backoff)
            backoff *= 2
            continue

        try:
            result = resp.json()
        except:
             return {"success": False, "text": None, "error": "Invalid JSON response"}

        if result.get("IsErroredOnProcessing"):
            err_msg = result.get("ErrorMessage")
            # Loop retry on limit? No, user prefers fail-fast or clean retry. 
            return {"success": False, "text": None, "error": str(err_msg)}

        parsed = result.get("ParsedResults", [])
        text = "\n".join([p.get("ParsedText", "") for p in parsed]).strip()
        return {"success": True, "text": text, "error": None}

    return {"success": False, "text": None, "error": "Max retries exceeded"}


# ================= SCORE PARSER (User Provided) =================

score_pattern = re.compile(
    r"(?P<team>[A-Z]{2,3})\s+"
    r"(?P<runs>\d+)\s*[-/]\s*(?P<wickets>\d+)\s+"
    r"(?P<phase>[A-Z])?\s*"
    r"(?P<overs>\d+(?:\.\d+)?)\s*/\s*(?P<maxovers>\d+)\s+"
    r"Toss\s+(?P<toss>[A-Z]{2,3})\s+"
    r"(?P<striker>[A-Za-z]+)\s+(?P<sruns>\d+)\s*\((?P<sballs>\d+)\)\s+"
    r"(?P<nonstriker>[A-Za-z]+)\s+(?P<nsruns>\d+)\s*\((?P<nsballs>\d+)\)\s+"
    r"(?P<bowl>[A-Z]{2,3})",
    re.IGNORECASE
)


def parse_score_text(text):
    """
    Parses OCR text using the specific regex provided by user.
    Maps results to the existing dict structure for compatibility.
    """
    txt = re.sub(r"\s+", " ", (text or "")).strip()

    # Default output structure
    out = {
        "team1_name": None,
        "team2_name": None,
        "team1_score": {"runs": None, "wickets": None, "overs": None},
        "team2_score": {"runs": None, "wickets": None, "overs": None},
        "raw_text": txt,
        "parsed": None # raw dictionary from regex
    }

    m = score_pattern.search(txt)
    if not m:
        # Fallback: Just return raw text in structure
        return out

    g = m.groupdict()
    
    # Map to project structure
    # Assumption: "team" is batting team (Team 1), "bowl" is bowling team (Team 2)
    out["team1_name"] = g["team"]
    out["team1_score"]["runs"] = int(g["runs"])
    out["team1_score"]["wickets"] = int(g["wickets"])
    out["team1_score"]["overs"] = g["overs"]
    
    out["team2_name"] = g["bowl"]
    out["team2_score"]["runs"] = None 
    out["team2_score"]["wickets"] = None
    out["team2_score"]["overs"] = None

    out["extra_info"] = {
        "max_overs": int(g["maxovers"]),
        "toss": g["toss"],
        "striker": g["striker"],
        "striker_runs": int(g["sruns"]),
        "striker_balls": int(g["sballs"]),
        "nonstriker": g["nonstriker"],
        "nonstriker_runs": int(g["nsruns"]),
        "nonstriker_balls": int(g["nsballs"]),
    }
    
    return out


# ================= FRAME ANALYSIS =================

def analyze_score_frame(image_path, do_crop=True, api_key=None):
    """
    Analyze frame. If do_crop=True, applies specific Sony LIV bottom crop.
    Also applies User's preprocessing (Contrast/Sharpness).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        
        if do_crop:
            img = crop_scorecard(img)
        
        # Apply strict preprocessing
        processed_img = preprocess_for_ocr(img)
        processed_img = resize_for_ocr(processed_img)

        # To Base64
        base64_img = image_to_base64_bytes(processed_img)
        
        # Call API
        resp = call_vision_ocr(base64_img, api_key=api_key)

        if not resp["success"]:
            return {"error": resp["error"], "ocr_text": None, "parsed": None}

        text = resp["text"].strip() if resp["text"] else ""
        parsed = parse_score_text(text)

        return {"ocr_text": text, "parsed": parsed, "error": None}

    except Exception as e:
        return {"error": str(e), "ocr_text": None, "parsed": None}


# ================= BATCH PROCESSING =================

def process_score_frames(frames_dir: Path,
                         output_json_path: Path,
                         output_csv_path: Path):
    """
    Run scorecard OCR on subsampled frames using strict Single Key logic.
    """
    # 🔹 Use subsampled frames 
    frame_paths = get_sampled_frame_paths(frames_dir, FRAME_SUBSAMPLE)
    print(f"Running scorecard OCR on {len(frame_paths)} frames (subsample={FRAME_SUBSAMPLE}) ...")
    
    # Use Key 1 for EVERYTHING as requested
    active_key = OCR_KEYS[0] if OCR_KEYS else None

    if not active_key:
        print("[OCR] CRITICAL: No OCR keys found! Skipping OCR.")
        return []

    results = []

    for f in tqdm(frame_paths, desc="Scorecard OCR"):
        f_str = str(f)
        
        # Time Logic: < 40s = No Crop (Intro), > 40s = Crop (Ticker)
        do_crop = True
        m = re.search(r"frame_(\d+)\.jpg", f.name)
        if m:
            frame_idx = int(m.group(1))
            time_sec = frame_idx / float(FRAME_RATE)
            if time_sec < 40.0:
                do_crop = False
            else:
                do_crop = True
        
        r = analyze_score_frame(f_str, do_crop=do_crop, api_key=active_key)
        entry = {"frame": f.name}
        entry.update(r)
        results.append(entry)
        
        # Respect rate limit
        time.sleep(1.5)

    # Save JSON
    with open(output_json_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

    # Flatten to CSV
    rows = []
    for r in results:
        parsed = r.get("parsed") or {}
        rows.append({
            "frame": r.get("frame"),
            "ocr_text": (r.get("ocr_text") or "")[:500],
            "team1_name": parsed.get("team1_name"),
            "team1_runs": parsed.get("team1_score", {}).get("runs"),
            "team1_wickets": parsed.get("team1_score", {}).get("wickets"),
            "team1_overs": parsed.get("team1_score", {}).get("overs"),
            "team2_name": parsed.get("team2_name"), # Bowling team
            "error": r.get("error"),
        })

    pd.DataFrame(rows).to_csv(output_csv_path, index=False, encoding="utf-8")

    print(f"\n✅ Scorecard OCR results saved:")
    print(f"   JSON: {output_json_path}")
    print(f"   CSV : {output_csv_path}")

    return results
