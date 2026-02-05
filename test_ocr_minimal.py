
import requests
import io
import os
from PIL import Image

# Configuration from .env
API_KEYS = ["K85580927588957", "K82421332388957"]
ROI_FRACTION_TOP = 0.70
ROI_FRACTION_BOTTOM = 1.0

def resize_for_ocr(img_pil: Image.Image, max_width=1200) -> Image.Image:
    w, h = img_pil.size
    if w <= max_width:
        return img_pil
    new_h = int(h * (max_width / w))
    return img_pil.resize((max_width, new_h), Image.Resampling.LANCZOS)


# -------------------------------------------------------------------
# Parsing Logic (Copied from ocr.py for verification)
# -------------------------------------------------------------------
import re

# generic team pattern (fallback)
team_score_pattern = re.compile(
    r"(?P<team>[A-Za-z .&'()-]{2,40})\s+?(?P<runs>\d{1,4})\s*[\/\\-]?\s*(?P<wickets>\d{1,2})?\s*(?:\(|\[)?\s*(?P<overs>\d{1,2}(?:\.\d)?)?",
    re.IGNORECASE
)

# compact TV strip pattern
overlay_score_pattern = re.compile(
    r"\b(?P<bat_team>[A-Za-z]{2,4})\s+"
    r"(?P<runs>\d+)\s*[-/]\s*(?P<wickets>\d+)\s+"
    r"(?P<phase>[A-Za-z])?\s*"
    r"(?P<overs>\d+\.\d|\d+)\s*/\s*(?P<max_overs>\d+)\s+"
    r"Toss\s+(?P<toss_team>[A-Za-z]{2,4})\s*"
    r"(?:[^A-Za-z0-9\s]\s*)?"
    r"(?P<striker>[A-Za-z]+)\s+(?P<striker_runs>\d+)\*?\s*\((?P<striker_balls>\d+)\)\s+"
    r"(?P<nonstriker>[A-Za-z]+)\s+(?P<nonstriker_runs>\d+)\s*\((?P<nonstriker_balls>\d+)\)\s+"
    r"(?P<bowl_team>[A-Za-z]{2,4})"
    r"(?:\s+(?P<speed_mph>\d+)\s*mph\s*/\s*(?P<speed_kph>\d+)\s*kph)?",
    re.IGNORECASE
)

def parse_score_text(text):
    txt = re.sub(r"\s+", " ", (text or "")).strip()
    out = {
        "team1_name": None, "team1_score": {"runs": None, "wickets": None, "overs": None},
        "team2_name": None, "team2_score": {"runs": None, "wickets": None, "overs": None},
        "raw_text": txt,
    }

    # 1) Compact overlay
    m_overlay = overlay_score_pattern.search(txt)
    if m_overlay:
        gd = m_overlay.groupdict()
        out["team1_name"] = (gd.get("bat_team") or "").strip()
        out["team2_name"] = (gd.get("bowl_team") or "").strip()
        out["team1_score"]["runs"] = int(gd["runs"]) if gd.get("runs") else None
        out["team1_score"]["wickets"] = int(gd["wickets"]) if gd.get("wickets") else None
        out["team1_score"]["overs"] = gd.get("overs")
        out["parsing_method"] = "Compact Overlay"
        return out

    # 2) Generic
    matches = list(team_score_pattern.finditer(txt))
    if matches:
        m1 = matches[0]
        out["team1_name"] = (m1.group("team") or "").strip()
        out["team1_score"]["runs"] = int(m1.group("runs")) if m1.group("runs") else None
        out["team1_score"]["wickets"] = int(m1.group("wickets")) if m1.group("wickets") else None
        out["team1_score"]["overs"] = m1.group("overs")
        out["parsing_method"] = "Generic Pattern"
        return out

    return out

# -------------------------------------------------------------------

def test_ocr(image_path):
    print(f"Loading image: {image_path}")
    if not os.path.exists(image_path):
        print("Image not found!")
        return

    img = Image.open(image_path)
    w, h = img.size
    
    # Apply Crop Logic (Same as ocr.py)
    top_y = int(h * ROI_FRACTION_TOP)
    bottom_y = int(h * ROI_FRACTION_BOTTOM)
    box = (0, top_y, w, bottom_y)
    
    print(f"Original Size: {w}x{h}")
    print(f"Cropping to: {box} (Top {ROI_FRACTION_TOP*100}% cut off, keeping bottom {100-int(ROI_FRACTION_TOP*100)}%)")
    
    img_cropped = img.crop(box)
    img_resized = resize_for_ocr(img_cropped)
    
    # Convert to bytes
    buf = io.BytesIO()
    img_resized.save(buf, format='JPEG', quality=95)
    image_bytes = buf.getvalue()

    # Call OCR API with PRIMARY KEY
    data = {
        "apikey": API_KEYS[0], 
        "language": "eng",
        "isOverlayRequired": False,
        "OCREngine": 2,
        "scale": True
    }
    
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    
    print("Sending to OCR.Space...")
    try:
        resp = requests.post("https://api.ocr.space/parse/image", data=data, files=files, timeout=30)
        print(f"Status Code: {resp.status_code}")
        
        try:
            result = resp.json()
        except Exception:
            print("Response is not JSON!")
            print(resp.text[:500])
            return

        if not isinstance(result, dict):
             print(f"Unexpected response type: {type(result)}")
             print(result)
             return

        parsed_results = result.get("ParsedResults")
        if parsed_results:
            text = parsed_results[0].get("ParsedText")
            print("\n" + "="*20)
            print("OCR RESULT SUCCCESS:")
            print("="*20)
            print(text)
            print("="*20)
            
            # VERIFY PARSING
            print("\n--- VERIFYING COLUMN EXTRACTION ---")
            parsed = parse_score_text(text)
            print(f"Parsing Method: {parsed.get('parsing_method', 'None')}")
            print(f"Team 1 Name : {parsed['team1_name']}")
            print(f"Team 1 Runs : {parsed['team1_score']['runs']}")
            print(f"Team 1 Wkts : {parsed['team1_score']['wickets']}")
            print(f"Team 1 Overs: {parsed['team1_score']['overs']}")
            print("-------------------------------------")

        else:
            print("No parsed results found in JSON.")
            print(result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_path = r"c:\project\Project-Display\frames\frame_000081.jpg" # A known "Empty" frame from CSV
    # Override for user's specific test if needed, but let's stick to the generated crop frame
    test_ocr(test_path)
