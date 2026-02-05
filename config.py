import os
from pathlib import Path
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None
    DEVICE = "cpu"


# =============================
# CONFIG
# =============================

BASE_DIR = Path(".").resolve()

# Main video for event detection + commentary
VIDEO_PATH = BASE_DIR / "endvideo.mp4"

FRAMES_DIR = BASE_DIR / "frames"
CLIPS_DIR = BASE_DIR / "clips"

# Model weights
YOLO_WEIGHTS   = BASE_DIR / "models" / "yolo-best.pt"
SHOT_WEIGHTS   = BASE_DIR / "models" / "best_shot_3class.pth"
UMPIRE_WEIGHTS = BASE_DIR / "models" / "best_umpire.pth"
RUNOUT_WEIGHTS = BASE_DIR / "models" / "best_runout_3class.pth"
R2P1D_WEIGHTS  = BASE_DIR / "models" / "R(2+1)best.pt"

# Optional JSON metadata files
SHOT_META_JSON   = BASE_DIR / "models" / "best_shot_3class.json"
UMPIRE_META_JSON = BASE_DIR / "models" / "best_umpire.json"
RUNOUT_META_JSON = BASE_DIR / "models" / "best_runout_3class.json"
R2P1D_META_JSON  = BASE_DIR / "models" / "R(2+1)best.json"

# Output files
RAW_RESULTS_JSON = BASE_DIR / "model_outputs.json"
TIMELINE_JSON    = BASE_DIR / "timeline_for_llm.json"
PROMPT_TXT       = BASE_DIR / "commentary_prompt.txt"

# 🔹 Scorecard OCR outputs
SCORE_JSON       = BASE_DIR / "score_data.json"
SCORE_CSV        = BASE_DIR / "score_data.csv"

# 🔹 TTS output
TTS_OUTPUT       = BASE_DIR / "commentary_audio.mp3"

# ffmpeg settings
FRAME_RATE  = 1   # 1 frame per second from ffmpeg
CLIP_LENGTH = 6   # 6-second clips

# 🔹 New: subsample heavy processing
# 🔹 New: subsample heavy processing
FRAME_SUBSAMPLE = 4  # Check every 3 seconds



# Load .env file
from dotenv import load_dotenv
load_dotenv(override=True)

# ===== LLM (Jina DeepSearch – OpenAI-compatible Chat API) =====
JINA_MODEL_ID = "jina-deepsearch-v1"
JINA_BASE_URL = "https://deepsearch.jina.ai/v1"
JINA_API_KEY = os.getenv("JINA_API_KEY")

# ===== OCR.Space API Key =====
OCRSPACE_API_KEY = os.getenv("OCRSPACE_API_KEY", "YOUR_OCRSPACE_API_KEY") # Primary
OCRSPACE_API_KEY_2 = os.getenv("OCRSPACE_API_KEY_2", "")                 # Secondary

# Store in a list for rotation. Filter out placeholders/empty.
OCR_KEYS = []
if OCRSPACE_API_KEY and len(OCRSPACE_API_KEY) > 10: 
    OCR_KEYS.append(OCRSPACE_API_KEY)
if OCRSPACE_API_KEY_2 and len(OCRSPACE_API_KEY_2) > 10: 
    OCR_KEYS.append(OCRSPACE_API_KEY_2)
if OCRSPACE_API_KEY_2 and len(OCRSPACE_API_KEY_2) > 10: 
    OCR_KEYS.append(OCRSPACE_API_KEY_2)

# Fallback if none found
if not OCR_KEYS:
    OCR_KEYS = ["helloworld"] # Default free key often works for testing

# ===== ElevenLabs TTS =====
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

# Crop region (bottom portion below the white line)
ROI_FRACTION_TOP = 0.70 # Relaxed to bottom 30% to ensure scorecard is not cut off
ROI_FRACTION_BOTTOM = 1.0

# OCR.Space endpoint
VISION_URL = "https://api.ocr.space/parse/image"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0
