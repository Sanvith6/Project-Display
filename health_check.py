
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import importlib

# 1. Environment Setup
BASE_DIR = Path(".").resolve()
load_dotenv(BASE_DIR / ".env")

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def check(condition, message):
    if condition:
        print(f"{GREEN}[OK]{RESET} {message}")
    else:
        print(f"{RED}[FAIL]{RESET} {message}")

def check_warn(condition, message):
    if condition:
        print(f"{GREEN}[OK]{RESET} {message}")
    else:
        print(f"{YELLOW}[WARN]{RESET} {message}")

print(f"\n--- Cricket Commentary AI System Health Check ---\n")

# 2. File Structure Verification
print(">>> Checking File Structure")
required_files = [
    "app.py",
    "commentator.py",
    "llm.py",
    "tts.py",
    "cricket_server.py",
    "config.py",
    "video_processing.py",
    "static/index.html",
    "static/style.css",
    "static/script.js",
    "static/bg.png"
]

for f in required_files:
    check((BASE_DIR / f).exists(), f"File exists: {f}")

# 3. Environment Variables
print("\n>>> Checking Configurations (.env)")
eleven_key = os.getenv("ELEVENLABS_API_KEY")
voice_id = os.getenv("ELEVENLABS_VOICE_ID")
jina_key = os.getenv("JINA_API_KEY")
ocr_key = os.getenv("OCRSPACE_API_KEY")

check(eleven_key and len(eleven_key) > 5, "ELEVENLABS_API_KEY is set")
check_warn(voice_id, "ELEVENLABS_VOICE_ID is set (Required for custom voice)")
check(jina_key and len(jina_key) > 5, "JINA_API_KEY is set")
check(ocr_key and len(ocr_key) > 5, "OCRSPACE_API_KEY is set")

# 4. Import Verification
print("\n>>> Checking Module Imports")
try:
    import fastapi
    check(True, "FastAPI installed")
except ImportError:
    check(False, "FastAPI missing")

try:
    import elevenlabs
    check(True, "ElevenLabs SDK installed")
except ImportError:
    check(False, "ElevenLabs SDK missing")

try:
    import cv2
    check(True, "OpenCV installed")
except ImportError:
    check(False, "OpenCV (cv2) missing")

# 5. Logic Inspection (Current Pipeline)
print("\n>>> Logic Inspection")

try:
    from ocr import process_score_frames
    check(True, "OCR module (process_score_frames) loaded")
except ImportError:
    check(False, "OCR module missing or broken")

try:
    from inference import run_on_frames
    check(True, "Inference module (run_on_frames) loaded")
except ImportError:
    check(False, "Inference module missing or broken")

try:
    from llm import summarize_text, build_commentary_prompt_from_timeline
    check(True, "LLM module loaded")
except ImportError:
    check(False, "LLM module missing")

print("\n--- End of Health Check ---")
