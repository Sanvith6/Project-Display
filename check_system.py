import os
import sys
from pathlib import Path
import time

print("🔍 STARTING SYSTEM HEALTH CHECK...")
print("="*40)

# 1. Config Check
print("\n[1] Checking Configuration...")
try:
    from config import OCR_KEYS, VISION_URL, JINA_API_KEY, FRAME_SUBSAMPLE, DEVICE
    print(f"   ✅ Config Loaded.")
    print(f"   - Device: {DEVICE}")
    print(f"   - OCR Keys Available: {len(OCR_KEYS)}")
    print(f"   - Jina Key: {'Found' if JINA_API_KEY else 'MISSING'}")
    if len(OCR_KEYS) == 0:
        print("   ❌ CRITICAL: No OCR Keys found in .env!")
except Exception as e:
    print(f"   ❌ Config Error: {e}")

# 2. OCR Module Check
print("\n[2] Checking OCR Module...")
try:
    from ocr import call_vision_ocr, parse_score_text
    print("   ✅ OCR Module Imported.")
    # Test Parser
    dummy_text = "IND 240/3 (45.2) vs AUS"
    parsed = parse_score_text(dummy_text)
    if parsed["team1_name"] == "IND":
        print("   ✅ Parser Logic Verified.")
    else:
        print(f"   ❌ Parser Logic Error: {parsed}")
except Exception as e:
    print(f"   ❌ OCR Module Error: {e}")

# 3. Model Imports Check
print("\n[3] Checking AI Models / Inference...")
try:
    from inference import run_on_frames
    # We won't load models fully (takes too long), but we check imports
    import models
    print("   ✅ Inference & Models Modules Imported.")
except ImportError as e:
    print(f"   ❌ Model Import Error (Torch/Ultralytics?): {e}")
except Exception as e:
    print(f"   ❌ Model Error: {e}")

# 4. Pipeline Check
print("\n[4] Checking Pipeline Class...")
try:
    from commentator import Commentator
    base_dir = Path(".").resolve()
    comm = Commentator(base_dir)
    print("   ✅ Commentator Class Initialized.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"   ❌ Commentator Init Error: {e}")

print("\n" + "="*40)
print("🏁 CHECK COMPLETE")
print("="*40)
