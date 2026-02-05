from ocr import analyze_score_frame
import json
from pathlib import Path

# Test frame 42 (42 seconds in) - Expected to be a match frame
test_file = r"c:\project\Project-Display\frames\frame_000042.jpg"

print(f"Testing Optimized OCR on: {test_file}")
# Manual Key 1 just for this test
api_key = "K85194943688957" 

# do_crop=True means "Use Sony LIV Crop 0.83-0.96"
result = analyze_score_frame(test_file, do_crop=True, api_key=api_key)

print("\n" + "="*40)
print("OCR RAW TEXT:")
print("="*40)
print(result["ocr_text"])

print("\n" + "="*40)
print("PARSED RESULT:")
print("="*40)
print(json.dumps(result["parsed"], indent=2))
