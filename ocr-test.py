import os
import sys

# Add current directory to path so we can import modules
sys.path.append(os.getcwd())

from ocr import analyze_score_frame, OCR_KEYS

if __name__ == "__main__":
    # Test on the problematic frame 30 to verify crop
    test_image_path = Path(r"c:\project\Project-Display\frames\test_crop_35.jpg")
    
    print(f"Testing OCR Module with {len(OCR_KEYS)} keys loaded.")
    print(f"Target Image: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print("Test image not found.")
        exit()

    try:
        # Calls the REAL function from ocr.py (with resize, 0.88 crop, key rotation)
        result = analyze_score_frame(test_image, do_crop=True)
        
        print("\n===== OCR RESULT =====")
        print(f"Text: {result.get('ocr_text')}")
        print(f"Parsed: {result.get('parsed')}")
        print(f"Error: {result.get('error')}")
        
    except Exception as e:
        print("ERROR:", e)
