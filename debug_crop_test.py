from PIL import Image
import os
from PIL import Image
import os
# from config import ROI_FRACTION_TOP, ROI_FRACTION_BOTTOM

ROI_FRACTION_TOP = 0.88  # Matching current config
ROI_FRACTION_BOTTOM = 1.0

# Path to the frame that started failing
FRAME_PATH = r"c:\project\Project-Display\frames\frame_000030.jpg"
OUTPUT_PATH = "debug_crop.jpg"

if not os.path.exists(FRAME_PATH):
    print(f"Error: {FRAME_PATH} not found.")
    exit()

img = Image.open(FRAME_PATH)
width, height = img.size

# Current settings (New fix: 0.88)
top = int(height * ROI_FRACTION_TOP)
bottom = int(height * ROI_FRACTION_BOTTOM)

print(f"Image Size: {width}x{height}")
print(f"Cropping from Y={top} to Y={bottom} (Height={bottom-top})")

cropped = img.crop((0, top, width, bottom))
cropped.save(OUTPUT_PATH)
print(f"Saved crop preview to {OUTPUT_PATH}")
