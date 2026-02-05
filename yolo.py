import cv2
from pathlib import Path
from ultralytics import YOLO

# ====================
# CONFIG
# ====================

CONF_THRESHOLD = 0.10   # 👈 Change this value (0.10 = 10%)
IMAGE_PATH = Path(r"C:\project\Project-Display\frames\frame_000059.jpg")
YOLO_WEIGHTS = Path(r"C:\project\Project-Display\models\yolo-best.pt")

# ====================
# LOAD MODEL
# ====================

model = YOLO(str(YOLO_WEIGHTS))

print("\n📌 YOLO Classes:")
for cid, cname in model.names.items():
    print(f"  {cid}: {cname}")

# ====================
# RUN YOLO
# ====================

img = cv2.imread(str(IMAGE_PATH))
if img is None:
    raise FileNotFoundError(f"❌ File not found: {IMAGE_PATH}")

results = model(img, conf=CONF_THRESHOLD, verbose=False)

# ====================
# PRINT DETECTIONS
# ====================

print(f"\n🔍 Detection results for: {IMAGE_PATH.name} (conf ≥ {CONF_THRESHOLD})")

if results:
    pred = results[0]
    if len(pred.boxes) == 0:
        print("❌ Nothing detected at this confidence.")
    else:
        for box in pred.boxes:
            cls_id = int(box.cls[0])
            cls_name = pred.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            print(f" ✔ {cls_name} ({conf:.2f}) → [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
else:
    print("❌ YOLO returned no results.")
