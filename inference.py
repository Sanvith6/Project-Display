import cv2
import torch
import torchvision.transforms as T
from pathlib import Path

from config import DEVICE, FRAME_SUBSAMPLE, FRAME_RATE, CLIP_LENGTH
from video_processing import get_sampled_frame_paths, frame_index_from_name, load_video_as_tensor

image_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
])

def run_on_frames(frames_dir: Path,
                  yolo_model,
                  shot_model,
                  umpire_model,
                  runout_model,
                  shot_classes,
                  umpire_classes,
                  runout_classes):

    # 🔹 Use the same subsampled frames as OCR
    frame_files = get_sampled_frame_paths(frames_dir, FRAME_SUBSAMPLE)
    print(f"Found {len(frame_files)} sampled frames for inference (subsample={FRAME_SUBSAMPLE}).")

    results = []

    for fpath in frame_files:
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"WARNING: failed to read frame {fpath}")
            continue

        # Recover original frame index & time based on filename
        frame_index = frame_index_from_name(fpath)
        time_sec = frame_index * (1.0 / FRAME_RATE)

        # --- 1. YOLO (The Gatekeeper) ---
        # TUNED: conf=0.45 to reduce false positives, imgsz=1280 for small objects (stumps)
        yolo_out = yolo_model(frame, verbose=False, conf=0.45, imgsz=1280)
        detections = []
        detected_names = set()
        
        if len(yolo_out) > 0:
            pred = yolo_out[0]
            for box in pred.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = pred.names.get(cls_id, str(cls_id))
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                })
                detected_names.add(cls_name.lower())

        # Define Triggers based on YOLO output
        shot_trigger = any(x in detected_names for x in ["batsman", "batter", "player", "person"])
        ump_trigger = ("umpire" in detected_names or "official" in detected_names)
        runout_trigger = any(x in detected_names for x in ["stump", "stumps", "wicket", "wickets"])
        
        # --- 2. Lazy Transformation (Optimization) ---
        img_tensor = None
        
        if shot_trigger or ump_trigger or runout_trigger:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = image_transform(img_rgb).unsqueeze(0).to(DEVICE)
        
        # --- 3. Conditional Inference ---
        
        # A) SHOT
        if shot_trigger and img_tensor is not None:
            with torch.no_grad():
                logits_shot = shot_model(img_tensor)
                probs_shot = torch.softmax(logits_shot, dim=1)[0]
                shot_prob, shot_idx = torch.max(probs_shot, dim=0)
                shot_label = shot_classes[int(shot_idx)]
                shot_conf  = float(shot_prob)
        else:
            shot_label = "no_detection"
            shot_conf = 0.0

        # B) UMPIRE
        if ump_trigger and img_tensor is not None:
            with torch.no_grad():
                logits_ump = umpire_model(img_tensor)
                probs_ump = torch.softmax(logits_ump, dim=1)[0]
                ump_prob, ump_idx = torch.max(probs_ump, dim=0)
                ump_label = umpire_classes[int(ump_idx)]
                ump_conf  = float(ump_prob)
        else:
            ump_label = "no_detection"
            ump_conf = 0.0

        # C) RUNOUT
        if runout_trigger and img_tensor is not None:
            with torch.no_grad():
                logits_run = runout_model(img_tensor)
                probs_run = torch.softmax(logits_run, dim=1)[0]
                run_prob, run_idx = torch.max(probs_run, dim=0)
                run_label = runout_classes[int(run_idx)]
                run_conf  = float(run_prob)
        else:
            run_label = "no_detection"
            run_conf = 0.0

        results.append({
            "frame_index": frame_index,
            "frame_path": str(fpath),
            "time_sec": time_sec,
            "yolo_detections": detections,
            "shot":   {"label": shot_label, "confidence": shot_conf},
            "umpire": {"label": ump_label, "confidence": ump_conf},
            "runout": {"label": run_label, "confidence": run_conf},
        })

    return results


def run_on_clips(clips_dir: Path, video_model, video_classes):

    clip_files = sorted(clips_dir.glob("clip_*.mp4"))
    print(f"Found {len(clip_files)} clips for R(2+1)D.")

    results = []

    for cpath in clip_files:
        cname = cpath.name
        idx_str = cname.replace("clip_", "").replace(".mp4", "")
        try:
            clip_index = int(idx_str)
        except ValueError:
            clip_index = None

        start_time = clip_index * CLIP_LENGTH if clip_index is not None else None
        end_time   = start_time + CLIP_LENGTH if start_time is not None else None

        try:
            video_tensor = load_video_as_tensor(
                cpath, num_frames=16, resize_hw=(112, 112)
            )
        except Exception as e:
            print(f"ERROR reading clip {cpath}: {e}")
            continue

        video_tensor = video_tensor.unsqueeze(0).to(DEVICE)

        try:
            with torch.no_grad():
                logits = video_model(video_tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                top_prob, top_idx = torch.max(probs, dim=0)
                label = video_classes[int(top_idx)]
                conf  = float(top_prob)
        except Exception as e:
            print(f"ERROR running R(2+1)D on {cpath}: {e}")
            continue

        results.append({
            "clip_name": cname,
            "clip_path": str(cpath),
            "clip_index": clip_index,
            "start_time": start_time,
            "end_time": end_time,
            "video_class": {"label": label, "confidence": conf},
        })

    return results
