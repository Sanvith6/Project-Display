from pathlib import Path

def build_timeline(frame_results, clip_results, score_by_frame=None):
    """
    Build a time-ordered list of events.
    Each event carries SCOREBOARD OCR + YOLO + (optional) clip context.
    """
    clip_ranges = []
    for c in clip_results:
        st = c.get("start_time")
        et = c.get("end_time")
        if st is None or et is None:
            continue
        clip_ranges.append((st, et, c))
    clip_ranges.sort(key=lambda x: x[0])

    events = []
    for fr in frame_results:
        t = fr["time_sec"]
        clip_ctx = None
        for st, et, c in clip_ranges:
            if t >= st and t < et:
                clip_ctx = c
                break

        frame_path = fr["frame_path"]
        frame_name = Path(frame_path).name
        score_entry = score_by_frame.get(frame_name) if score_by_frame else None

        events.append({
            "time_sec": t,
            "frame_path": frame_path,
            "models": {
                "shot": fr["shot"],
                "umpire": fr["umpire"],
                "runout": fr["runout"],
                "yolo_detections": fr["yolo_detections"],
            },
            "score_ocr_text": (score_entry or {}).get("ocr_text"),
            "score_parsed": (score_entry or {}).get("parsed"),
            "clip_context": clip_ctx,
        })

    events.sort(key=lambda e: e["time_sec"])
    return events
