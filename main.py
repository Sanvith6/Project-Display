import time
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

# Config
from config import (
    BASE_DIR, VIDEO_PATH, FRAMES_DIR, CLIPS_DIR, FRAME_RATE, CLIP_LENGTH,
    SCORE_JSON, SCORE_CSV, RAW_RESULTS_JSON, TIMELINE_JSON, PROMPT_TXT, TTS_OUTPUT,
    YOLO_WEIGHTS, SHOT_WEIGHTS, UMPIRE_WEIGHTS, RUNOUT_WEIGHTS, R2P1D_WEIGHTS,
    SHOT_META_JSON, UMPIRE_META_JSON, RUNOUT_META_JSON, R2P1D_META_JSON
)

# Modules
from video_processing import run_ffmpeg_split
from ocr import process_score_frames
from models import (
    load_yolo_model, load_efficientnet_classifier, load_umpire_model, load_r2plus1d_model
)
from inference import run_on_frames, run_on_clips
from timeline import build_timeline
from llm import build_commentary_prompt_from_timeline, call_llm
from tts import synthesize_commentary_audio

def main():
    overall_start = time.time()
    stage_times = {}

    # --- STEP 1: Splitting video ---
    t0 = time.time()
    print("=== STEP 1: Splitting video with ffmpeg ===")
    run_ffmpeg_split(VIDEO_PATH, FRAMES_DIR, CLIPS_DIR, FRAME_RATE, CLIP_LENGTH)
    stage_times["ffmpeg_split"] = time.time() - t0

    # --- STEP 1b: Scorecard OCR ---
    t0 = time.time()
    print("=== STEP 1b: Running scorecard OCR on sampled frames ===")
    score_results = process_score_frames(FRAMES_DIR, SCORE_JSON, SCORE_CSV)
    stage_times["scorecard_ocr"] = time.time() - t0

    # Build quick lookup: frame_name -> score_entry
    score_by_frame = {
        entry["frame"]: entry
        for entry in score_results
        if entry.get("frame") is not None
    }

    # --- STEP 2: Load models ---
    t0 = time.time()
    print("=== STEP 2: Loading models ===")
    yolo_model = load_yolo_model(YOLO_WEIGHTS)

    shot_model,   shot_classes   = load_efficientnet_classifier(SHOT_WEIGHTS,   SHOT_META_JSON)
    umpire_model, umpire_classes = load_umpire_model(UMPIRE_WEIGHTS,           UMPIRE_META_JSON)
    runout_model, runout_classes = load_efficientnet_classifier(RUNOUT_WEIGHTS, RUNOUT_META_JSON)
    video_model,  video_classes  = load_r2plus1d_model(R2P1D_WEIGHTS,          R2P1D_META_JSON)
    stage_times["load_models"] = time.time() - t0

    # --- STEP 3: Frame inference ---
    t0 = time.time()
    print("=== STEP 3: Inference on sampled frames ===")
    # Note: we pass class lists now, as they are returned by load functions
    frame_results = run_on_frames(
        FRAMES_DIR,
        yolo_model, 
        shot_model, 
        umpire_model, 
        runout_model,
        shot_classes,
        umpire_classes,
        runout_classes
    )
    stage_times["frame_inference"] = time.time() - t0

    # --- STEP 4: Clip inference (R(2+1)D) ---
    t0 = time.time()
    print("=== STEP 4: Inference on clips (R(2+1)D) ===")
    clip_results = run_on_clips(CLIPS_DIR, video_model, video_classes)
    stage_times["clip_inference"] = time.time() - t0

    # Save raw outputs
    t0 = time.time()
    all_outputs = {"frames": frame_results, "clips": clip_results}
    with open(RAW_RESULTS_JSON, "w") as f:
        json.dump(all_outputs, f, indent=2)
    print(f"Saved raw model outputs to {RAW_RESULTS_JSON}")
    stage_times["save_raw_outputs"] = time.time() - t0

    # --- STEP 5: Timeline (with OCR attached) ---
    t0 = time.time()
    print("=== STEP 5: Building timeline (match frames to clips + OCR) ===")
    timeline = build_timeline(frame_results, clip_results, score_by_frame=score_by_frame)
    with open(TIMELINE_JSON, "w") as f:
        json.dump({"events": timeline}, f, indent=2)
    print(f"Saved timeline to {TIMELINE_JSON}")
    stage_times["build_timeline"] = time.time() - t0

    # --- STEP 7: Build LLM prompt from timeline ---
    t0 = time.time()
    print("=== STEP 7: Building LLM prompt (timeline-based) ===")
    prompt = build_commentary_prompt_from_timeline(timeline)
    with open(PROMPT_TXT, "w", encoding="utf-8") as f:
        f.write(prompt)
    print(f"Saved LLM prompt to {PROMPT_TXT}")
    stage_times["build_prompt"] = time.time() - t0

    # --- STEP 8: LLM commentary ---
    t0 = time.time()
    print("=== STEP 8: Calling LLM for commentary ===")
    commentary = call_llm(prompt)
    stage_times["llm_commentary"] = time.time() - t0

    print("\n===== GENERATED COMMENTARY =====\n")
    print(commentary)

    # Optional: quick peek at last score entry
    if score_results:
        print("\n===== SAMPLE SCORECARD OCR ENTRY (last sampled frame) =====")
        print(json.dumps(score_results[-1], indent=2, ensure_ascii=False))

    # --- STEP 9: TTS ---
    t0 = time.time()
    if commentary and not commentary.startswith("[LLM ERROR]"):
        print("\n=== STEP 9: Converting commentary to audio (ElevenLabs) ===")
        synthesize_commentary_audio(commentary, TTS_OUTPUT)
    else:
        print("[TTS] Skipping TTS because commentary generation failed or returned an error.")
    stage_times["tts"] = time.time() - t0

    # --- Final latency report ---
    total_time = time.time() - overall_start
    print("\n===== PIPELINE LATENCY REPORT =====")
    for stage, secs in stage_times.items():
        print(f"{stage:20s}: {secs:6.2f} s")
    print(f"{'TOTAL (start→end)':20s}: {total_time:6.2f} s")

    # Save to JSON for logging
    with open(BASE_DIR / "latency_report.json", "w") as f:
        json.dump({"stages": stage_times, "total_seconds": total_time}, f, indent=2)
        print(f"Saved latency report to {BASE_DIR / 'latency_report.json'}")


if __name__ == "__main__":
    main()
