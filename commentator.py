from pathlib import Path
import subprocess
import shutil
import time

# Import new modules
from config import (
    FRAME_RATE, CLIP_LENGTH, FRAME_SUBSAMPLE, 
    YOLO_WEIGHTS, SHOT_WEIGHTS, UMPIRE_WEIGHTS, RUNOUT_WEIGHTS, R2P1D_WEIGHTS,
    SHOT_META_JSON, UMPIRE_META_JSON, RUNOUT_META_JSON, R2P1D_META_JSON,
    SCORE_JSON, SCORE_CSV, FRAMES_DIR, CLIPS_DIR
)
from video_processing import run_ffmpeg_split
from ocr import process_score_frames
from models import (
    load_yolo_model, load_efficientnet_classifier, load_umpire_model, load_r2plus1d_model
)
from inference import run_on_frames, run_on_clips
from timeline import build_timeline
from llm import build_commentary_prompt_from_timeline, call_llm, summarize_text
from tts import synthesize_commentary_audio

class Commentator:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir.resolve()
        
        # Directories (Reuse global config names relative to base_dir if needed, 
        # but config.py uses absolute paths or "." relative. 
        # We'll assume usage of the global config paths for simplicity/consistency)
        self.frames_dir = FRAMES_DIR
        self.clips_dir = CLIPS_DIR
        
        # Models
        self.yolo_model = None
        self.shot_model = None
        self.umpire_model = None
        self.runout_model = None
        self.video_model = None
        
        # Classes
        self.shot_classes = []
        self.umpire_classes = []
        self.runout_classes = []
        self.video_classes = []
        
        self.models_loaded = False

    def load_models_lazy(self):
        if self.models_loaded:
            return

        print("Loading models...")
        self.yolo_model = load_yolo_model(YOLO_WEIGHTS)
        self.shot_model, self.shot_classes = load_efficientnet_classifier(SHOT_WEIGHTS, SHOT_META_JSON)
        self.umpire_model, self.umpire_classes = load_umpire_model(UMPIRE_WEIGHTS, UMPIRE_META_JSON)
        self.runout_model, self.runout_classes = load_efficientnet_classifier(RUNOUT_WEIGHTS, RUNOUT_META_JSON)
        self.video_model, self.video_classes = load_r2plus1d_model(R2P1D_WEIGHTS, R2P1D_META_JSON)
        
        self.models_loaded = True

    def merge_audio_video(self, video_path: Path, audio_path: Path, output_path: Path):
        """
        Mixes generated commentary audio with original video audio.
        Uses amix filter to play both.
        """
        print(f"Merging {video_path} + {audio_path} -> {output_path}")
        
        # Filter complex: 
        # [0:a]volume=0.4[a1];  <-- Original video audio at 40% volume (background)
        # [1:a]volume=1.0[a2];  <-- Commentary audio at 100% volume (foreground)
        # [a1][a2]amix=inputs=2:duration=first[aout] <-- Mix them, end when first input (video) ends
        # If original video has no audio stream, this might fail or warn. 
        # We can try a safer simple overlay if amix fails, but amix is best for "Commentary".
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-filter_complex", "[0:a]volume=0.4[a1];[1:a]volume=1.0[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",

            str(output_path)
        ]
        
        # Fallback command if video has NO audio stream (silent video)
        # In that case, we just map 1:a
        cmd_fallback = [
             "ffmpeg", "-y",
            "-i", str(video_path),
            "-i", str(audio_path),
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v",
            "-map", "1:a", # Use new audio as the only audio

            str(output_path)
        ]

        try:
            # Try mixing first
            print("Attempting to mix audio streams...")
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Mixing failed (maybe video has no audio?). Retrying with simple replacement. Error: {e.stderr.decode()}")
            try:
                subprocess.run(cmd_fallback, check=True)
                return True
            except subprocess.CalledProcessError as e2:
                print(f"Fallback merge failed: {e2}")
                return False

    def process_video(self, video_path: Path, has_scorecard: bool = True, update_callback=None):
        def notify(msg):
            print(f"[PIPELINE] {msg}")
            if update_callback: update_callback(msg)

        try:
            # 1. Split
            notify("Step 1/7: Analyzing video structure...")
            # We must use specific frames/clips dir if we want isolation, 
            # but for now we follow global config
            run_ffmpeg_split(video_path, self.frames_dir, self.clips_dir, FRAME_RATE, CLIP_LENGTH)
            
            # 2. OCR (Optional)
            score_by_frame = {}
            if has_scorecard:
                notify("Step 2/7: Reading scoreboard data...")
                score_results = process_score_frames(self.frames_dir, SCORE_JSON, SCORE_CSV)
                score_by_frame = {e["frame"]: e for e in score_results if e.get("frame")}
            else:
                notify("Step 2/7: OCR Skipped (No Scorecard selected)...")
            
            # 3. Models
            notify("Step 3/7: Loading AI models...")
            self.load_models_lazy()
            
            # 4. Inference
            notify("Step 4/7: Detecting events (Visual AI)...")
            frame_results = run_on_frames(
                self.frames_dir, self.yolo_model, self.shot_model, self.umpire_model, self.runout_model,
                self.shot_classes, self.umpire_classes, self.runout_classes
            )
            clip_results = run_on_clips(self.clips_dir, self.video_model, self.video_classes)
            
            # 5. Timeline & Prompt
            notify("Step 5/7: Generating Commentary Script...")
            timeline = build_timeline(frame_results, clip_results, score_by_frame)
            
            # Save Timeline for Match Analyst (Chat)
            import json
            with open(self.base_dir / "timeline_for_llm.json", "w") as f:
                json.dump(timeline, f, indent=2)

            prompt = build_commentary_prompt_from_timeline(timeline)
            commentary = call_llm(prompt)
            
            # Save raw commentary to file as requested
            try:
                (self.base_dir / "commentary_raw.txt").write_text(commentary, encoding="utf-8")
            except Exception as e:
                print(f"Could not save commentary file: {e}")
            
            if "[LLM ERROR]" in commentary:
                notify("Commentary generation failed.")
                return None
            
            # Length Check & Summarization Loop
            # Keep summarizing until under 9500 chars to ensure safety for edge-tts and file limits
            current_commentary = commentary
            iteration = 0
            max_iterations = 3
            
            while len(current_commentary) > 9500 and iteration < max_iterations:
                notify(f"Commentary long ({len(current_commentary)} chars). Summarizing (Attempt {iteration+1})...")
                new_summary = summarize_text(current_commentary, max_chars=9000) # Aim lower to be safe
                
                # If summarization failed or returned empty/error, break to avoid infinite loop of nothingness
                if not new_summary or len(new_summary) >= len(current_commentary) or "[LLM ERROR]" in new_summary:
                    notify("Summarization didn't reduce length sufficiently. Moving to hard truncation.")
                    break
                    
                current_commentary = new_summary
                iteration += 1
            
            # Final Safety Hard Truncation if MULTIPLE summarizations failed
            if len(current_commentary) > 9800:
                notify("Still too long. Forcing hard truncation to 9500 chars.")
                current_commentary = current_commentary[:9500]
                # Try to cut at last sentence
                last_dot = current_commentary.rfind('.')
                if last_dot > 0:
                    current_commentary = current_commentary[:last_dot+1]
            
            commentary = current_commentary
            
            # Save final processed commentary
            try:
                (self.base_dir / "commentary_final.txt").write_text(commentary, encoding="utf-8")
            except:
                pass
            
            # 6. TTS
            notify("Step 6/7: Synthesizing Audio Voice...")
            audio_out = self.base_dir / "commentary_output.mp3"
            success = synthesize_commentary_audio(commentary, audio_out)
            
            if not success:
                notify("TTS generation failed. Check API keys.")
                return None
                
            # 7. Merge
            notify("Step 7/7: Finalizing production...")
            final_vid = self.base_dir / "final_output.mp4"
            if self.merge_audio_video(video_path, audio_out, final_vid):
                notify("Displaying result now!")
                return str(final_vid.resolve())
            else:
                notify("Merge failed.")
                return None
                
        except Exception as e:
            notify(f"Processing Error: {e}")
            import traceback
            traceback.print_exc()
            return None
