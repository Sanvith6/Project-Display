import subprocess
import re
import cv2
import numpy as np
import torch
from pathlib import Path

# Import the OpenCV-based splitter from our local ffmpeg.py
from ffmpeg import extract_frames_and_clips

def run_ffmpeg_split(video_path: Path, frames_dir: Path, clips_dir: Path,
                     frame_rate: int = 1, clip_len: int = 6):
    """
    Splits video into frames and clips using OpenCV (via ffmpeg.py).
    Does NOT require external ffmpeg binary in PATH.
    """
    print(f"[VIDEO_PROC] Splitting {video_path} into frames/clips using OpenCV...")
    extract_frames_and_clips(
        video_path=video_path,
        frames_dir=frames_dir,
        clips_dir=clips_dir,
        frame_rate_out=float(frame_rate),
        clip_len_seconds=float(clip_len),
        verbose=True
    )



def get_sampled_frame_paths(frames_dir: Path, step: int = 1):
    """
    Returns a subsampled, sorted list of frame Paths.
    If step=2 → every 2nd frame; if step=1 → all frames.
    """
    all_frames = sorted(frames_dir.glob("frame_*.jpg"))
    if step <= 1:
        return all_frames
    return all_frames[::step]


def frame_index_from_name(path: Path) -> int:
    """
    Parse original 0-based frame index from filename frame_000001.jpg.
    ffmpeg creates 1-based numbering, so index = number - 1.
    """
    m = re.search(r"frame_(\d+)\.jpg", path.name)
    if not m:
        return 0
    return int(m.group(1)) - 1


def load_video_as_tensor(video_path: Path,
                         num_frames: int = 16,
                         resize_hw=(112, 112)) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"No frames in video: {video_path}")

    indices = torch.linspace(0, total_frames - 1, steps=num_frames).long().tolist()

    frames = []
    cur_idx = 0
    target_ptr = 0
    target_idx = indices[target_ptr]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cur_idx == target_idx:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, resize_hw)
            frames.append(frame_resized)

            target_ptr += 1
            if target_ptr >= len(indices):
                break
            target_idx = indices[target_ptr]

        cur_idx += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Failed to sample frames from video: {video_path}")

    frames_np = np.stack(frames).astype("float32") / 255.0  # (T, H, W, C)
    frames_tensor = torch.from_numpy(frames_np)             # (T, H, W, C)
    frames_tensor = frames_tensor.permute(3, 0, 1, 2)       # (C, T, H, W)
    return frames_tensor
