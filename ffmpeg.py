import cv2
from pathlib import Path
from collections import deque
import json
import sys


def extract_frames_and_clips(
    video_path,
    frames_dir="frames",
    clips_dir="clips",
    frame_rate_out=1.0,        # 1 frame per second (for saving frames)
    clip_len_seconds=6.0,      # length of each clip in seconds
    stride_seconds=None,       # seconds to advance between clip starts; None -> no-overlap
    clip_writer_codec='mp4v',  # codec for cv2.VideoWriter
    verbose=True
):
    # Setup output directories
    frames_dir = Path(frames_dir)
    clips_dir = Path(clips_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    if stride_seconds is None:
        stride_seconds = clip_len_seconds

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = None
    if total_frames > 0 and orig_fps > 0:
        duration_sec = total_frames / orig_fps

    if verbose:
        print(f"Video: {video_path}")
        print(f"Original FPS: {orig_fps:.3f}, total frames: {total_frames}, approx duration: {duration_sec}")

    frame_idx = 0
    saved_frame_count = 0

    # Next time (in seconds) to save a frame (1 FPS)
    next_frame_time = 0.0
    eps = 1e-3

    # For clips: start at t=0, then t=stride, t=2*stride, ...
    next_clip_start = 0.0
    clip_len = float(clip_len_seconds)
    stride = float(stride_seconds)

    # Buffer of recent frames: (timestamp_sec, frame_bgr, global_frame_idx)
    buf = deque()

    clip_counter = 0
    clips_metadata = []

    # VideoWriter config (lazy init)
    writer_initialized = False
    writer_width = None
    writer_height = None
    fourcc = cv2.VideoWriter_fourcc(*clip_writer_codec)

    if verbose:
        print("Starting processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Timestamp (seconds) for current frame
        timestamp = frame_idx / orig_fps if orig_fps > 0 else 0.0

        # Add to buffer
        buf.append((timestamp, frame.copy(), frame_idx))

        # Prune old frames from buffer (we keep some padding before next_clip_start)
        prune_before = max(0.0, next_clip_start - 1.0)
        while buf and buf[0][0] < prune_before:
            buf.popleft()

        # ============================
        # Save frames at ~1 FPS
        # ============================
        if timestamp + eps >= next_frame_time:
            out_path = frames_dir / f"frame_{saved_frame_count:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            if verbose:
                print(f"[frame] t={timestamp:.2f}s -> {out_path.name}")
            saved_frame_count += 1
            next_frame_time += 1.0  # next whole second

        # ============================
        # Create clips
        # ============================
        while timestamp + eps >= next_clip_start + eps:
            clip_start_t = next_clip_start
            clip_end_t = next_clip_start + clip_len

            # Get frames in [clip_start_t, clip_end_t)
            clip_frames = [f for (t, f, idx) in buf if (t >= clip_start_t - eps and t < clip_end_t - eps)]
            clip_frame_idxs = [idx for (t, f, idx) in buf if (t >= clip_start_t - eps and t < clip_end_t - eps)]

            if len(clip_frames) == 0:
                if verbose:
                    print(f"[clip] no frames for clip starting {clip_start_t:.2f}s, skipping.")
                next_clip_start += stride
                continue

            # Init writer size the first time
            if not writer_initialized:
                h, w = clip_frames[0].shape[:2]
                writer_width, writer_height = w, h
                writer_initialized = True

            # Ensure frame size matches writer size
            def ensure_size(f):
                if f.shape[1] != writer_width or f.shape[0] != writer_height:
                    return cv2.resize(f, (writer_width, writer_height))
                return f

            clip_name = f"clip_{clip_counter:06d}.mp4"
            clip_path = clips_dir / clip_name

            out = cv2.VideoWriter(str(clip_path), fourcc, orig_fps, (writer_width, writer_height))
            for f in clip_frames:
                out.write(ensure_size(f))
            out.release()

            meta = {
                "clip_index": clip_counter,
                "clip_name": clip_name,
                "start_time": round(clip_start_t, 3),
                "end_time": round(clip_end_t, 3),
                "num_frames": len(clip_frames),
                "frame_indexes": clip_frame_idxs
            }
            clips_metadata.append(meta)
            if verbose:
                print(f"[clip] saved {clip_name} start={clip_start_t:.2f}s end={clip_end_t:.2f}s frames={len(clip_frames)}")

            clip_counter += 1
            next_clip_start += stride

        frame_idx += 1

    cap.release()

    # ============================
    # Handle leftover windows at end
    # ============================
    if buf:
        last_time = buf[-1][0]
        while next_clip_start <= last_time:
            clip_start_t = next_clip_start
            clip_end_t = clip_start_t + clip_len

            clip_frames = [f for (t, f, idx) in buf if (t >= clip_start_t - eps and t < clip_end_t - eps)]
            clip_frame_idxs = [idx for (t, f, idx) in buf if (t >= clip_start_t - eps and t < clip_end_t - eps)]

            if clip_frames:
                if not writer_initialized:
                    h, w = clip_frames[0].shape[:2]
                    writer_width, writer_height = w, h
                    writer_initialized = True

                def ensure_size(f):
                    if f.shape[1] != writer_width or f.shape[0] != writer_height:
                        return cv2.resize(f, (writer_width, writer_height))
                    return f

                clip_name = f"clip_{clip_counter:06d}.mp4"
                clip_path = clips_dir / clip_name

                out = cv2.VideoWriter(str(clip_path), fourcc, orig_fps, (writer_width, writer_height))
                for f in clip_frames:
                    out.write(ensure_size(f))
                out.release()

                meta = {
                    "clip_index": clip_counter,
                    "clip_name": clip_name,
                    "start_time": round(clip_start_t, 3),
                    "end_time": round(clip_end_t, 3),
                    "num_frames": len(clip_frames),
                    "frame_indexes": clip_frame_idxs
                }
                clips_metadata.append(meta)
                if verbose:
                    print(f"[clip-final] saved {clip_name} start={clip_start_t:.2f}s end={clip_end_t:.2f}s frames={len(clip_frames)}")

                clip_counter += 1

            next_clip_start += stride

    # ============================
    # Save metadata
    # ============================
    meta_path = clips_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "clips": clips_metadata,
                "total_clips": clip_counter,
                "total_saved_frames": saved_frame_count
            },
            f,
            indent=2
        )

    if verbose:
        print(f"Done. saved {saved_frame_count} frames into '{frames_dir}', {clip_counter} clips into '{clips_dir}'.")
        print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    # Option 1: pass path via command line:
    #   python extract_frames_and_clips.py C:\path\to\video.mp4
    if len(sys.argv) >= 2:
        VIDEO_PATH = sys.argv[1]
    else:
        # Option 2: hardcode your video path here if you don't want CLI
        # Replace this with your actual path:
        VIDEO_PATH = r"C:\project\Project-Display\highlights1.mp4"

    extract_frames_and_clips(
        VIDEO_PATH,
        frames_dir="frames",
        clips_dir="clips",
        frame_rate_out=1.0,   # 1 frame per second
        clip_len_seconds=6.0, # 6-second clips
        stride_seconds=6.0,   # no overlap; e.g., 3.0 for 50% overlap
        clip_writer_codec='mp4v',
        verbose=True
    )
