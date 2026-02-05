
import cv2
import os

video_path = r"c:\project\Project-Display\endvideo.mp4"
output_path = r"c:\project\Project-Display\frames\test_crop_35.jpg"
timestamp_seconds = 35

def extract_frame():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Calculate frame number (assuming approx fps, or getting fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_no = int(fps * timestamp_seconds)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"Success: Saved frame at {timestamp_seconds}s to {output_path}")
    else:
        print(f"Error: Could not read frame at {timestamp_seconds}s")
    
    cap.release()

if __name__ == "__main__":
    if not os.path.exists("frames"):
        os.makedirs("frames")
    extract_frame()
