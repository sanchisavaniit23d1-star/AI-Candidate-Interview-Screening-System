import cv2
import os
import numpy as np

VIDEO_FILE = "test_sample_1.mp4"

TARGET_FRAMES = 15
TARGET_SIZE = (224, 224)
FPS = 30
DURATION = 3.6


def select_distributed(m, n):
    return [
        i * n // m + n // (2 * m)
        for i in range(m)
    ]


def preprocess_video(video_file):

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Original video information:")
    print("Total frames:", total_frames)
    print("FPS:", fps)

    required_frames = int(FPS * DURATION)

    # Use the first 3.6 seconds for this initial test
    frames_to_use = min(required_frames, total_frames)

    selected_indices = select_distributed(
        TARGET_FRAMES,
        frames_to_use
    )

    faces = []

    for index in selected_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", index)
            continue

        # OpenCV uses BGR; convert to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # For this first test, we are not doing face detection yet.
        # We simply resize the video frames.
        frame = cv2.resize(
            frame,
            TARGET_SIZE
        )

        faces.append(frame)

    cap.release()

    return np.array(faces)


if __name__ == "__main__":

    frames = preprocess_video(VIDEO_FILE)

    print("\nVideo preprocessing successful!")
    print("Output shape:", frames.shape)
    print("Expected shape:", "(15, 224, 224, 3)")

    os.makedirs("test_frames", exist_ok=True)

    for i, frame in enumerate(frames):

        # Convert RGB back to BGR for OpenCV saving
        frame_bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        filename = f"test_frames/frame_{i + 1:02d}.jpg"

        cv2.imwrite(
            filename,
            frame_bgr
        )

    print("Saved frames to: test_frames/")