import cv2
import os
import numpy as np

VIDEO_FILE = "test_sample_1.mp4"

TARGET_FRAMES = 15
TARGET_SIZE = (224, 224)

FPS = 30
WINDOW_DURATION = 3.6


def select_distributed(m, n):
    return [
        i * n // m + n // (2 * m)
        for i in range(m)
    ]


def preprocess_video(video_file):

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Original video information:")
    print("Total frames:", total_frames)
    print("FPS:", fps)

    frames_per_window = int(
        FPS * WINDOW_DURATION
    )

    total_windows = total_frames // frames_per_window

    print("Frames per window:", frames_per_window)
    print("Number of complete windows:", total_windows)

    all_frames = []

    for window in range(total_windows):

        window_start = window * frames_per_window

        print(
            f"\nProcessing window {window + 1}: "
            f"frames {window_start} - "
            f"{window_start + frames_per_window - 1}"
        )

        selected_indices = select_distributed(
            TARGET_FRAMES,
            frames_per_window
        )

        window_frames = []

        for relative_index in selected_indices:

            absolute_index = (
                window_start + relative_index
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                absolute_index
            )

            success, frame = cap.read()

            if not success:

                print(
                    "Could not read frame:",
                    absolute_index
                )

                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                TARGET_SIZE
            )

            window_frames.append(frame)

        all_frames.append(window_frames)

    cap.release()

    return all_frames


if __name__ == "__main__":

    windows = preprocess_video(
        VIDEO_FILE
    )

    print("\n====================================")
    print("TIMELINE PREPROCESSING COMPLETE")
    print("====================================")

    print(
        "Number of windows:",
        len(windows)
    )

    print(
        "Frames per window:",
        [len(w) for w in windows]
    )


    # Save each window separately

    os.makedirs(
        "timeline_faces",
        exist_ok=True
    )

    for window_index, frames in enumerate(windows):

        window_folder = os.path.join(
            "timeline_faces",
            f"window_{window_index + 1}"
        )

        os.makedirs(
            window_folder,
            exist_ok=True
        )

        for frame_index, frame in enumerate(frames):

            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )

            filename = os.path.join(
                window_folder,
                f"frame_{frame_index + 1:02d}.jpg"
            )

            cv2.imwrite(
                filename,
                frame_bgr
            )

    print(
        "\nSaved timeline frames to:"
        " timeline_faces/"
    )