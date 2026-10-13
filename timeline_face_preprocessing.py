import os
import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN


# ============================================================
# CONFIG
# ============================================================

VIDEO_FILE = "test_sample_1.mp4"

OUTPUT_DIR = "timeline_faces_mtcnn"

FPS = 30
WINDOW_SECONDS = 3.6
FRAMES_PER_WINDOW = 15
FRAMES_IN_WINDOW = int(FPS * WINDOW_SECONDS)

IMAGE_SIZE = 224


# ============================================================
# CREATE FACE DETECTOR
# ============================================================

print("Loading MTCNN face detector...")

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    keep_all=False,
    post_process=False,
    device="cpu"
)

print("MTCNN loaded.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {VIDEO_FILE}"
    )

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

video_fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("\nVideo information:")
print("Total frames:", total_frames)
print("FPS:", video_fps)


# ============================================================
# DETERMINE WINDOWS
# ============================================================

num_windows = total_frames // FRAMES_IN_WINDOW

print("Frames per window:", FRAMES_IN_WINDOW)
print("Number of complete windows:", num_windows)


# ============================================================
# PROCESS EACH WINDOW
# ============================================================

for window_number in range(1, num_windows + 1):

    print(
        f"\nProcessing window {window_number}..."
    )

    window_dir = os.path.join(
        OUTPUT_DIR,
        f"window_{window_number}"
    )

    os.makedirs(
        window_dir,
        exist_ok=True
    )

    window_start = (
        (window_number - 1)
        * FRAMES_IN_WINDOW
    )

    frame_positions = np.linspace(
        window_start,
        window_start + FRAMES_IN_WINDOW - 1,
        FRAMES_PER_WINDOW,
        dtype=int
    )

    saved_count = 0

    for frame_position in frame_positions:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_position)
        )

        success, frame = cap.read()

        if not success:
            continue

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Detect and crop face
        # ----------------------------------------------------

        face = mtcnn(
            Image.fromarray(rgb)
        )

        if face is None:

            print(
                f"  Frame {frame_position}: "
                f"NO FACE"
            )

            continue

        # ----------------------------------------------------
        # MTCNN returns tensor:
        # [3, 224, 224]
        # ----------------------------------------------------

        face = face.permute(
            1, 2, 0
        ).numpy()

        face = np.clip(
            face,
            0,
            255
        ).astype(
            np.uint8
        )

        output_file = os.path.join(
            window_dir,
            f"frame_{saved_count + 1:02d}.jpg"
        )

        cv2.imwrite(
            output_file,
            cv2.cvtColor(
                face,
                cv2.COLOR_RGB2BGR
            )
        )

        saved_count += 1

    print(
        f"  Saved {saved_count}/"
        f"{FRAMES_PER_WINDOW} face frames"
    )


cap.release()


print("\n=============================================")
print("MTCNN FACE PREPROCESSING COMPLETE")
print("=============================================")
print(
    "Output directory:",
    OUTPUT_DIR
)