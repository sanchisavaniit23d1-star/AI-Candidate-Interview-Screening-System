import cv2
import os
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

VIDEO_FILE = "test_sample_2.mp4"
OUTPUT_DIR = "timeline_faces_mtcnn"

FPS = 30
WINDOW_SECONDS = 3.6
NUM_FRAMES = 15
NUM_WINDOWS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)

mtcnn = MTCNN(
    image_size=224,
    margin=0,
    keep_all=False,
    post_process=False,
    device="cpu"
)

cap = cv2.VideoCapture(VIDEO_FILE)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_FILE}")

actual_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video FPS: {actual_fps}")
print(f"Total frames: {total_frames}")
print(f"Processing first {NUM_WINDOWS * WINDOW_SECONDS:.1f} seconds...")

for window in range(NUM_WINDOWS):

    window_dir = os.path.join(
        OUTPUT_DIR,
        f"window_{window + 1}"
    )

    os.makedirs(window_dir, exist_ok=True)

    start_frame = int(window * WINDOW_SECONDS * actual_fps)
    end_frame = int((window + 1) * WINDOW_SECONDS * actual_fps)

    frame_indices = np.linspace(
        start_frame,
        end_frame - 1,
        NUM_FRAMES,
        dtype=int
    )

    print(f"\nWindow {window + 1}")
    print(f"Frames: {start_frame} → {end_frame - 1}")

    saved = 0

    for i, frame_idx in enumerate(frame_indices):

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()

        if not ret:
            print(f"  Could not read frame {frame_idx}")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        face = mtcnn(image)

        if face is None:
            print(f"  Frame {i + 1}: no face detected")
            continue

        face = face.numpy()

        # MTCNN returns C,H,W → convert to H,W,C
        face = np.transpose(face, (1, 2, 0))

        # Convert float image to uint8
        face = np.clip(face, 0, 255).astype(np.uint8)

        output_path = os.path.join(
            window_dir,
            f"frame_{i + 1:02d}.jpg"
        )

        Image.fromarray(face).save(output_path)

        saved += 1

    # Pad missing detections with black frames
    while saved < NUM_FRAMES:

        output_path = os.path.join(
            window_dir,
            f"frame_{saved + 1:02d}.jpg"
        )

        black = np.zeros((224, 224, 3), dtype=np.uint8)
        Image.fromarray(black).save(output_path)

        saved += 1

    print(f"  Saved {saved}/15 face frames")

cap.release()

print("\n===================================")
print("Sample 2 face preprocessing complete!")
print("===================================")
print(f"Output: {OUTPUT_DIR}")