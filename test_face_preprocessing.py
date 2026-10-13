import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import torch


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


# Use GPU if available, otherwise CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# MTCNN face detector
mtcnn = MTCNN(
    keep_all=False,
    device=device
)


def preprocess_video(video_file):

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Total frames:", total_frames)
    print("FPS:", fps)

    required_frames = int(FPS * DURATION)

    frames_to_use = min(
        required_frames,
        total_frames
    )

    selected_indices = select_distributed(
        TARGET_FRAMES,
        frames_to_use
    )

    faces = []

    for index in selected_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            index
        )

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", index)
            continue

        # OpenCV BGR → RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to PIL image for MTCNN
        image = Image.fromarray(frame)

        # Detect and crop face
        face = mtcnn(image)

        if face is None:
            print(
                f"No face detected in frame {index}"
            )
            continue

        # MTCNN returns a tensor
        # Convert to numpy for saving
        face = face.permute(
            1, 2, 0
        ).cpu().numpy()

        # Convert from [-1, 1] to [0, 255]
        face = (
            (face + 1) * 127.5
        ).clip(0, 255).astype(np.uint8)

        face = cv2.resize(
            face,
            TARGET_SIZE
        )

        faces.append(face)

    cap.release()

    return np.array(faces)


if __name__ == "__main__":

    faces = preprocess_video(
        VIDEO_FILE
    )

    print("\nFace preprocessing complete!")

    print(
        "Detected face frames:",
        len(faces)
    )

    print(
        "Output shape:",
        faces.shape
    )

    os.makedirs(
        "test_faces",
        exist_ok=True
    )

    for i, face in enumerate(faces):

        filename = (
            f"test_faces/face_{i + 1:02d}.jpg"
        )

        cv2.imwrite(
            filename,
            cv2.cvtColor(
                face,
                cv2.COLOR_RGB2BGR
            )
        )

    print(
        "Saved detected faces to: test_faces/"
    )