import cv2
from facenet_pytorch import MTCNN

VIDEO_FILE = "test_sample_1.mp4"

mtcnn = MTCNN(
    image_size=224,
    margin=0,
    keep_all=False,
    post_process=False,
    device="cpu"
)

cap = cv2.VideoCapture(VIDEO_FILE)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Total frames:", total_frames)

test_positions = [
    0,
    total_frames // 4,
    total_frames // 2,
    (total_frames * 3) // 4,
    total_frames - 1
]

for position in test_positions:

    cap.set(cv2.CAP_PROP_POS_FRAMES, position)

    success, frame = cap.read()

    if not success:
        print("Could not read frame:", position)
        continue

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    face = mtcnn(rgb)

    if face is None:

        print(
            f"Frame {position}: NO FACE DETECTED"
        )

    else:

        print(
            f"Frame {position}: FACE DETECTED",
            "shape =", tuple(face.shape)
        )

cap.release()