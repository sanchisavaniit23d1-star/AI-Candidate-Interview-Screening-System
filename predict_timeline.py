import os
import cv2
import torch
import librosa
import numpy as np

from models.multimodalcnn import MultiModalCNN
from models.transformer_timm import Attention


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"

FACE_DIR = "test_faces"
AUDIO_FILE = "test_audio/extracted_audio.wav"

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

WINDOW_SECONDS = 3.6
SAMPLE_RATE = 22050
NUM_FRAMES = 15


# ============================================================
# CHECKPOINT-COMPATIBLE MODEL
# ============================================================

class CheckpointCompatibleMultiModalCNN(MultiModalCNN):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # These layers exist in the checkpoint.
        self.av = Attention(
            in_dim_k=128,
            in_dim_q=128,
            out_dim=128,
            num_heads=kwargs.get("num_heads", 1)
        )

        self.va = Attention(
            in_dim_k=128,
            in_dim_q=128,
            out_dim=128,
            num_heads=kwargs.get("num_heads", 1)
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("Creating model...")

model = CheckpointCompatibleMultiModalCNN(
    num_classes=8,
    fusion="ia",
    seq_length=NUM_FRAMES,
    pretr_ef=EFFICIENTFACE,
    num_heads=1
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

state_dict = checkpoint["state_dict"]

# Remove DataParallel "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()

print("Trained model loaded successfully!")


# ============================================================
# LOAD ALL FACE FRAMES
# ============================================================

print("\nLoading face frames...")

face_files = [
    os.path.join(FACE_DIR, f)
    for f in os.listdir(FACE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

face_files.sort()

print("Total face frames:", len(face_files))


def load_frames(files):

    frames = []

    for file in files:

        frame = cv2.imread(file)

        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0

        frame = (frame * 2.0) - 1.0

        frame = torch.from_numpy(frame)

        frame = frame.permute(2, 0, 1)

        frames.append(frame)

    return torch.stack(frames, dim=0)


# ============================================================
# LOAD AUDIO
# ============================================================

print("Loading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE
)

print("Audio length:", round(len(audio) / sr, 2), "seconds")


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(face_files_window, audio_window):

    # ---------------------------
    # Visual
    # ---------------------------

    visual = load_frames(face_files_window)

    # Ensure exactly 15 frames
    if visual.shape[0] < NUM_FRAMES:

        missing = NUM_FRAMES - visual.shape[0]

        padding = torch.zeros(
            missing,
            3,
            224,
            224
        )

        visual = torch.cat(
            [visual, padding],
            dim=0
        )

    elif visual.shape[0] > NUM_FRAMES:

        visual = visual[:NUM_FRAMES]


    # ---------------------------
    # Audio
    # ---------------------------

    required_samples = int(
        WINDOW_SECONDS * SAMPLE_RATE
    )

    if len(audio_window) < required_samples:

        audio_window = np.pad(
            audio_window,
            (0, required_samples - len(audio_window))
        )

    else:

        audio_window = audio_window[:required_samples]


    mfcc = librosa.feature.mfcc(
        y=audio_window,
        sr=SAMPLE_RATE,
        n_mfcc=10
    )

    audio_tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)


    # ---------------------------
    # Model
    # ---------------------------

    with torch.no_grad():

        output = model(
            audio_tensor,
            visual
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    probabilities = probabilities.numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_emotion = EMOTIONS[predicted_index]


    return probabilities, predicted_emotion


# ============================================================
# CREATE WINDOWS
# ============================================================

audio_duration = len(audio) / SAMPLE_RATE

num_windows = int(
    np.ceil(audio_duration / WINDOW_SECONDS)
)

print("\n=============================================")
print("TIMELINE ANALYSIS")
print("=============================================")

results = []


for window_index in range(num_windows):

    start_time = window_index * WINDOW_SECONDS
    end_time = min(
        (window_index + 1) * WINDOW_SECONDS,
        audio_duration
    )


    # --------------------------------------------------------
    # Audio section
    # --------------------------------------------------------

    start_sample = int(
        start_time * SAMPLE_RATE
    )

    end_sample = int(
        end_time * SAMPLE_RATE
    )

    audio_window = audio[
        start_sample:end_sample
    ]


    # --------------------------------------------------------
    # Face section
    #
    # 15 frames represent one 3.6-second window.
    # --------------------------------------------------------

    start_frame = window_index * NUM_FRAMES

    end_frame = start_frame + NUM_FRAMES

    face_window = face_files[
        start_frame:end_frame
    ]


    # If there aren't enough faces,
    # we'll pad inside predict_window().
    if len(face_window) == 0:
        break


    probabilities, emotion = predict_window(
        face_window,
        audio_window
    )


    probability_dict = {
        EMOTIONS[i]: float(probabilities[i])
        for i in range(len(EMOTIONS))
    }


    result = {
        "window": window_index + 1,
        "start": round(start_time, 2),
        "end": round(end_time, 2),
        "emotion": emotion,
        "probabilities": probability_dict
    }

    results.append(result)


    print(
        f"\nWindow {window_index + 1}"
        f"  [{start_time:.1f}s - {end_time:.1f}s]"
    )

    print(
        "Prediction:",
        emotion.upper()
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n=============================================")
print("FINAL TIMELINE")
print("=============================================")

for result in results:

    print(
        f"{result['start']:>5.1f}s - "
        f"{result['end']:>5.1f}s : "
        f"{result['emotion'].upper()}"
    )

print("\nTimeline analysis complete.")