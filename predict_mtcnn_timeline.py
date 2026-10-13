import os
import torch
import librosa
import numpy as np
import json

from models.multimodalcnn import MultiModalCNN
from models.transformer_timm import Attention


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"

TIMELINE_DIR = "timeline_faces_mtcnn"
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

def calculate_hr_scores(probabilities):
    positive = (
        probabilities["calm"]
        + probabilities["happy"]
        + probabilities["neutral"]
    )

    negative = (
        probabilities["angry"]
        + probabilities["fearful"]
        + probabilities["sad"]
        + probabilities["disgust"]
    )

    engagement = (
        probabilities["happy"]
        + probabilities["surprised"]
        + probabilities["neutral"]
    )

    return {
        "composure": round(positive * 100, 2),
        "stress": round(negative * 100, 2),
        "engagement": round(engagement * 100, 2)
    }

SAMPLE_RATE = 22050
WINDOW_SECONDS = 3.6
NUM_FRAMES = 15


# ============================================================
# CHECKPOINT-COMPATIBLE MODEL
# ============================================================

class CheckpointCompatibleMultiModalCNN(MultiModalCNN):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

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
# LOAD AUDIO
# ============================================================

print("\nLoading audio...")

audio, sr = librosa.load(
    AUDIO_FILE,
    sr=SAMPLE_RATE
)

audio_duration = len(audio) / SAMPLE_RATE

print(
    "Audio duration:",
    round(audio_duration, 2),
    "seconds"
)


# ============================================================
# LOAD ONE WINDOW OF FACE FRAMES
# ============================================================

def load_visual(window_folder):

    frame_files = [
        os.path.join(window_folder, file)
        for file in os.listdir(window_folder)
        if file.lower().endswith(".jpg")
    ]

    frame_files.sort()

    frames = []

    for file in frame_files:

        # Same preprocessing used by our
        # working real-prediction baseline.

        import cv2

        frame = cv2.imread(file)

        if frame is None:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = frame.astype(
            np.float32
        ) / 255.0

        frame = (frame * 2.0) - 1.0

        frame = torch.from_numpy(frame)

        frame = frame.permute(
            2, 0, 1
        )

        frames.append(frame)


    visual = torch.stack(
        frames,
        dim=0
    )


    return visual


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(window_number):

    # --------------------------------------------------------
    # Visual
    # --------------------------------------------------------

    window_folder = os.path.join(
        TIMELINE_DIR,
        f"window_{window_number}"
    )

    visual = load_visual(
        window_folder
    )


    # --------------------------------------------------------
    # Audio
    # --------------------------------------------------------

    start_time = (
        (window_number - 1)
        * WINDOW_SECONDS
    )

    end_time = (
        start_time
        + WINDOW_SECONDS
    )


    start_sample = int(
        start_time * SAMPLE_RATE
    )

    end_sample = int(
        end_time * SAMPLE_RATE
    )

    audio_window = audio[
        start_sample:end_sample
    ]


    required_samples = int(
        WINDOW_SECONDS * SAMPLE_RATE
    )


    if len(audio_window) < required_samples:

        audio_window = np.pad(
            audio_window,
            (
                0,
                required_samples
                - len(audio_window)
            )
        )


    mfcc = librosa.feature.mfcc(
        y=audio_window,
        sr=SAMPLE_RATE,
        n_mfcc=10
    )


    audio_tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    predicted_emotion = EMOTIONS[
        predicted_index
    ]


    return probabilities, predicted_emotion


# ============================================================
# RUN ALL WINDOWS
# ============================================================

print("\n=============================================")
print("REAL MULTIMODAL EMOTION TIMELINE")
print("=============================================")


results = []

for window_number in range(1, 7):

    start_time = (
        (window_number - 1)
        * WINDOW_SECONDS
    )

    end_time = (
        window_number
        * WINDOW_SECONDS
    )


    probabilities, emotion = predict_window(
        window_number
    )


    probability_dict = {
        EMOTIONS[i]:
        float(probabilities[i])
        for i in range(len(EMOTIONS))
    }


    hr_scores = calculate_hr_scores(probability_dict)
    result = {
        "window": window_number,
        "start": round(start_time, 2),
        "end": round(end_time, 2),
        "emotion": emotion,
        "probabilities": {
            emotion_name: round(probability, 6)
            for emotion_name, probability in probability_dict.items()
        },
        "hr_scores": hr_scores
    }

    results.append(result)

    print(
        f"\nWindow {window_number}"
        f"  [{start_time:.1f}s - {end_time:.1f}s]"
    )

    print(
        "Prediction:",
        emotion.upper()
    )

    print("Probabilities:")

    for emotion_name, probability in probability_dict.items():

        print(
            f"  {emotion_name:<10}: "
            f"{probability * 100:6.2f}%"
        )
        
    print("HR Scores:")
    print(f"  Composure : {hr_scores['composure']:.2f}")
    print(f"  Stress    : {hr_scores['stress']:.2f}")
    print(f"  Engagement: {hr_scores['engagement']:.2f}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n=============================================")
print("FINAL EMOTION TIMELINE")
print("=============================================")


for result in results:

    print(
        f"{result['start']:>5.1f}s - "
        f"{result['end']:>5.1f}s : "
        f"{result['emotion'].upper()}"
    )


print("\nTimeline prediction complete.")
# ============================================================
# SAVE RESULTS TO JSON
# ============================================================

output_file = "timeline_results.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")