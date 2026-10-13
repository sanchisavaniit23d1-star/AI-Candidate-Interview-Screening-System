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
AUDIO_FILE = "test_audio/test_sample_2_audio.wav"

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

print("\nAUDIO DIAGNOSTIC")
print("Total samples:", len(audio))
print("Audio min:", audio.min())
print("Audio max:", audio.max())
print("Audio mean:", audio.mean())
print("Audio std:", audio.std())

for i in range(6):
    start = int(i * 3.6 * SAMPLE_RATE)
    end = int((i + 1) * 3.6 * SAMPLE_RATE)

    segment = audio[start:end]

    print(
        f"Segment {i+1}: "
        f"mean={segment.mean():.6f}, "
        f"std={segment.std():.6f}, "
        f"min={segment.min():.6f}, "
        f"max={segment.max():.6f}"
    )

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

        frame = frame.astype(np.float32) / 255.0

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

    print(
        f"Window {window_number} "
        f"MFCC mean={mfcc.mean():.4f}, "
        f"std={mfcc.std():.4f}, "
        f"min={mfcc.min():.4f}, "
        f"max={mfcc.max():.4f}"
    )

    audio_tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    with torch.no_grad():

        # --------------------------------------------
        # AUDIO ONLY
        # --------------------------------------------

        audio_output = model(
            audio_tensor,
            torch.zeros_like(visual)
        )

        audio_probabilities = torch.softmax(
            audio_output,
            dim=1
        )[0]


        # --------------------------------------------
        # VISUAL ONLY
        # --------------------------------------------

        visual_output = model(
            torch.zeros_like(audio_tensor),
            visual
        )

        visual_probabilities = torch.softmax(
            visual_output,
            dim=1
        )[0]


        # --------------------------------------------
        # AUDIO + VISUAL
        # --------------------------------------------

        multimodal_output = model(
            audio_tensor,
            visual
        )

        multimodal_probabilities = torch.softmax(
            multimodal_output,
            dim=1
        )[0]


    audio_probabilities = audio_probabilities.numpy()
    visual_probabilities = visual_probabilities.numpy()
    multimodal_probabilities = multimodal_probabilities.numpy()


    audio_emotion = EMOTIONS[
        int(np.argmax(audio_probabilities))
    ]

    visual_emotion = EMOTIONS[
        int(np.argmax(visual_probabilities))
    ]

    multimodal_emotion = EMOTIONS[
        int(np.argmax(multimodal_probabilities))
    ]


    return (
        audio_probabilities,
        audio_emotion,
        visual_probabilities,
        visual_emotion,
        multimodal_probabilities,
        multimodal_emotion
    )


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


    (
        audio_probabilities,
        audio_emotion,
        visual_probabilities,
        visual_emotion,
        multimodal_probabilities,
        multimodal_emotion
    ) = predict_window(window_number)

    probability_dict = {
        EMOTIONS[i]:
        float(multimodal_probabilities[i])
        for i in range(len(EMOTIONS))
    }


    hr_scores = calculate_hr_scores(probability_dict)
    result = {
        "window": window_number,
        "start": round(start_time, 2),
        "end": round(end_time, 2),
  
        "audio_emotion": audio_emotion,
        "visual_emotion": visual_emotion,
        "multimodal_emotion": multimodal_emotion,

        "audio_probabilities": {
            emotion_name: round(float(audio_probabilities[i]), 6)
            for i, emotion_name in enumerate(EMOTIONS)
        },

        "visual_probabilities": {
            emotion_name: round(float(visual_probabilities[i]), 6)
            for i, emotion_name in enumerate(EMOTIONS)
        },

        "multimodal_probabilities": {
            emotion_name: round(float(multimodal_probabilities[i]), 6)
            for i, emotion_name in enumerate(EMOTIONS)
        },

        "hr_scores": hr_scores
    }

    results.append(result)

    print(
        f"\nWindow {window_number}"
        f"  [{start_time:.1f}s - {end_time:.1f}s]"
    )

    print(
        "Audio only      :",
        audio_emotion.upper()
    )

    print(
        "Visual only     :",
        visual_emotion.upper()
    )

    print(
        "Audio + Visual  :",
        multimodal_emotion.upper()
    )

    print("\nTop probabilities:")

    print(
        f"  Audio   → "
        f"{audio_emotion:<10} "
        f"{audio_probabilities[EMOTIONS.index(audio_emotion)] * 100:.2f}%"
    )

    print(
        f"  Visual  → "
        f"{visual_emotion:<10} "
        f"{visual_probabilities[EMOTIONS.index(visual_emotion)] * 100:.2f}%"
    )

    print(
        f"  Fusion  → "
        f"{multimodal_emotion:<10} "
        f"{multimodal_probabilities[EMOTIONS.index(multimodal_emotion)] * 100:.2f}%"
    )
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
        f"{result['multimodal_emotion'].upper()}"
    )


print("\nTimeline prediction complete.")
# ============================================================
# SAVE RESULTS TO JSON
# ============================================================

output_file = "timeline_results.json"

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {output_file}")