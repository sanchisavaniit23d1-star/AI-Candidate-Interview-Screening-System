import os
import cv2
import torch
import librosa
import numpy as np

from models.multimodalcnn import MultiModalCNN
from models.transformer_timm import Attention

# --------------------------------------------------
# Configuration
# --------------------------------------------------

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


# --------------------------------------------------
# Load model
# --------------------------------------------------
class CheckpointCompatibleMultiModalCNN(MultiModalCNN):

    def __init__(
        self,
        num_classes=8,
        seq_length=15,
        pretr_ef="None",
        num_heads=1
    ):
        super().__init__(
            num_classes=num_classes,
            fusion="ia",
            seq_length=seq_length,
            pretr_ef=pretr_ef,
            num_heads=num_heads
        )

        self.av = Attention(
            in_dim_k=128,
            in_dim_q=128,
            out_dim=128,
            num_heads=num_heads
        )

        self.va = Attention(
            in_dim_k=128,
            in_dim_q=128,
            out_dim=128,
            num_heads=num_heads
        )
print("Creating model...")

model = CheckpointCompatibleMultiModalCNN(
    num_classes=8,
#   fusion="ia",
    seq_length=15,
    pretr_ef=EFFICIENTFACE,
    num_heads=1
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

state_dict = checkpoint["state_dict"]

# Remove DataParallel's "module." prefix
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()

print("Trained model loaded successfully!")


# --------------------------------------------------
# Load face frames
# --------------------------------------------------

print("\nLoading face frames...")

faces = []

for i in range(1, 16):

    filename = os.path.join(
        FACE_DIR,
        f"face_{i:02d}.jpg"
    )

    image = cv2.imread(filename)

    if image is None:
        raise RuntimeError(
            f"Could not read {filename}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    # Convert [0,1] → [-1,1]
    image = (image * 2.0) - 1.0

    # H,W,C → C,H,W
    image = torch.from_numpy(
        image
    ).permute(2, 0, 1)

    faces.append(image)

visual = torch.stack(faces)

print("Visual tensor shape:", visual.shape)


# --------------------------------------------------
# Load audio
# --------------------------------------------------

print("\nLoading audio...")

# Load audio at the official sampling rate
y, sr = librosa.load(AUDIO_FILE, sr=22050)

# Official preprocessing uses a 3.6-second audio window
target_length = int(3.6 * sr)

# Crop audio if longer than 3.6 seconds
if len(y) > target_length:
    y = y[:target_length]

# Pad with zeros if shorter than 3.6 seconds
elif len(y) < target_length:
    padding = target_length - len(y)
    y = np.pad(y, (0, padding), mode="constant")

# Extract 10 MFCC coefficients
mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=10
)

# Convert to PyTorch tensor
audio = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

print("Audio tensor shape:", audio.shape)

# Add batch dimension
#audio = audio.unsqueeze(0)

print("Audio tensor shape:", audio.shape)


# --------------------------------------------------
# Prepare visual input
# --------------------------------------------------

# Model receives:
# batch × frames × channels × height × width
#
# Validation code in the original repository
# then flattens batch and frames.

visual = visual
print("Model visual input shape:", visual.shape)


# --------------------------------------------------
# Run inference
# --------------------------------------------------

print("\nRunning multimodal inference...")

with torch.no_grad():

    # -------------------------------
    # Audio only
    # -------------------------------
    audio_only_output = model(
        audio,
        torch.zeros_like(visual)
    )

    # -------------------------------
    # Visual only
    # -------------------------------
    visual_only_output = model(
        torch.zeros_like(audio),
        visual
    )

    # -------------------------------
    # Audio + Visual
    # -------------------------------
    multimodal_output = model(
        audio,
        visual
    )

audio_probabilities = torch.softmax(
    audio_only_output,
    dim=1
)[0]

visual_probabilities = torch.softmax(
    visual_only_output,
    dim=1
)[0]

multimodal_probabilities = torch.softmax(
    multimodal_output,
    dim=1
)[0]

# --------------------------------------------------
# Display results
# --------------------------------------------------

def display_prediction(title, probabilities):

    print("\n" + "=" * 45)
    print(title)
    print("=" * 45)

    for emotion, probability in zip(
        EMOTIONS,
        probabilities
    ):
        print(
            f"{emotion:<10}: "
            f"{probability.item() * 100:.2f}%"
        )

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_emotion = EMOTIONS[
        predicted_index
    ]

    print(
        f"\nPredicted emotion: "
        f"{predicted_emotion.upper()}"
    )


# -----------------------------------------
# Display all three experiments
# -----------------------------------------

display_prediction(
    "AUDIO ONLY",
    audio_probabilities
)

display_prediction(
    "VISUAL ONLY",
    visual_probabilities
)

display_prediction(
    "AUDIO + VISUAL",
    multimodal_probabilities
)