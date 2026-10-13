import torch
from models.multimodalcnn import MultiModalCNN

print("Creating model...")

model = MultiModalCNN(
    num_classes=8,
    fusion="ia",
    seq_length=15,
    pretr_ef=None,
    num_heads=1
)

model.eval()

print("Model created.")
print("Creating dummy input...")

# 1 sample
# 15 video frames
# Each frame = 3 channels, 224 x 224
video = torch.randn(15, 3, 224, 224)

# Audio: 10 MFCC coefficients
# The exact time dimension can be tested with a reasonable MFCC length.
audio = torch.randn(1, 10, 173)

print("Video shape:", video.shape)
print("Audio shape:", audio.shape)

print("\nRunning forward pass...")

with torch.no_grad():
    output = model(audio, video)

print("\nForward pass successful!")

print("Output shape:", output.shape)
print("Raw output:", output)

probabilities = torch.softmax(output, dim=1)

print("\nEmotion probabilities:")

emotions = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

for emotion, probability in zip(emotions, probabilities[0]):
    print(f"{emotion:10s}: {probability.item() * 100:.2f}%")