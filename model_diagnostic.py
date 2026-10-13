import torch

from models.multimodalcnn import MultiModalCNN


print("Creating MultiModalCNN...")

model = MultiModalCNN(
    num_classes=8,
    fusion="ia",
    seq_length=15,
    pretr_ef="EfficientFace_Trained_on_AffectNet7.pth.tar",
    num_heads=1
)

print("\nModel created successfully!")
print("\nModel input configuration:")
print("Audio input: 10 MFCC features")
print("Video frames: 15")
print("Video frame size: 224 x 224")
print("Video channels: 3")
print("Number of emotions: 8")

print("\nExpected tensor shapes:")

print("Audio:")
print("  [batch, 10, time]")

print("Video before model:")
print("  [batch, 15, 3, 224, 224]")

print("\nModel:")
print(model)