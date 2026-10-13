import torch
from models.multimodalcnn import MultiModalCNN


CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"


print("Creating MultiModalCNN...")

model = MultiModalCNN(
    num_classes=8,
    fusion="ia",
    seq_length=15,
    pretr_ef=EFFICIENTFACE,
    num_heads=1
)

print("Model created successfully!")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

print("Loading trained checkpoint...")

result = model.load_state_dict(
    checkpoint["state_dict"],
    strict=True
)

print("Checkpoint loaded successfully!")
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)

model.eval()

print("Model is ready for inference.")