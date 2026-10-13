import torch
from models.multimodalcnn import MultiModalCNN
from models.transformer_timm import Attention

CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"


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

        # The checkpoint contains a second pair of Attention
        # modules named av / va.
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


print("Creating checkpoint-compatible model...")

model = CheckpointCompatibleMultiModalCNN(
    num_classes=8,
    seq_length=15,
    pretr_ef=EFFICIENTFACE,
    num_heads=1
)

print("Model created.")

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)

state_dict = checkpoint["state_dict"]

# Remove DataParallel prefix
state_dict = {
    key.replace("module.", "", 1): value
    for key, value in state_dict.items()
}

print("Loading checkpoint...")

result = model.load_state_dict(
    state_dict,
    strict=True
)

print("SUCCESS!")
print(result)