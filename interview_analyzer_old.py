import os
import json
import subprocess
import cv2
import torch
import librosa
import numpy as np

from models.multimodalcnn import MultiModalCNN
from models.transformer_timm import Attention
from hr_scoring import calculate_scores


# ============================================================
# CONFIG
# ============================================================

CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"

SAMPLE_RATE = 22050
WINDOW_SECONDS = 3.6
NUM_FRAMES = 15

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

def load_model():

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

    return model


# ============================================================
# EXTRACT AUDIO FROM VIDEO
# ============================================================

def extract_audio(video_path, audio_path):

    os.makedirs(
        os.path.dirname(audio_path) or ".",
        exist_ok=True
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        audio_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return audio_path


# ============================================================
# GET VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    if fps <= 0:
        raise RuntimeError(
            "Could not determine video FPS."
        )

    duration = frame_count / fps

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration
    }


# ============================================================
# EXTRACT FACE FRAMES
# ============================================================

def prepare_faces(video_path, output_dir):

    from facenet_pytorch import MTCNN

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    info = get_video_info(video_path)

    fps = info["fps"]
    duration = info["duration"]

    num_windows = int(
        np.ceil(
            duration / WINDOW_SECONDS
        )
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

#mtcnn = MTCNN(image_size=224, margin=0, keep_all=Flase, device=device)

    mtcnn = MTCNN(
        image_size=224,
        margin=0,
        keep_all=False,
        post_process=False,
        device=device
    )

    cap = cv2.VideoCapture(video_path)

    for window_index in range(num_windows):

        start_time = (
            window_index * WINDOW_SECONDS
        )

        end_time = min(
            (window_index + 1) * WINDOW_SECONDS,
            duration
        )

        window_dir = os.path.join(
            output_dir,
            f"window_{window_index + 1}"
        )

        os.makedirs(
            window_dir,
            exist_ok=True
        )

        frame_indices = np.linspace(
            int(start_time * fps),
            max(
                int(end_time * fps) - 1,
                int(start_time * fps)
            ),
            NUM_FRAMES
        ).astype(int)

        saved = 0

        for frame_index in frame_indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_index)
            )

            success, frame = cap.read()

            if not success:
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            face = mtcnn(rgb)

            if face is None:
                continue

            face = face.permute(
                1, 2, 0
            ).cpu().numpy()

#face = np.clip(face*225, 0, 255).astype(np.uint8)   # no "* 255"
            face = np.clip(face, 0, 255).astype(np.uint8)

            face_path = os.path.join(
                window_dir,
                f"frame_{saved + 1:02d}.jpg"
            )

            cv2.imwrite(
                face_path,
                cv2.cvtColor(
                    face,
                    cv2.COLOR_RGB2BGR
                )
            )

            saved += 1

        # Pad missing frames with black images.
        while saved < NUM_FRAMES:

            black = np.zeros(
                (224, 224, 3),
                dtype=np.uint8
            )

            face_path = os.path.join(
                window_dir,
                f"frame_{saved + 1:02d}.jpg"
            )

            cv2.imwrite(
                face_path,
                black
            )

            saved += 1

    cap.release()

    return num_windows


# ============================================================
# LOAD VISUAL WINDOW
# ============================================================

def load_visual(window_folder):

    frame_files = sorted(
        [
            os.path.join(
                window_folder,
                file
            )
            for file in os.listdir(window_folder)
            if file.lower().endswith(".jpg")
        ]
    )

    frames = []

    for file in frame_files:

        frame = cv2.imread(file)

        if frame is None:
            continue

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame = (
            frame.astype(np.float32)
            / 255.0
        )

        frame = torch.from_numpy(
            frame
        ).permute(
            2, 0, 1
        )

        frames.append(frame)

    if len(frames) != NUM_FRAMES:

        raise RuntimeError(
            f"Expected {NUM_FRAMES} frames "
            f"but found {len(frames)} in "
            f"{window_folder}"
        )

    visual = torch.stack(
        frames,
        dim=0
    )

    return visual


# ============================================================
# PREDICT ONE WINDOW
# ============================================================

def predict_window(
    model,
    audio,
    window_index,
    timeline_dir
):

    start = int(
        window_index
        * WINDOW_SECONDS
        * SAMPLE_RATE
    )

    end = int(
        (window_index + 1)
        * WINDOW_SECONDS
        * SAMPLE_RATE
    )

    audio_segment = audio[start:end]

    if len(audio_segment) == 0:

        return None

    # Pad final audio window if necessary.
    required_samples = int(
        WINDOW_SECONDS
        * SAMPLE_RATE
    )

    if len(audio_segment) < required_samples:

        audio_segment = np.pad(
            audio_segment,
            (
                0,
                required_samples
                - len(audio_segment)
            )
        )

    mfcc = librosa.feature.mfcc(
        y=audio_segment,
        sr=SAMPLE_RATE,
        n_mfcc=10
    )

    audio_tensor = torch.tensor(
        mfcc,
        dtype=torch.float32
    ).unsqueeze(0)

    visual_tensor = load_visual(
        os.path.join(
            timeline_dir,
            f"window_{window_index + 1}"
        )
    )

    output_fusion = model(
        audio_tensor,
        visual_tensor
    )

    fusion_prob = torch.softmax(
        output_fusion,
        dim=1
    )[0].detach().numpy()
    
    def convert(probabilities):
        return {
            emotion: float(probabilities[index])
            for index, emotion in enumerate(EMOTIONS)
        }

    fusion_probs = convert(fusion_prob)

    fusion_scores = calculate_scores(
        fusion_probs
    )

    fusion_emotion = EMOTIONS[
        int(np.argmax(fusion_prob))
    ]

    return {
        "window": window_index + 1,

        "start_time": round(
            window_index * WINDOW_SECONDS,
            2
        ),

        "end_time": round(
            min(
                (window_index + 1) * WINDOW_SECONDS,
                len(audio) / SAMPLE_RATE
            ),
            2
        ),

        "fusion": {
            "emotion": fusion_emotion,
            "probabilities": fusion_probs
        },

        "hr_scores": fusion_scores
    }


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_interview(video_path):

    if not os.path.exists(video_path):

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print("\n======================================")
    print("AI INTERVIEW ANALYZER")
    print("======================================")

    info = get_video_info(
        video_path
    )

    print(
        f"Video duration: "
        f"{info['duration']:.2f} seconds"
    )

    base_name = os.path.splitext(
        os.path.basename(video_path)
    )[0]

    work_dir = os.path.join(
        "uploads",
        f"{base_name}_analysis"
    )

    audio_path = os.path.join(
        work_dir,
        "audio.wav"
    )

    faces_dir = os.path.join(
        work_dir,
        "faces"
    )

    os.makedirs(
        work_dir,
        exist_ok=True
    )

    print("\n[1/4] Loading model...")

    model = load_model()

    print("\n[2/4] Extracting audio...")

    extract_audio(
        video_path,
        audio_path
    )

    audio, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    print("\n[3/4] Extracting faces...")

    num_windows = prepare_faces(
        video_path,
        faces_dir
    )

    print(
        f"Created {num_windows} windows."
    )

    print("\n[4/4] Running inference...")

    timeline = []

    for window_index in range(
        num_windows
    ):

        print(
            f"Processing window "
            f"{window_index + 1}/{num_windows}..."
        )

        result = predict_window(
            model,
            audio,
            window_index,
            faces_dir
        )

        if result is not None:

            timeline.append(
                result
            )

    result = {
        "video": os.path.basename(
            video_path
        ),
        "duration": round(
            info["duration"],
            2
        ),
        "window_seconds": WINDOW_SECONDS,
        "timeline": timeline
    }

    output_path = os.path.join(
        work_dir,
        "timeline_results.json"
    )

    with open(
        output_path,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )

    print("\n======================================")
    print("ANALYSIS COMPLETE")
    print("======================================")

    print(
        f"Windows processed: "
        f"{len(timeline)}"
    )

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    return result


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python interview_analyzer.py "
            "<video_path>"
        )

        raise SystemExit(1)

    analyze_interview(
        sys.argv[1]
    )