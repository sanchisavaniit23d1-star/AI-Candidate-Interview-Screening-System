"""
Interview analyzer (patched version).

Changes vs the previous interview_analyzer.py:
  * MTCNN(post_process=False) and no "* 255": the old code standardized faces to [-1, 1] and then
    clipped them, which blacked out every pixel darker than mid-gray.
  * Optional BGR channel order (env FLIP_CHANNELS=1) to match the original training preprocessing.
    Decide this from evaluate_ravdess_variants.py results.
  * Model and MTCNN are loaded once and reused (needed for an API).
  * torch.no_grad() during inference.
  * Handles videos with no audio track; warns on silent audio and on windows with few detected faces.
  * progress_cb(fraction, message) for the API.
"""
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

CHECKPOINT = "ia_1head_moddrop_2.pth"
EFFICIENTFACE = "EfficientFace_Trained_on_AffectNet7.pth.tar"

SAMPLE_RATE = 22050
WINDOW_SECONDS = 3.6
NUM_FRAMES = 15

#RGB (accuracy drops)
#FLIP_CHANNELS = os.environ.get("FLIP_CHANNELS", "0") == "1"
#BGR (so model was most likely trained on BGR)
FLIP_CHANNELS = os.environ.get("FLIP_CHANNELS", "1") == "1"

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

DISCLAIMER = (
    "Experimental model output. Emotion probabilities come from a model trained on acted speech "
    "(RAVDESS) and are not validated on real interviews. Composure, stress and engagement are "
    "heuristic scores derived from those probabilities, not psychological measurements, and must "
    "not be used for hiring decisions."
)


class CheckpointCompatibleMultiModalCNN(MultiModalCNN):
    """Adds the unused av/va attention modules so the checkpoint loads with strict=True."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.av = Attention(in_dim_k=128, in_dim_q=128, out_dim=128, num_heads=kwargs.get("num_heads", 1))
        self.va = Attention(in_dim_k=128, in_dim_q=128, out_dim=128, num_heads=kwargs.get("num_heads", 1))


def load_model():
    print("Creating model...")
    model = CheckpointCompatibleMultiModalCNN(
        num_classes=8, fusion="ia", seq_length=NUM_FRAMES, pretr_ef=EFFICIENTFACE, num_heads=1
    )
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    clean = {(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(clean, strict=True)
    model.eval()
    print("Trained model loaded successfully!")
    return model


_MODEL = None
_MTCNN = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def get_mtcnn():
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MTCNN = MTCNN(image_size=224, margin=0, keep_all=False, post_process=False, device=device)
    return _MTCNN


def extract_audio(video_path, audio_path):
    os.makedirs(os.path.dirname(audio_path) or ".", exist_ok=True)
    command = ["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), audio_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_path


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError("Could not determine video FPS or length.")
    return {"fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def prepare_faces(video_path, output_dir):
    """Saves 15 face crops per window; writes face_stats.json with real detections per window."""
    os.makedirs(output_dir, exist_ok=True)
    info = get_video_info(video_path)
    fps, duration = info["fps"], info["duration"]
    num_windows = int(np.ceil(duration / WINDOW_SECONDS))
    mtcnn = get_mtcnn()
    cap = cv2.VideoCapture(video_path)
    stats = {}

    for w in range(num_windows):
        start_time = w * WINDOW_SECONDS
        end_time = min((w + 1) * WINDOW_SECONDS, duration)
        window_dir = os.path.join(output_dir, f"window_{w + 1}")
        os.makedirs(window_dir, exist_ok=True)

        first = int(start_time * fps)
        last = max(int(end_time * fps) - 1, first)
        frame_indices = np.linspace(first, last, NUM_FRAMES).astype(int)

        saved = 0
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            face = mtcnn(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if face is None:
                continue
            face = face.permute(1, 2, 0).cpu().numpy()
            if face.max() <= 1.0:  # safety: handle 0..1 output
                face = face * 255
            face = np.clip(face, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(window_dir, f"frame_{saved + 1:02d}.jpg"), cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            saved += 1

        stats[w + 1] = saved
        while saved < NUM_FRAMES:  # black padding, as in the original preprocessing
            cv2.imwrite(os.path.join(window_dir, f"frame_{saved + 1:02d}.jpg"), np.zeros((224, 224, 3), np.uint8))
            saved += 1

    cap.release()
    with open(os.path.join(output_dir, "face_stats.json"), "w") as f:
        json.dump(stats, f)
    return num_windows


def load_visual(window_folder):
    files = sorted(os.path.join(window_folder, f) for f in os.listdir(window_folder) if f.lower().endswith(".jpg"))
    frames = []
    for file in files:
        frame = cv2.imread(file)  # BGR
        if frame is None:
            continue
        if not FLIP_CHANNELS:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1)
        frames.append(frame)
    if len(frames) != NUM_FRAMES:
        raise RuntimeError(f"Expected {NUM_FRAMES} frames but found {len(frames)} in {window_folder}")
    return torch.stack(frames, dim=0)


def predict_window(model, audio, window_index, timeline_dir):
    start = int(window_index * WINDOW_SECONDS * SAMPLE_RATE)
    end = int((window_index + 1) * WINDOW_SECONDS * SAMPLE_RATE)
    segment = audio[start:end]
    if len(segment) == 0:
        return None
    required = int(WINDOW_SECONDS * SAMPLE_RATE)
    if len(segment) < required:
        segment = np.pad(segment, (0, required - len(segment)))

    mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_mfcc=10)
    audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
    visual_tensor = load_visual(os.path.join(timeline_dir, f"window_{window_index + 1}"))

    with torch.no_grad():
        logits = model(audio_tensor, visual_tensor)
        prob = torch.softmax(logits, dim=1)[0].numpy()

    probs = {e: float(prob[i]) for i, e in enumerate(EMOTIONS)}
    return {
        "window": window_index + 1,
        "start_time": round(window_index * WINDOW_SECONDS, 2),
        "end_time": round(min((window_index + 1) * WINDOW_SECONDS, len(audio) / SAMPLE_RATE), 2),
        "fusion": {"emotion": EMOTIONS[int(np.argmax(prob))], "probabilities": probs},
        "hr_scores": calculate_scores(probs),
    }


def analyze_interview(video_path, work_dir=None, progress_cb=None):
    def progress(fraction, message):
        print(f"[{fraction * 100:5.1f}%] {message}")
        if progress_cb:
            progress_cb(fraction, message)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    info = get_video_info(video_path)
    base = os.path.splitext(os.path.basename(video_path))[0]
    work_dir = work_dir or os.path.join("uploads", f"{base}_analysis")
    os.makedirs(work_dir, exist_ok=True)
    warnings = []

    progress(0.02, "Loading model")
    model = get_model()

    progress(0.08, "Extracting audio")
    audio_path = os.path.join(work_dir, "audio.wav")
    try:
        extract_audio(video_path, audio_path)
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception:
        audio = np.zeros(int(info["duration"] * SAMPLE_RATE), dtype=np.float32)
        warnings.append("No usable audio track was found; audio input was replaced by silence, so results rely on the face only and are unreliable.")
    if len(audio) and float(np.std(audio)) < 1e-4 and not warnings:
        warnings.append("The audio track is silent; audio-based evidence is meaningless for this video.")

    progress(0.12, "Detecting faces")
    faces_dir = os.path.join(work_dir, "faces")
    num_windows = prepare_faces(video_path, faces_dir)
    with open(os.path.join(faces_dir, "face_stats.json")) as f:
        face_stats = {int(k): v for k, v in json.load(f).items()}

    timeline = []
    for i in range(num_windows):
        progress(0.3 + 0.68 * i / max(num_windows, 1), f"Analysing window {i + 1}/{num_windows}")
        result = predict_window(model, audio, i, faces_dir)
        if result is None:
            continue
        result["faces_detected"] = face_stats.get(i + 1, 0)
        timeline.append(result)

    low = [t["window"] for t in timeline if t["faces_detected"] < NUM_FRAMES // 2]
    if low:
        warnings.append(f"Fewer than half of the frames had a detected face in windows: {low}. Predictions there are unreliable.")

    result = {
        "video": os.path.basename(video_path),
        "duration": round(info["duration"], 2),
        "window_seconds": WINDOW_SECONDS,
        "settings": {"flip_channels": FLIP_CHANNELS},
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
        "timeline": timeline,
    }
    with open(os.path.join(work_dir, "timeline_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    progress(1.0, "Done")
    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python interview_analyzer.py <video_path>")
        raise SystemExit(1)
    analyze_interview(sys.argv[1])
