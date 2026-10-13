"""
Pilot evaluation of preprocessing variants on RAVDESS (UNTESTED here: run it on your machine).

Mimics the base project's training-time preprocessing:
  * video: 15 frames at indices i*N//15 + N//30 over the FIRST 3.6 s (N = 3.6*fps), MTCNN box crop, resize 224
  * audio: whole clip centre-cropped / zero-padded to 3.6 s, then 10 MFCC at 22,050 Hz
It runs each clip with two channel orders (RGB vs BGR-as-in-training) and three modes (fused, audio probe, visual probe).

Usage (from the project folder, venv active):
    python evaluate_ravdess_variants.py "E:\\AI-Interview-Project\\RAVDESS" 01 02 03 04
"""
import os, sys, json, csv, subprocess
import cv2, numpy as np, torch, librosa
from facenet_pytorch import MTCNN
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from interview_analyzer import load_model, EMOTIONS, SAMPLE_RATE, WINDOW_SECONDS, NUM_FRAMES

CODES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
         "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
OUT = os.path.join("uploads", "variant_eval")


def audio_training_style(video_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), wav_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    target = int(SAMPLE_RATE * WINDOW_SECONDS)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        remain = len(y) - target
        y = y[remain // 2: len(y) - (remain - remain // 2)]
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=10)
    return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)


def faces_training_style(video_path, mtcnn):
    """Returns (bgr_tensor, rgb_tensor), each [15,3,224,224] in 0..1."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(round(WINDOW_SECONDS * fps))
    wanted = [i * n // NUM_FRAMES + n // (2 * NUM_FRAMES) for i in range(NUM_FRAMES)]
    bgr_list, rgb_list = [], []
    last_box = None
    idx = 0
    while idx <= max(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            boxes, _ = mtcnn.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if boxes is not None and len(boxes) > 0:
                last_box = boxes[0]
            if last_box is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = [int(round(v)) for v in last_box]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                crop = cv2.resize(frame[y1:y2, x1:x2], (224, 224))
            else:
                crop = np.zeros((224, 224, 3), np.uint8)
            bgr_list.append(crop)
            rgb_list.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    while len(bgr_list) < NUM_FRAMES:  # black padding, as in the original script
        z = np.zeros((224, 224, 3), np.uint8)
        bgr_list.append(z); rgb_list.append(z)
    to_t = lambda lst: torch.stack([torch.from_numpy(a.astype(np.float32) / 255.0).permute(2, 0, 1) for a in lst])
    return to_t(bgr_list), to_t(rgb_list)


def main():
    root, actors = sys.argv[1], sys.argv[2:] or ["01"]
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=False, device=device)  # only .detect() is used, so post_process is irrelevant
    model = load_model()
    rows = []
    for actor in actors:
        adir = os.path.join(root, f"Actor_{actor}")
        clips = sorted(f for f in os.listdir(adir) if f.endswith(".mp4") and f.split("-")[0] == "01")
        for k, name in enumerate(clips, 1):
            code = name[:-4].split("-")[2]
            path = os.path.join(adir, name)
            print(f"Actor {actor} [{k}/{len(clips)}] {name}")
            try:
                a = audio_training_style(path, os.path.join(OUT, "tmp", name[:-4] + ".wav"))
                bgr, rgb = faces_training_style(path, mtcnn)
                row = {"actor": actor, "file": name, "actual": CODES[code]}
                with torch.no_grad():
                    for vname, v in (("bgr", bgr), ("rgb", rgb)):
                        for mode, (aa, vv) in {"fused": (a, v), "audio": (a, torch.zeros_like(v)),
                                               "visual": (torch.zeros_like(a), v)}.items():
                            p = torch.softmax(model(aa, vv), dim=1)[0].numpy()
                            row[f"{vname}_{mode}"] = EMOTIONS[int(np.argmax(p))]
                rows.append(row)
            except Exception as e:
                print("  FAILED:", e)
    with open(os.path.join(OUT, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for col in [c for c in rows[0] if c.startswith(("bgr_", "rgb_"))]:
        y, p = [r["actual"] for r in rows], [r[col] for r in rows]
        summary[col] = {"accuracy": round(accuracy_score(y, p), 4),
                        "macro_f1": round(f1_score(y, p, labels=EMOTIONS, average="macro", zero_division=0), 4),
                        "predicted_class_counts": {e: p.count(e) for e in EMOTIONS if p.count(e)}}
        print(col, summary[col])
    summary["n_clips"] = len(rows); summary["actors"] = actors
    summary["note"] = "Pilot only. Training/test overlap with these actors is unverified."
    json.dump(summary, open(os.path.join(OUT, "variants_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
