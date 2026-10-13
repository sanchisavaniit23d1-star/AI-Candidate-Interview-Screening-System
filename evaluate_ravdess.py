import os
import csv
import json
import traceback
import numpy as np
import librosa

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from interview_analyzer_old import (
    load_model,
    extract_audio,
    prepare_faces,
    predict_window,
    SAMPLE_RATE,
    EMOTIONS,
)

DATASET_DIR = r"E:\AI-Interview-Project\RAVDESS\Actor_01"
OUTPUT_DIR = os.path.join("uploads", "ravdess_pilot_evaluation")

# RAVDESS emotion codes in the filename's third field.
EMOTION_CODES = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    clips = sorted(
        name for name in os.listdir(DATASET_DIR)
        if name.lower().endswith(".mp4")
        and name.split("-")[0] == "01"
    )

    print(f"Audio-video clips found: {len(clips)}")
    if not clips:
        raise RuntimeError("No audio-video clips found.")

    print("Loading model once...")
    model = load_model()

    rows = []
    failures = []

    for i, filename in enumerate(clips, start=1):
        path = os.path.join(DATASET_DIR, filename)
        parts = filename[:-4].split("-")

        if len(parts) != 7 or parts[2] not in EMOTION_CODES:
            failures.append({
                "file": filename,
                "error": "Unexpected RAVDESS filename format",
            })
            continue

        actual = EMOTION_CODES[parts[2]]
        work_dir = os.path.join(
            OUTPUT_DIR, os.path.splitext(filename)[0]
        )
        audio_path = os.path.join(work_dir, "audio.wav")
        faces_dir = os.path.join(work_dir, "faces")
        os.makedirs(work_dir, exist_ok=True)

        print(f"[{i}/{len(clips)}] {filename} -> expected: {actual}")

        try:
            extract_audio(path, audio_path)
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)

            num_windows = prepare_faces(path, faces_dir)
            window_results = []

            for window_index in range(num_windows):
                prediction = predict_window(
                    model, audio, window_index, faces_dir
                )
                if prediction is not None:
                    window_results.append(prediction)

            if not window_results:
                raise RuntimeError("No valid prediction windows.")

            # Average the model's eight-class probabilities across the clip.
            prob_arrays = [
                [w["fusion"]["probabilities"][e] for e in EMOTIONS]
                for w in window_results
            ]
            mean_probs = np.mean(prob_arrays, axis=0)
            total = float(mean_probs.sum())

            if not np.isfinite(mean_probs).all() or total <= 0:
                raise RuntimeError("Invalid averaged probabilities.")

            mean_probs = mean_probs / total
            predicted = EMOTIONS[int(np.argmax(mean_probs))]

            row = {
                "file": filename,
                "actual": actual,
                "predicted": predicted,
                "windows_processed": len(window_results),
            }
            for emotion, probability in zip(EMOTIONS, mean_probs):
                row[f"prob_{emotion}"] = round(float(probability), 6)

            rows.append(row)

        except Exception as exc:
            failures.append({
                "file": filename,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  FAILED: {exc}")

    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        fields = (
            ["file", "actual", "predicted", "windows_processed"]
            + [f"prob_{e}" for e in EMOTIONS]
        )
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    failures_path = os.path.join(OUTPUT_DIR, "failures.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    if rows:
        y_true = [r["actual"] for r in rows]
        y_pred = [r["predicted"] for r in rows]

        metrics = {
            "evaluation_type": "single-actor RAVDESS pilot",
            "checkpoint_training_overlap_verified": False,
            "clips_found": len(clips),
            "clips_evaluated": len(rows),
            "clips_failed": len(failures),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=EMOTIONS,
                         average="macro", zero_division=0)
            ),
            "labels": EMOTIONS,
            "confusion_matrix_rows_actual_columns_predicted":
                confusion_matrix(y_true, y_pred, labels=EMOTIONS).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=EMOTIONS,
                output_dict=True, zero_division=0
            ),
            "warning": (
                "Single actor only. Checkpoint overlap with these clips "
                "is unverified. These results do not establish real-world "
                "interview accuracy or hiring suitability."
            ),
        }

        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\n===== PILOT RESULTS =====")
        print(f"Evaluated: {len(rows)}/{len(clips)}")
        print(f"Failed: {len(failures)}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print("\nPer-class metrics:")
        print(classification_report(
            y_true, y_pred, labels=EMOTIONS,
            zero_division=0
        ))
        print("\nConfusion matrix (rows=actual, columns=predicted):")
        print(confusion_matrix(y_true, y_pred, labels=EMOTIONS))
        print(f"\nSaved results in: {OUTPUT_DIR}")
    else:
        print("No clips were successfully evaluated; check failures.json.")

    if failures:
        print(f"Failure details: {failures_path}")


if __name__ == "__main__":
    main()