import json
import os
from collections import Counter
from statistics import mean

FILES = [
    "uploads/test_sample_1_analysis/timeline_results.json",
    "uploads/test_sample_2_analysis/timeline_results.json",
]

def inspect_file(path):
    with open(path, "r", encoding="utf-8") as f:
        result = json.load(f)

    timeline = result.get("timeline", [])
    if not timeline:
        raise ValueError("No timeline entries found.")

    emotions = Counter()
    confidences = []
    score_values = {"composure": [], "stress": [], "engagement": []}
    probability_sum_errors = []
    invalid_windows = 0
    gaps = []

    previous_end = None

    for point in timeline:
        fusion = point.get("fusion", {})
        probabilities = fusion.get("probabilities", {})
        emotion = fusion.get("emotion")

        if not probabilities or emotion not in probabilities:
            invalid_windows += 1
            continue

        emotions[emotion] += 1
        confidences.append(float(probabilities[emotion]))

        probability_sum_errors.append(
            abs(sum(float(v) for v in probabilities.values()) - 1.0)
        )

        scores = point.get("hr_scores", {})
        for name in score_values:
            if name in scores:
                score_values[name].append(float(scores[name]))

        start = float(point["start_time"])
        end = float(point["end_time"])

        if end < start:
            invalid_windows += 1

        if previous_end is not None:
            gaps.append(start - previous_end)

        previous_end = end

    valid_count = sum(emotions.values())

    print("\n" + "=" * 58)
    print("PIPELINE DIAGNOSTICS:", os.path.basename(path))
    print("=" * 58)
    print("Video:", result.get("video", "Unknown"))
    print("Timeline windows:", len(timeline))
    print("Valid prediction windows:", valid_count)
    print("Invalid or incomplete windows:", invalid_windows)

    if valid_count:
        print("\nPredicted emotion distribution:")
        for emotion, count in emotions.most_common():
            print(
                "  {:12s} {:3d} windows ({:.1f}%)".format(
                    emotion, count, 100 * count / valid_count
                )
            )

        print(
            "\nMean winning-class probability: {:.2f}%".format(
                100 * mean(confidences)
            )
        )

        print(
            "Mean probability-sum deviation from 1: {:.6f}".format(
                mean(probability_sum_errors)
            )
        )

        print("\nMean model-derived HR proxy scores:")
        for name, values in score_values.items():
            if values:
                print("  {}: {:.2f}".format(name.title(), mean(values)))
            else:
                print("  {}: unavailable".format(name.title()))

    if gaps:
        print("\nTimeline continuity:")
        print("  Largest absolute gap/overlap: {:.3f} seconds".format(
            max(abs(gap) for gap in gaps)
        ))
        print("  Negative gaps indicate overlapping windows.")
        print("  Positive gaps indicate gaps between windows.")

    print("\nInterpretation: diagnostic results only; no ground-truth")
    print("labels are available here, so accuracy/F1 are not calculated.")

def main():
    report = {
        "evaluation_type": "unlabeled_pipeline_diagnostics",
        "files": []
    }

    for path in FILES:
        if not os.path.isfile(path):
            print("\nMissing file:", path)
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)

            timeline = result.get("timeline", [])
            counts = Counter(
                p.get("fusion", {}).get("emotion", "unknown")
                for p in timeline
            )

            confidences = [
                float(p["fusion"]["probabilities"][p["fusion"]["emotion"]])
                for p in timeline
                if p.get("fusion", {}).get("emotion")
                in p.get("fusion", {}).get("probabilities", {})
            ]

            report["files"].append({
                "file": path,
                "video": result.get("video", "Unknown"),
                "windows": len(timeline),
                "emotion_counts": dict(counts),
                "mean_winning_probability": (
                    mean(confidences) if confidences else None
                ),
            })

            inspect_file(path)

        except Exception as exc:
            print("\nCould not inspect {}: {}".format(path, exc))

    output_path = "uploads/pipeline_diagnostics.json"
    os.makedirs("uploads", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nSummary JSON saved to:", output_path)

if __name__ == "__main__":
    main()
