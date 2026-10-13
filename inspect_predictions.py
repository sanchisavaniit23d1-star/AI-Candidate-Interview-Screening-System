import json
from pathlib import Path
from collections import Counter

files = [
    Path("uploads/test_sample_1_analysis/timeline_results.json"),
    Path("uploads/test_sample_2_analysis/timeline_results.json"),
]

for path in files:
    data = json.loads(path.read_text(encoding="utf-8"))
    print("\n" + "=" * 60)
    print(data["video"])

    winners = Counter()
    margins = []
    confidences = []

    for window in data["timeline"]:
        probs = window["fusion"]["probabilities"]
        ranked = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        winners[ranked[0][0]] += 1
        confidences.append(ranked[0][1])
        margins.append(ranked[0][1] - ranked[1][1])

        print(
            "Window {:02d} ({:5.1f}-{:.1f}s): "
            "{:10s} {:6.2f}% | runner-up {:10s} {:6.2f}% | margin {:6.2f}%".format(
                window["window"],
                window["start_time"],
                window["end_time"],
                ranked[0][0],
                ranked[0][1] * 100,
                ranked[1][0],
                ranked[1][1] * 100,
                margins[-1] * 100,
            )
        )

    n = len(data["timeline"])
    print("\nWinning-class counts:", dict(winners))
    print("Mean winning probability: {:.2f}%".format(
        sum(confidences) / n * 100
    ))
    print("Mean top-two probability margin: {:.2f}%".format(
        sum(margins) / n * 100
    ))
