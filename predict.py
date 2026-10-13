import json
import random
from hr_scoring import calculate_scores


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


def generate_emotion_probabilities():
    """Temporary predictor until the trained model checkpoint is available."""

    values = [random.random() for _ in EMOTIONS]
    total = sum(values)

    probabilities = {
        emotion: value / total
        for emotion, value in zip(EMOTIONS, values)
    }

    return probabilities


def generate_timeline(duration=20, window_size=2):

    timeline = []

    for time in range(0, duration, window_size):

        probabilities = generate_emotion_probabilities()
        scores = calculate_scores(probabilities)

        timeline.append({
            "time": time,
            "emotions": probabilities,
            "composure": scores["composure"],
            "stress": scores["stress"],
            "engagement": scores["engagement"]
        })

    return timeline


def predict():

    timeline = generate_timeline(duration=20, window_size=2)

    result = {
        "timeline": timeline
    }

    # Save results as JSON
    with open("timeline.json", "w") as file:
        json.dump(result, file, indent=2)

    print("Analysis complete.")
    print("Timeline saved to timeline.json")


if __name__ == "__main__":
    predict()