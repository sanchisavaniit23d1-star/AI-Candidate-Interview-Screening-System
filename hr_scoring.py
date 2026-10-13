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


def calculate_scores(probabilities):
    """
    probabilities: dictionary containing emotion probabilities
    """

    positive = (
        probabilities["calm"] +
        probabilities["happy"] +
        probabilities["neutral"]
    )

    negative = (
        probabilities["angry"] +
        probabilities["fearful"] +
        probabilities["sad"] +
        probabilities["disgust"]
    )

    engagement = (
        probabilities["happy"] +
        probabilities["surprised"] +
        probabilities["neutral"]
    )

    composure = max(0, min(100, positive * 100))
    stress = max(0, min(100, negative * 100))
    engagement = max(0, min(100, engagement * 100))

    return {
        "composure": round(composure, 2),
        "stress": round(stress, 2),
        "engagement": round(engagement, 2)
    }