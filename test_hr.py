from hr_scoring import calculate_scores


sample = {
    "neutral": 0.40,
    "calm": 0.30,
    "happy": 0.15,
    "sad": 0.05,
    "angry": 0.03,
    "fearful": 0.02,
    "disgust": 0.01,
    "surprised": 0.04
}

scores = calculate_scores(sample)

print("Emotion probabilities:")
for emotion, probability in sample.items():
    print(f"{emotion}: {probability:.2f}")

print("\nHR Analytics:")
print(f"Composure:  {scores['composure']}")
print(f"Stress:     {scores['stress']}")
print(f"Engagement: {scores['engagement']}")