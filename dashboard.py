import json
import matplotlib.pyplot as plt


# Load analysis results
with open("timeline.json", "r") as file:
    data = json.load(file)


# Get timeline
timeline = data["timeline"]


# Extract values
times = [point["time"] for point in timeline]

composure = [point["composure"] for point in timeline]
stress = [point["stress"] for point in timeline]
engagement = [point["engagement"] for point in timeline]


# Create graph
plt.figure(figsize=(10, 5))

plt.plot(times, composure, marker="o", label="Composure")
plt.plot(times, stress, marker="o", label="Stress")
plt.plot(times, engagement, marker="o", label="Engagement")


plt.xlabel("Interview Time (seconds)")
plt.ylabel("Score")
plt.title("Interview HR Analytics")

plt.ylim(0, 100)
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()