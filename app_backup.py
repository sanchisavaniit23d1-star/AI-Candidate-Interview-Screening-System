import streamlit as st
import json
import os
import matplotlib.pyplot as plt

from predict import generate_timeline


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI Candidate Interview Screening",
    page_icon="🎤",
    layout="wide"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("🎤 AI Candidate Interview Screening System")

st.write(
    "Upload an interview recording to analyze emotional patterns "
    "and generate HR analytics."
)


# --------------------------------------------------
# Upload section
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Interview Video",
    type=["mp4", "avi", "mov", "mkv"]
)


# --------------------------------------------------
# Analyze button
# --------------------------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔍 Analyze Interview"):

        # Save uploaded video temporarily
        video_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        os.makedirs("uploads", exist_ok=True)

        with open(video_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        st.info("Analyzing interview...")

        # Temporary prediction
        timeline = generate_timeline(
            duration=20,
            window_size=2
        )

        # Save timeline
        result = {
            "timeline": timeline
        }

        with open("timeline.json", "w") as file:
            json.dump(result, file, indent=2)

        st.success("Analysis complete!")


        # --------------------------------------------------
        # Overall scores
        # --------------------------------------------------

        latest = timeline[-1]

        st.subheader("HR Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Composure",
                f"{latest['composure']:.1f}"
            )

        with col2:
            st.metric(
                "Stress",
                f"{latest['stress']:.1f}"
            )

        with col3:
            st.metric(
                "Engagement",
                f"{latest['engagement']:.1f}"
            )


        # --------------------------------------------------
        # Timeline graph
        # --------------------------------------------------

        st.subheader("Emotion & HR Analytics Timeline")

        times = [
            point["time"]
            for point in timeline
        ]

        composure = [
            point["composure"]
            for point in timeline
        ]

        stress = [
            point["stress"]
            for point in timeline
        ]

        engagement = [
            point["engagement"]
            for point in timeline
        ]


        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            times,
            composure,
            marker="o",
            label="Composure"
        )

        ax.plot(
            times,
            stress,
            marker="o",
            label="Stress"
        )

        ax.plot(
            times,
            engagement,
            marker="o",
            label="Engagement"
        )

        ax.set_xlabel("Interview Time (seconds)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 100)

        ax.grid(True)
        ax.legend()

        st.pyplot(fig)


        # --------------------------------------------------
        # Emotion probabilities
        # --------------------------------------------------

        st.subheader("Emotion Analysis")

        final_emotions = timeline[-1]["emotions"]

        for emotion, probability in final_emotions.items():

            st.write(
                f"**{emotion.capitalize()}**: "
                f"{probability * 100:.1f}%"
            )

            st.progress(float(probability))


        st.caption(
            "⚠️ Current prototype uses a temporary prediction "
            "engine. Model-based emotion predictions will be "
            "connected once the compatible pretrained checkpoint "
            "is available."
        )