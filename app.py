import hashlib
import json
import os

import matplotlib.pyplot as plt
import streamlit as st

from interview_analyzer import analyze_interview


ANALYSIS_CACHE_DIR = os.path.join("uploads", "analysis_cache")
os.makedirs(ANALYSIS_CACHE_DIR, exist_ok=True)


st.set_page_config(
    page_title="AI Candidate Interview Screening",
    page_icon="🎤",
    layout="wide",
)


# Session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analyzed_file_hash" not in st.session_state:
    st.session_state.analyzed_file_hash = None

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = None


# Page title
st.title("🎤 AI Candidate Interview Screening System")
st.write(
    "Upload an interview recording to analyze emotional patterns "
    "and generate HR analytics."
)


# Video upload
uploaded_file = st.file_uploader(
    "Upload Interview Video",
    type=["mp4", "avi", "mov", "mkv"],
)


if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    st.success(f"Uploaded: {uploaded_file.name}")

    # Clear results from a different video.
    if st.session_state.analyzed_file_hash != file_hash:
        st.session_state.analysis_result = None
        st.session_state.analyzed_file_hash = None
        st.session_state.analysis_error = None

    if st.button("🔍 Analyze Interview", key="analyze_interview"):
        os.makedirs("uploads", exist_ok=True)
        os.makedirs(ANALYSIS_CACHE_DIR, exist_ok=True)

        safe_filename = os.path.basename(uploaded_file.name)
        video_path = os.path.join("uploads", safe_filename)
        cache_path = os.path.join(ANALYSIS_CACHE_DIR, f"{file_hash}.json")

        try:
            analysis_result = None
            timeline = []
            loaded_from_cache = False

            # 1. Try loading and validating the cached result.
            if os.path.isfile(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as file:
                        cached_result = json.load(file)

                    cached_timeline = (
                        cached_result.get("timeline")
                        if isinstance(cached_result, dict)
                        else None
                    )

                    if not isinstance(cached_timeline, list) or not cached_timeline:
                        raise ValueError("Cached timeline is missing or empty.")

                    for point in cached_timeline:
                        if not isinstance(point, dict):
                            raise ValueError("Invalid timeline entry.")

                        required_fields = (
                            "start_time",
                            "end_time",
                            "fusion",
                            "hr_scores",
                        )
                        if not all(key in point for key in required_fields):
                            raise ValueError(
                                "Timeline entry is missing required fields."
                            )

                        if not isinstance(point["fusion"], dict):
                            raise ValueError("Invalid fusion data.")

                        if not isinstance(
                            point["fusion"].get("probabilities"), dict
                        ):
                            raise ValueError("Emotion probabilities are missing.")

                        if not isinstance(point["hr_scores"], dict):
                            raise ValueError("Invalid HR score data.")

                        required_scores = ("composure", "stress", "engagement")
                        if not all(
                            key in point["hr_scores"] for key in required_scores
                        ):
                            raise ValueError("HR scores are incomplete.")

                    analysis_result = cached_result
                    timeline = cached_timeline
                    loaded_from_cache = True
                    st.info("Loaded the previously saved analysis.")

                except (
                    json.JSONDecodeError,
                    OSError,
                    ValueError,
                    TypeError,
                ):
                    # IMPORTANT: this is the inner handler. A bad cache should
                    # not escape to the outer handler; it should trigger inference.
                    st.warning(
                        "The saved analysis is invalid or unreadable. "
                        "Running the analysis again."
                    )
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass

            # 2. If the cache was absent or invalid, analyze the video.
            if not loaded_from_cache:
                with open(video_path, "wb") as file:
                    file.write(file_bytes)

                with st.spinner(
                    "Analyzing interview video. This may take some time..."
                ):
                    analysis_result = analyze_interview(video_path)

                if not isinstance(analysis_result, dict):
                    raise ValueError("Analyzer returned an invalid result.")

                timeline = analysis_result.get("timeline")
                if not isinstance(timeline, list) or not timeline:
                    raise ValueError(
                        "The analyzer completed but returned an empty or invalid timeline."
                    )

                # Save the new valid result for future uploads.
                with open(cache_path, "w", encoding="utf-8") as file:
                    json.dump(analysis_result, file, indent=2)

                st.success("Analysis completed and cached.")

            # 3. Save the result in session state and the standard timeline file.
            st.session_state.analysis_result = analysis_result
            st.session_state.analyzed_file_hash = file_hash
            st.session_state.analysis_error = None

            with open("timeline.json", "w", encoding="utf-8") as file:
                json.dump({"timeline": timeline}, file, indent=2)

        except Exception as error:
            st.session_state.analysis_result = None
            st.session_state.analyzed_file_hash = None
            st.session_state.analysis_error = str(error)

    # Show analysis errors, if any.
    if st.session_state.analysis_error:
        st.error(f"Analysis failed: {st.session_state.analysis_error}")

    # Display saved results without running inference again.
    analysis_result = st.session_state.analysis_result

    if (
        analysis_result is not None
        and st.session_state.analyzed_file_hash == file_hash
    ):
        timeline = analysis_result["timeline"]
        st.success("Analysis results ready.")

        # Latest model-derived proxy scores
        latest = timeline[-1]
        latest_scores = latest["hr_scores"]

        st.subheader("Exploratory HR Proxy Indicators")
        st.caption(
            "These scores are calculated from predicted emotion probabilities. "
            "They are experimental indicators, not validated measurements of "
            "psychological traits or candidate suitability."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Composure Proxy",
                f"{latest_scores['composure']:.1f}/100",
            )

        with col2:
            st.metric(
                "Stress-Associated Emotion Proxy",
                f"{latest_scores['stress']:.1f}/100",
            )

        with col3:
            st.metric(
                "Engagement Proxy",
                f"{latest_scores['engagement']:.1f}/100",
            )

        # Timeline of exploratory proxy scores
        st.subheader("Proxy Indicators Over Time")

        times = [
            (point["start_time"] + point["end_time"]) / 2
            for point in timeline
        ]
        composure = [
            point["hr_scores"]["composure"] for point in timeline
        ]
        stress = [
            point["hr_scores"]["stress"] for point in timeline
        ]
        engagement = [
            point["hr_scores"]["engagement"] for point in timeline
        ]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

        chart_data = [
            ("Composure Proxy", composure),
            ("Stress-Associated Emotion Proxy", stress),
            ("Engagement Proxy", engagement),
        ]

        for ax, (label, values) in zip(axes, chart_data):
            ax.plot(
                times,
                values,
                marker="o",
                markersize=4,
                linewidth=1.8,
            )
            ax.set_ylabel("Score / 100")
            ax.set_ylim(0, 100)
            ax.set_title(label, loc="left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Interview Time (seconds)")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Latest emotion probabilities
        st.subheader("Emotion Analysis")
        final_emotions = timeline[-1]["fusion"]["probabilities"]

        for emotion, probability in final_emotions.items():
            st.write(f"**{emotion.capitalize()}**: {probability * 100:.1f}%")
            st.progress(max(0.0, min(1.0, float(probability))))

        st.caption(
            "Note: This prototype uses a pretrained audiovisual emotion "
            "recognition model. Emotion predictions and proxy scores are "
            "experimental model-derived indicators, not validated "
            "psychological measurements or hiring recommendations."
        )
else:
    st.info("Upload an interview video to begin analysis.")
