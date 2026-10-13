import subprocess
import librosa
import numpy as np
import os

VIDEO_FILE = "test_sample_1.mp4"

TARGET_SR = 22050
TARGET_DURATION = 3.6


def extract_audio(video_file, output_file):
    command = [
        "ffmpeg",
        "-y",
        "-i", video_file,
        "-vn",
        "-ac", "1",
        "-ar", str(TARGET_SR),
        output_file
    ]

    subprocess.run(command, check=True)


def preprocess_audio(audio_file):

    y, sr = librosa.load(
        audio_file,
        sr=TARGET_SR
    )

    target_length = int(
        TARGET_SR * TARGET_DURATION
    )

    if len(y) < target_length:
        y = np.pad(
            y,
            (0, target_length - len(y))
        )

    elif len(y) > target_length:
        start = (len(y) - target_length) // 2
        y = y[
            start:start + target_length
        ]

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=10
    )

    return y, mfcc


if __name__ == "__main__":

    os.makedirs(
        "test_audio",
        exist_ok=True
    )

    audio_file = (
        "test_audio/extracted_audio.wav"
    )

    print("Extracting audio...")

    extract_audio(
        VIDEO_FILE,
        audio_file
    )

    print("Audio extracted successfully.")

    audio, mfcc = preprocess_audio(
        audio_file
    )

    print("\nAudio preprocessing successful!")
    print("Sampling rate:", TARGET_SR)
    print(
        "Audio samples:",
        len(audio)
    )
    print(
        "Audio duration:",
        len(audio) / TARGET_SR,
        "seconds"
    )
    print(
        "MFCC shape:",
        mfcc.shape
    )