# AI-Based Interview Analysis using Audiovisual Emotion Recognition

An interview video is analysed using **both the candidate's face and voice**. The system splits the video into 3.6-second windows and estimates emotion probabilities per window with a multimodal deep-learning model (audio branch + visual branch + attention-based fusion). The results are saved as a timeline in JSON, together with simple HR-oriented indicators derived from the probabilities.

**Project status: Working prototype (Project-1 stage).** The audiovisual inference pipeline processes uploaded videos, and the Streamlit UI displays model predictions and experimental HR proxy indicators. The prototype has not undergone labeled accuracy evaluation and is not a validated hiring or psychological assessment tool. See [Project status](#project-status).

Sarvajanik College of Engineering and Technology, Information Technology Department. Supervisor: Prof. Bhumika Patel.

---

## Team

| Enrollment | Name | Pair |
|---|---|---|
| ET23BTIT046 | Riddhi Senjaliya | B |
| ET23BTIT048 | Sanchi Savani | A |
| ET23BTIT057 | Utsav Davaria | B |
| ET23BTIT068 | Kavya Bhalala | A |

- **Pair A, ML & Data Engineering:** dataset, preprocessing, model, inference, audio features, visual features, fusion, emotion probabilities, timeline generation, HR scoring logic.
- **Pair B, Full Stack & Integration:** frontend, API, upload workflow, backend integration, visualization, database, final user interface.

---

## Project status

| Component | Status |
|---|---|
| Load pretrained multimodal checkpoint (`strict=True`) | Implemented |
| Audio loading and 10-coefficient MFCC extraction (22,050 Hz) | Implemented |
| Face detection and visual preprocessing | Implemented |
| Fused audiovisual emotion inference | Implemented |
| Full-video processing in 3.6-second windows | Implemented in the current analyzer |
| Streamlit UI with video upload, analysis, charts and caching | Implemented as a prototype |
| Exploratory HR proxy indicators | Implemented; experimental and not validated |
| Labeled accuracy evaluation on our own data | Not completed |
| FastAPI backend and database | Planned |
---

## How it works

Interview video
|-- Video: frames -> MTCNN face crop -> 224x224 RGB -> 15 frames per 3.6-second window
|            -> EfficientFace -> visual features
|-- Audio: waveform -> 22,050 Hz mono -> 10 MFCC coefficients
|            -> Audio CNN -> audio features
|-- Fusion: audio + visual features -> intermediate attention fusion
|            -> 8 emotion probabilities per window
|-- Output: timeline JSON + experimental HR proxy indicators
- **Emotions (8):** neutral, calm, happy, sad, angry, fearful, disgust, surprised.
- **Model input:** audio `[batch, 10, time]`, video `[batch, 15, 3, 224, 224]` (in `predict_modality_test.py` the visual tensor is passed per sample as `[15, 3, 224, 224]`).
- **Audio-only / visual-only results** in our script are produced by running the *fused* model with the other modality's input set to zeros. They are probes, not separately trained classifiers.
- **Pretrained components (not trained by us):**
  - `EfficientFace_Trained_on_AffectNet7.pth.tar`: pretrained facial-expression network that initializes the visual branch.
  - `ia_1head_moddrop_2.pth`: pretrained multimodal checkpoint (intermediate attention, 1 head, modality dropout). *Source: https://github.com/anuj3509/Multimodal-Emotion-Recognition-System/blob/478769b40e32e94fa78292f6821f64b798d25613/pretrained_models/ia_1head_moddrop_2.pth*

---

## Installation

The base code was tested with **Python 3.9 and PyTorch 1.9.0** (our environment also used Python 3.9).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install facenet-pytorch streamlit matplotlib soundfile
```

`requirements.txt` currently pins `torch==1.9.0`, `torchvision==0.10.0`, `librosa==0.8.1`, plus `opencv-python` and `tqdm`. The extra packages above are used by our scripts but are not yet listed there. *To do: pin exact versions with `pip freeze` from the working environment.*

FFmpeg is needed to extract the audio track from a video.

### Required files (not tracked in Git)
Place these in the project folder:
- `ia_1head_moddrop_2.pth`
- `EfficientFace_Trained_on_AffectNet7.pth.tar`
- a test video with an audio track (e.g. `test_sample_2.mp4`)

---

## Running the demo

**1. Create the face windows** (6 windows × 15 face crops from the first 21.6 s):
```powershell
python prepare_test_sample_2.py          # reads test_sample_2.mp4, writes timeline_faces_mtcnn/window_1..6
```

**2. Extract the audio** (22,050 Hz, mono), for example:
```powershell
ffmpeg -i test_sample_2.mp4 -vn -ac 1 -ar 22050 test_audio/test_sample_2_audio.wav
```

**3. Run the multimodal timeline**
```powershell
python predict_modality_test.py
```
Prints an audio diagnostic, then audio / visual / fused predictions per window, and saves `timeline_results.json`.

**4. Run the UI prototype**
```powershell
streamlit run app.py
```
> The UI currently calls a **placeholder** predictor that generates random probabilities (`predict.py`) and ignores the uploaded video's content. It is a layout prototype until the model is connected.

### Important: the audio must not be silent
One of our first test videos had an all-zero audio track. The audio diagnostic printed at the start of `predict_modality_test.py` (min / max / std) makes this visible. Audio-branch outputs on silent audio are not meaningful.

---

## Example output (real, from `predict_modality_test.py`)

| Window | Time (s) | Audio probe | Visual probe | Fused |
|---|---|---|---|---|
| 1 | 0.0–3.6 | surprised 99.95% | angry 84.39% | surprised 63.58% |
| 2 | 3.6–7.2 | surprised 99.95% | angry 63.89% | surprised 99.13% |
| 3 | 7.2–10.8 | surprised 94.80% | angry 66.06% | surprised 36.31% |
| 4 | 10.8–14.4 | surprised 61.75% | angry 39.61% | surprised 94.93% |
| 5 | 14.4–18.0 | surprised 50.06% | angry 75.36% | happy 81.75% |
| 6 | 18.0–21.6 | happy 80.03% | angry 78.54% | happy 97.23% |

- Percentages are **softmax probabilities for that window, not accuracy**.
- The modalities can disagree (window 6). A higher fused probability does not mean the fused result is more correct.
- One test video; no accuracy evaluation has been done on our own data.

### `timeline_results.json` (per window)
`window`, `start`, `end`, `audio_emotion`, `visual_emotion`, `multimodal_emotion`, `audio_probabilities`, `visual_probabilities`, `multimodal_probabilities`, `hr_scores`.

### HR-style indicators
Computed from the **fused** probabilities (`hr_scoring.py`):
- Composure = calm + happy + neutral
- Stress = angry + fearful + sad + disgust
- Engagement = happy + surprised + neutral

These are **project-defined heuristics, not validated psychological measurements**. An emotion classifier cannot measure a person's stress level.

---

## Repository layout (current)

```
app.py                       Streamlit UI prototype (placeholder predictor)
predict.py                   Placeholder timeline generator (random probabilities)
hr_scoring.py                HR-style indicator formulas
dashboard.py                 Matplotlib plot of timeline.json
predict_modality_test.py     Main demo: real multimodal timeline → timeline_results.json
prepare_test_sample_2.py     Video → MTCNN face windows
models/                      MultiModalCNN, EfficientFace, attention (from base project)
datasets/, dataset.py, main.py, train.py, validation.py, opts.py, transforms.py, utils.py
                             Training/evaluation code from the base project (not used for inference)
ravdess_preprocessing/       Dataset preprocessing scripts from the base project
```

---

## Limitations

- The pretrained model is associated with RAVDESS, an acted-emotion dataset; generalization to real interviews has not been established.
- Two sample videos have been processed by the current pipeline, but neither provides a labeled evaluation of model accuracy.
- In the separate six-window modality-probe experiment, the visual probe predicted *angry* in all six windows. These are zero-input probes of the fused model, not independently trained modality-specific classifiers.
- The analyzer uses non-overlapping 3.6-second windows and replaces missing face detections with black frames.
- Softmax probabilities indicate the model's output distribution; they are not accuracy measurements.
- Composure, stress-associated emotion, and engagement scores are project-defined probability-based proxies, not validated psychological measurements or measures of candidate suitability.
- Robustness to missing or noisy modalities has not been independently evaluated in this project.

---

## Planned work
Full-length video processing; automatic upload → processing → results pipeline (FastAPI); database; dashboard; evaluation on labeled clips and real interview recordings; ablation of audio / visual / fused results.

---

## Credits and licence

This project adapts the public implementation of **"Self-attention fusion for audiovisual emotion recognition with incomplete data"** (K. Chumachenko, A. Iosifidis, M. Gabbouj, ICPR 2022), repository: <https://github.com/katerynaCh/multimodal-emotion-recognition> (MIT licence, see `LICENSE`). The original README is kept in `docs/UPSTREAM_README.md`.

Also used: EfficientFace (Zhao et al.), facenet-pytorch (MTCNN), PyTorch, librosa, OpenCV, Streamlit. Datasets/checkpoints have their own licences; the upstream README asks users to check the RAVDESS licence before using the shared models.
