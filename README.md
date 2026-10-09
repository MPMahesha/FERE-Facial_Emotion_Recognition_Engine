# Real-Time Facial Emotion Recogniser

P9 MSc Data Science project by Amal and Mahesha. A React webcam application sends frames to a
FastAPI service, which detects faces with OpenCV and classifies each detected face into one of
the seven FER-2013 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

The same web application also includes a **voice / text emotion** mode. It transcribes speech in
the browser (Web Speech API) and classifies the transcript with a locally trained
**TF-IDF + Logistic Regression** NLP model into one of six DAIR.AI emotions: sadness, joy, love,
anger, fear, and surprise. These six text classes are **different** from the seven facial classes
and come from a different dataset and model.

## Run locally

Requirements: Python 3.10 or newer and Node.js 18 or newer.

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Start the API from the repository root:

```bash
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`). The API status endpoint
is `http://localhost:8000/status`; interactive API documentation is at
`http://localhost:8000/docs`. Allow camera access in the browser. The frontend defaults to an API
at `http://localhost:8000`.

## Dataset and training

The training scripts expect FER-2013 in class folders:

```text
Dataset/archive/
├── train/<emotion>/
└── test/<emotion>/
```

By default, `config.py` uses `Dataset/archive` relative to the project root. To use a different
location, set `FER_DATASET_PATH` to the absolute path of the directory containing `train` and
`test` before running the scripts. The data pipeline makes a fixed-seed, stratified validation
split from `train`; the official `test` split is kept separate.

Train the VGG-style CNN and save its best validation checkpoint to `models/emotion_cnn.pth`:

```bash
python notebooks/03_train_cnn.py --epochs 15
```

The API loads `models/emotion_cnn.pth` when it starts. The training pipeline uses 48×48 grayscale
images, augmentation, batch normalization, dropout, and square-root-damped class weights to
address FER-2013 class imbalance, especially the small disgust class.

## Model work and evaluation

- `notebooks/02_from_scratch.py` implements a 2D convolution with NumPy loops and checks its
  output against PyTorch.
- `notebooks/04_feature_map_analysis.py` generates layer-by-layer CNN feature-map visualisations.
- `notebooks/04_transfer_learning.py` contains ResNet-18, EfficientNet-B0, and a small ViT for
  architecture experiments. ResNet-18 and EfficientNet-B0 use upscaled 224×224 three-channel
  inputs. Run the experiment on a machine with the FER-2013 data configured:

  ```bash
  python notebooks/04_transfer_learning.py --epochs 10
  ```

- `models/experiments_comparison.json` records CNN training ablations. The earlier
  `models/model_comparison.json` contains short-run exploratory transfer results; do not treat
  those figures as full-training model comparisons.
- `models/final_evaluation_metrics.json` and `models/confusion_matrix.png` contain the final
  held-out test results for the deployed horizontal-flip averaged inference.

The final reported evaluation uses the original face crop and its horizontal flip, averaging
their softmax probabilities. Results on all 7,178 official test images:

| Test accuracy | Macro F1 | Weighted F1 | Parameters | Model size | Batch-1 CPU model FPS |
|---:|---:|---:|---:|---:|---:|
| 66.58% | 63.18% | 66.24% | 2,312,007 | 8.82 MB | 66.2 |

Accuracy exceeds the project minimum of 65%; it does not reach 70%. The benchmark is model-only,
not end-to-end webcam FPS. In the test confusion matrix, 239 of 1,247 sad faces were classified
as neutral, and 219 of 1,024 fear faces as sad. Disgust recall was 70.27% and precision was
43.09%.

## Web application

The frontend provides webcam controls, per-face labels and bounding boxes, temporally smoothed
probabilities, a seven-class probability chart, and a prediction history that can be exported as
CSV. Camera frames are processed for prediction and are never stored. The API exposes `POST
/predict` for frame inference and `GET /status` for readiness and model status.

The header has two tabs: **Facial Emotion** (the original live webcam pipeline) and **Voice & Text
Emotion** (the new NLP pipeline). Both share the same visual theme.

## Voice & text emotion recognition (NLP)

### Dataset and exact import line

The text classifier is trained on the DAIR.AI Emotion dataset from Hugging Face. It is loaded with
its predefined splits using exactly this line:

```python
dataset = load_dataset("dair-ai/emotion", "split")
```

The dataset has three predefined splits and six labels:

- `train` (16,000 rows), `validation` (2,000 rows), `test` (2,000 rows)
- Labels: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`

These are **not** the same as the FER-2013 facial labels. The two models are independent.

### Install the NLP dependencies

```bash
pip install datasets scikit-learn joblib
```

(or simply `pip install -r requirements.txt`, which now includes `datasets` and `joblib`).

The dataset downloads and caches automatically the first time the training script runs and
requires internet access to `https://huggingface.co`. If the download fails, the script prints a
clear error instead of failing silently.

### Train the text model

Run from the repository root:

```bash
python -m nlp.train_text_model
```

This writes the following to `models/text_emotion/`:

- `text_emotion_pipeline.joblib` — the fitted scikit-learn `Pipeline`
- `label_mapping.json` — label order and index/label mappings
- `metrics.json` — test accuracy, precision, recall, F1, and the confusion matrix
- `confusion_matrix.png` — confusion-matrix heatmap
- `model_card.json` — short summary of the chosen configuration

### NLP workflow

1. **Input** — recognised or manually typed text.
2. **Preprocessing** — `nlp/text_preprocessing.py` lowercases and collapses whitespace.
3. **Feature extraction** — a TF-IDF `FeatureUnion` combining word n-grams (1–2) and character
   n-grams (`char_wb`, 3–5); `min_df=2`, `max_features=50,000` each, `sublinear_tf=True`.
4. **Classification** — Logistic Regression with a fixed `random_state=42` and
   `class_weight='balanced'`.
5. **Output** — predicted emotion, confidence, and a probability for every class.

Preprocessing, vectorisation, and classification live in a single scikit-learn `Pipeline`, so the
exact same transformation is applied at training and inference time. `class_weight='balanced'`
counters the dataset skew towards joy and sadness (together ~63% of the data) so the smaller
`love` and `surprise` classes are not ignored.

**Model selection.** The regularisation strength `C` is chosen using only the validation split
(`C ∈ {2, 3, 4, 6}`, best macro-F1). The final model is then fit on train + validation and
evaluated **once** on the untouched test split.

### Evaluation metrics

Results on the 2,000-example test split (selected `C = 3.0`):

| Metric | Value |
|---|---:|
| Accuracy | 88.15% |
| Macro precision | 82.06% |
| Macro recall | 87.03% |
| Macro F1 | 84.07% |
| Weighted F1 | 88.39% |

Per-class F1: sadness 0.92, joy 0.91, anger 0.88, fear 0.84, love 0.77, surprise 0.72. The
balanced class weights raised the recall of the small `love` and `surprise` classes (to 0.85 and
0.86 respectively). The full confusion matrix is in
`models/text_emotion/confusion_matrix.png`.

### Backend endpoint

`POST /predict-text` accepts JSON `{ "text": "..." }` and returns:

```json
{
  "success": true,
  "emotion": "joy",
  "confidence": 0.9876,
  "probabilities": { "sadness": 0.001, "joy": 0.9876, "love": 0.008, "anger": 0.001, "fear": 0.001, "surprise": 0.001 },
  "analyzed_text": "the text that was classified",
  "model_type": "TF-IDF (word 1-2 + char_wb 3-5) + LogisticRegression (balanced)",
  "message": "Text emotion predicted successfully."
}
```

The response echoes the exact `analyzed_text` so the UI can show which words were classified.
The backend also watches the model file: if `models/text_emotion/text_emotion_pipeline.joblib`
changes on disk, it is reloaded automatically on the next request, so retraining does **not**
require restarting the server.

Validation rules:

- Empty or whitespace-only text returns `422`.
- Text longer than 1,000 characters returns `422`.
- If the model file is missing, the endpoint returns `503` with the training command — it never
  returns fake predictions.

The existing `POST /predict` facial endpoint is unchanged.

### Browser microphone permissions and compatibility

- Speech-to-text uses the browser **Web Speech API** (`SpeechRecognition` /
  `webkitSpeechRecognition`), currently best supported in **Google Chrome** and **Microsoft
  Edge**. Firefox and some Safari versions do not expose it — in that case the UI disables
  recording and lets the user **type the sentence manually**.
- The browser asks for microphone permission the first time you press record. If permission is
  denied, the UI shows an explanatory error and you can still type text.
- **These browser speech services are online.** Chrome/Edge send the audio to the vendor's online
  speech service (e.g. Google/Microsoft) for transcription; that step does **not** work offline.
  Only the NLP classification step is fully local.
- The app does not permanently store audio or transcripts. Audio is handled by the browser; only
  the editable transcript is sent to the local FastAPI backend for classification.

### Viva summary

The project combines two independent emotion-recognition pipelines. The facial pipeline is a
from-scratch CNN trained on FER-2013 for seven classes. The text pipeline is a classical NLP
baseline: the DAIR.AI emotion dataset, lightweight lowercasing, word + character TF-IDF features,
and balanced Logistic Regression, chosen because it is fast, reproducible, interpretable, and
strong for short sentences (~88% test accuracy, 0.84 macro F1). Probabilities come from
`predict_proba`, so the confidence scores shown in the UI are the model's real class
probabilities, not mock values.

## Main project files

- `backend/` — FastAPI application, model loading, image preprocessing, and face detection.
- `backend/text_emotion.py` — loads the trained text pipeline (`/predict-text`).
- `nlp/` — text preprocessing and the reproducible training/evaluation script.
- `frontend/src/` — React application, webcam, prediction display, speech/text UI, and styles.
- `notebooks/` — EDA, from-scratch convolution, CNN training, feature-map analysis, architecture
  experiments, and evaluation scripts.
- `models/` — trained checkpoints and generated evaluation artifacts.
- `models/text_emotion/` — trained NLP pipeline, label mapping, metrics, and confusion matrix.
- `config.py`, `data_pipeline.py` — dataset configuration, reproducible splits, transforms, and
  class weights.
