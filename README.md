# Real-Time Facial Emotion Recogniser

P9 MSc Data Science project by Amal and Mahesha. A React webcam application sends frames to a
FastAPI service, which detects faces with OpenCV and classifies each detected face into one of
the seven FER-2013 emotions: angry, disgust, fear, happy, neutral, sad, and surprise.

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

## Main project files

- `backend/` — FastAPI application, model loading, image preprocessing, and face detection.
- `frontend/src/` — React application, webcam, prediction display, and styles.
- `notebooks/` — EDA, from-scratch convolution, CNN training, feature-map analysis, architecture
  experiments, and evaluation scripts.
- `models/` — trained checkpoint and generated evaluation artifacts.
- `config.py`, `data_pipeline.py` — dataset configuration, reproducible splits, transforms, and
  class weights.
