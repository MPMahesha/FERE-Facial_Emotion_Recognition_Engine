# P9: Real-Time Facial Emotion Recogniser in the Browser

**MSc Data Science Project**

An end-to-end web application that performs real-time facial emotion recognition from a live browser webcam stream.

---

## 🎭 Supported Emotions (FER-2013)

The application classifies 7 facial emotion categories:
1. **Angry** 😠
2. **Disgust** 🤢
3. **Fear** 😨
4. **Happy** 😄
5. **Neutral** 😐
6. **Sad** 😢
7. **Surprise** 😲

---

## 🏛️ Project Architecture & Data Flow

```text
+-----------------------+           +------------------------+
|  React Frontend (UI)  |  Base64   |    FastAPI Backend     |
|  - Glassmorphism UI   | --------> |  - OpenCV Haar Cascade |
|  - React Webcam       |  Frame    |  - Face Bounding Box   |
|  - Prediction Smooth  | <-------- |  - PyTorch CNN / Mock  |
|  - Probability Bars   |   JSON    |  - 7-Class Softmax     |
+-----------------------+           +------------------------+
```

1. **Browser Webcam**: Streams video and periodically captures frames as base64 JPEG strings.
2. **FastAPI (`/predict`)**: Receives the frame, decodes it into an OpenCV image matrix (`numpy.ndarray`).
3. **OpenCV Face Detection**: Uses standard Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to identify face bounding boxes `(x, y, width, height)`.
4. **PyTorch Inference**: Crops each detected face, resizes to 48x48 grayscale, normalizes pixel intensities `[0, 1]`, and passes through the CNN model (`EmotionCNN`).
   - *Phase 1 Fallback*: When no `.pth` weights are present, the backend utilizes an internal mock prediction engine with realistic probability distributions.
5. **Response & UI Visualization**: Returns face coordinates, dominant emotion, confidence percentage, and full 7-class probability distributions to update the glassmorphism UI in real-time.

---

## 📁 Directory Structure

```text
FERE-Facial_Emotion_Recognition_Engine/
│
├── backend/
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI endpoints (/predict, /status, /) & CORS setup
│   ├── model_loader.py      # PyTorch EmotionCNN architecture & mock fallback engine
│   └── utils.py             # Image decoding & OpenCV Haar Cascade face detection
│
├── frontend/
│   ├── package.json         # React 18 & Vite configuration
│   ├── index.html           # HTML5 shell with Google Fonts
│   ├── vite.config.js       # Vite bundler configuration
│   ├── src/
│   │   ├── main.jsx         # React DOM mount point
│   │   ├── App.jsx          # Main application container, smoothing & logging
│   │   ├── components/
│   │   │   ├── Header.jsx           # App title & dynamic system status badges
│   │   │   ├── Webcam.jsx           # Live webcam stream & bounding box canvas overlay
│   │   │   ├── EmotionResult.jsx    # Dominant emotion highlight card with dynamic glow
│   │   │   └── ProbabilityBars.jsx  # 7-class probability distribution bars
│   │   └── styles/
│   │       └── style.css            # Glassmorphism aesthetic stylesheet
│   └── public/
│
├── models/
│   ├── emotion_cnn.pth      # Trained VGG-style FER-2013 checkpoint
│   ├── final_evaluation_metrics.json
│   ├── experiments_comparison.json
│   ├── model_comparison.json
│   ├── confusion_matrix.png
│   └── feature_maps_analysis.png
├── config.py                # Dataset paths, class names, split and seed
├── data_pipeline.py         # FER-2013 loading, augmentation and class weights
├── notebooks/               # EDA, training, transfer learning and evaluation scripts
│
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation & viva guide
└── .gitignore               # Ignored files (models, virtualenvs, node_modules)
```

---

## 🚀 Getting Started

### 1. Backend Setup (FastAPI + OpenCV + PyTorch)

Make sure you have Python 3.10+ installed.

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI backend server (port 8000)
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend documentation will be accessible at:
- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/status`

---

### 2. Frontend Setup (React + Vite)

Make sure you have Node.js 18+ installed.

```bash
# Navigate to the frontend directory
cd frontend

# Install Node dependencies
npm install

# Start Vite development server (port 5173)
npm run dev
```

Open your browser at `http://localhost:5173`.

---

## 🤝 Trained Model

The repository includes the trained `models/emotion_cnn.pth` checkpoint. The API loads this
checkpoint on startup. The final evaluation in `models/final_evaluation_metrics.json` reports
66.58% FER-2013 test accuracy for the deployed horizontal-flip test-time-augmentation inference
on all 7,178 test images. The earlier 66.24% result was the original single-view evaluation.

### FER-2013 data location

The training scripts expect `train/` and `test/` class folders under `Dataset/archive/` by
default. Alternatively, set `FER_DATASET_PATH` to the absolute path to the FER-2013 archive
before running the scripts. Keep the official `test/` directory separate; use the validation
split for model selection and evaluate the test set only once for the final reported model.

---

## 💡 Key Technical Features

1. **Prediction Smoothing (Exponential Moving Average)**:
   $$\text{Smoothed}_t = \alpha \cdot \text{Current}_t + (1 - \alpha) \cdot \text{Smoothed}_{t-1}$$
   Eliminates visual jitter and rapid flickering across sequential video frames.
2. **Face Bounding Box Overlay**:
   Canvas overlay dynamically maps detected OpenCV face coordinates `(x, y, w, h)` onto the live video feed.
3. **Prediction Logging & Export**:
   Maintains a real-time log of emotion predictions and provides one-click export to CSV.
4. **Graceful Handling of Edge Cases**:
   Clear visual indicators when no face is present or camera permission is disabled.
5. **Frame privacy**:
   The web application processes live frames for prediction and does not store them.

---

## Evaluation and submission readiness

The repository contains the implementation and generated artifacts for EDA, the NumPy
convolution-vs-PyTorch check, the VGG-style CNN, FER-2013 augmentation and square-root class
weights, feature-map visualisation, and the saved confusion matrix and final metrics.
The original single-view evaluation reported 66.24% test accuracy and 274.8 CNN-only CPU FPS.
The final flip-averaged inference reports 66.58% accuracy, 63.18% macro F1, and 66.24% weighted
F1 on the official 7,178-image test split. It measured 66.2 FPS for batch-1 two-pass model-only
CPU inference; the full batched test evaluation processed 77.6 images/s. Neither figure is an
end-to-end webcam FPS measurement.
The final confusion matrix highlights the remaining errors: 239 of 1,247 sad faces were
predicted neutral, and 219 of 1,024 fear faces were predicted sad. The square-root class weighting
gives the minority disgust class 70.27% recall, while its 43.09% precision shows the remaining
false-positive trade-off.

The current API averages predictions from the original and horizontally flipped face crop.
On the full validation split this measured 65.34% versus 64.94% for single-view inference.
The final held-out test evaluation of this validation-selected inference measured 66.58%.

The existing `models/model_comparison.json` is a legacy, short-run comparison and should not be
used as a full transfer-learning result. The corrected transfer experiment upscales the
grayscale images to 224×224, repeats them across three channels for ResNet-18 and EfficientNet-B0,
trains on the full training split, evaluates on the full validation split, and benchmarks model
latency on CPU. Run it on a GPU where available:

```bash
python notebooks/04_transfer_learning.py --epochs 10
```

Before final submission, the team still needs to provide the 15–25 page report, 10–12 slide
presentation, signed declaration, and a public app URL or 5–8 minute recorded demo. The report
should include the per-class confusion analysis, architecture/latency table, limitations,
contributions from both members, and honest test metrics for the final selected inference
pipeline. Transfer-learning results in `models/model_comparison.json` are short-run exploratory
figures, not full-data, fully trained comparison results.