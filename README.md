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
│   └── .gitkeep             # Destination for Member A's trained .pth weights
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

## 🤝 Member A Handoff: How to Connect the Trained `.pth` Model

Member A trains the CNN on FER-2013 and exports the PyTorch state dictionary or model:

1. Save the trained model checkpoint to the `models/` folder:
   ```bash
   models/fer_model.pth
   ```
2. The `backend/model_loader.py` automatically detects any `.pth` / `.pt` file inside `models/` upon server startup and switches from **Mock Mode** to **PyTorch CNN Mode** immediately.
3. No changes to the frontend or API contract are required.

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