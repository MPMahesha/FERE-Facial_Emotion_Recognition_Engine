import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import Webcam from './components/Webcam';
import EmotionResult from './components/EmotionResult';
import ProbabilityBars from './components/ProbabilityBars';
import SpeechEmotion from './components/SpeechEmotion';
import { Sliders, History, Download, Trash2, ScanFace, AudioLines } from 'lucide-react';
import './styles/style.css';

const API_BASE_URL = 'http://localhost:8000';
const EMOTIONS_LIST = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

// ============================================================
// MAIN APPLICATION COMPONENT
// ============================================================

export default function App() {
  // Active analysis mode: 'face' or 'speech'
  const [activeTab, setActiveTab] = useState('face');

  // System & API State
  const [isApiOnline, setIsApiOnline] = useState(false);
  const [isMockMode, setIsMockMode] = useState(true);
  const [modelFile, setModelFile] = useState(null);
  const [fps, setFps] = useState(0);

  // Webcam & Inference State
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [faces, setFaces] = useState([]);
  const [primaryEmotion, setPrimaryEmotion] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [probabilities, setProbabilities] = useState({});
  const [hasFace, setHasFace] = useState(false);

  // Smoothing & Performance Controls
  const [enableSmoothing, setEnableSmoothing] = useState(true);
  const [smoothingWeight, setSmoothingWeight] = useState(0.4); // Exponential Moving Average Alpha
  const [captureInterval, setCaptureInterval] = useState(50); // ms capture cadence; actual FPS depends on inference time
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);

  // Prediction History Logs
  const [predictionLogs, setPredictionLogs] = useState([]);

  // Internal References for Smoothing & FPS Calculation
  const smoothedProbsRef = useRef([]);
  const lastFrameTimeRef = useRef(Date.now());
  const isRequestInFlight = useRef(false);

  // ============================================================
  // BACKEND HEALTH CHECK
  // ============================================================

  const checkApiStatus = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/status`);
      if (response.ok) {
        const data = await response.json();
        setIsApiOnline(true);
        setIsMockMode(data.is_mock);
        setModelFile(data.model_file);
      } else {
        setIsApiOnline(false);
      }
    } catch {
      setIsApiOnline(false);
    }
  }, []);

  useEffect(() => {
    checkApiStatus();
    const interval = setInterval(checkApiStatus, 4000);
    return () => clearInterval(interval);
  }, [checkApiStatus]);

  // ============================================================
  // PREDICTION SMOOTHING ENGINE (EXPONENTIAL MOVING AVERAGE)
  // ============================================================

  const applyPredictionSmoothing = (rawProbs, currentEmotion, faceIndex) => {
    const previousProbs = smoothedProbsRef.current[faceIndex];
    if (!enableSmoothing || !previousProbs) {
      smoothedProbsRef.current[faceIndex] = { ...rawProbs };
      return { ...rawProbs };
    }

    const previousEmotion = Object.entries(previousProbs)
      .reduce((top, entry) => entry[1] > top[1] ? entry : top)[0];
    const alpha = currentEmotion !== previousEmotion
      ? Math.max(smoothingWeight, 0.85)
      : smoothingWeight;
    const smoothed = {};
    EMOTIONS_LIST.forEach((emotion) => {
      const currentVal = rawProbs[emotion] || 0;
      const prevVal = previousProbs[emotion] || currentVal;
      // Exponential Moving Average: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
      smoothed[emotion] = parseFloat((alpha * currentVal + (1 - alpha) * prevVal).toFixed(4));
    });

    smoothedProbsRef.current[faceIndex] = smoothed;
    return smoothed;
  };

  // ============================================================
  // FRAME CAPTURE & INFERENCE HANDLER
  // ============================================================

  const handleCaptureFrame = useCallback(
    async (base64Image) => {
      // Prevent overlapping concurrent API requests
      if (isRequestInFlight.current || !isApiOnline) return;
      isRequestInFlight.current = true;

      try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: base64Image }),
        });

        if (!response.ok) {
          throw new Error(`Inference error HTTP ${response.status}`);
        }

        const data = await response.json();

        // Calculate real-time FPS
        const now = Date.now();
        const delta = (now - lastFrameTimeRef.current) / 1000;
        lastFrameTimeRef.current = now;
        if (delta > 0) {
          setFps(Math.round(1 / delta));
        }

        if (data.faces && data.faces.length > 0) {
          setHasFace(true);

          const smoothedFaces = data.faces.map((face, index) => {
            const finalProbs = applyPredictionSmoothing(
              face.probabilities,
              face.emotion,
              index
            );
            const [emotion, score] = Object.entries(finalProbs)
              .reduce((top, entry) => entry[1] > top[1] ? entry : top);
            return { ...face, emotion, confidence: score, probabilities: finalProbs };
          });
          const primaryFace = smoothedFaces[0];
          const maxEmotion = primaryFace.emotion;
          const maxScore = primaryFace.confidence;

          setPrimaryEmotion(maxEmotion);
          setConfidence(maxScore);
          setProbabilities(primaryFace.probabilities);
          setFaces(smoothedFaces);

          // Log prediction to history
          const newLog = {
            id: Date.now() + Math.random(),
            time: new Date().toLocaleTimeString(),
            emotion: maxEmotion,
            confidence: (maxScore * 100).toFixed(1),
            faceCount: data.face_count,
          };
          setPredictionLogs((prev) => [newLog, ...prev.slice(0, 49)]);
        } else {
          setHasFace(false);
          setFaces([]);
          smoothedProbsRef.current = [];
        }
      } catch (err) {
        console.error('Frame processing failed:', err);
      } finally {
        isRequestInFlight.current = false;
      }
    },
    [isApiOnline, enableSmoothing, smoothingWeight]
  );

  // ============================================================
  // LOG EXPORT & CLEAR HANDLERS
  // ============================================================

  const exportLogsAsCSV = () => {
    if (predictionLogs.length === 0) return;
    const headers = 'Time,Emotion,Confidence (%),Faces\n';
    const rows = predictionLogs
      .map((log) => `"${log.time}","${log.emotion}","${log.confidence}","${log.faceCount}"`)
      .join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emotion_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const clearLogs = () => {
    setPredictionLogs([]);
  };

  return (
    <div className="app-container">
      {/* Header with metadata & status indicators */}
      <Header
        isApiOnline={isApiOnline}
        isMockMode={isMockMode}
        modelFile={modelFile}
        fps={isCameraActive ? fps : 0}
      />

      {/* Analysis mode tabs */}
      <div className="tab-bar" role="tablist" aria-label="Analysis mode">
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'face'}
          className={`tab-button ${activeTab === 'face' ? 'is-active' : ''}`}
          onClick={() => setActiveTab('face')}
        >
          <ScanFace size={17} />
          Facial Emotion
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={activeTab === 'speech'}
          className={`tab-button ${activeTab === 'speech' ? 'is-active' : ''}`}
          onClick={() => setActiveTab('speech')}
        >
          <AudioLines size={17} />
          Voice &amp; Text Emotion
        </button>
      </div>

      {activeTab === 'speech' ? (
        <SpeechEmotion apiBaseUrl={API_BASE_URL} />
      ) : (
      /* Main Interactive Grid */
      <div className="main-grid">
        {/* Left Column: Live Webcam Stream */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <Webcam
            isActive={isCameraActive}
            onToggleCamera={() => setIsCameraActive((prev) => !prev)}
            faces={faces}
            onCaptureFrame={handleCaptureFrame}
            captureInterval={captureInterval}
            showBoundingBoxes={showBoundingBoxes}
          />

          {/* Engine Settings & Smoothing Controls */}
          <div className="glass-panel">
            <div className="panel-header">
              <div className="panel-title">
                <Sliders size={20} />
                <span>Engine Controls & Smoothing</span>
              </div>
            </div>

            <div className="settings-grid">
              {/* Smoothing Alpha Slider */}
              <div className="setting-card">
                <label>
                  <span>Smoothing Factor (EMA &alpha;)</span>
                  <span>{smoothingWeight}</span>
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={smoothingWeight}
                  onChange={(e) => setSmoothingWeight(parseFloat(e.target.value))}
                  className="range-slider"
                  disabled={!enableSmoothing}
                />
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  Lower = smoother transitions; Higher = faster reaction
                </span>
              </div>

              {/* Capture Rate Slider */}
              <div className="setting-card">
                <label>
                  <span>Inference Delay</span>
                  <span>{captureInterval} ms</span>
                </label>
                <input
                  type="range"
                  min="50"
                  max="1000"
                  step="50"
                  value={captureInterval}
                  onChange={(e) => setCaptureInterval(parseInt(e.target.value, 10))}
                  className="range-slider"
                />
                <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                  Capture cadence up to {(1000 / captureInterval).toFixed(1)} frames/sec; actual speed depends on inference time.
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.88rem' }}>
                <input
                  type="checkbox"
                  checked={enableSmoothing}
                  onChange={(e) => setEnableSmoothing(e.target.checked)}
                />
                Enable Temporal Prediction Smoothing
              </label>

              <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.88rem' }}>
                <input
                  type="checkbox"
                  checked={showBoundingBoxes}
                  onChange={(e) => setShowBoundingBoxes(e.target.checked)}
                />
                Show Face Bounding Boxes
              </label>
            </div>
          </div>
        </div>

        {/* Right Column: Emotion Results & Probabilities */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Dominant Detected Emotion */}
          <EmotionResult
            primaryEmotion={primaryEmotion}
            confidence={confidence}
            hasFace={hasFace && isCameraActive}
            isSmoothingActive={enableSmoothing}
          />

          {/* 7 Emotion Probabilities Breakdown */}
          <div className="glass-panel">
            <div className="panel-header">
              <div className="panel-title">
                <span>Emotion Probability Distribution</span>
              </div>
            </div>

            <ProbabilityBars
              probabilities={probabilities}
              topEmotion={hasFace && isCameraActive ? primaryEmotion : ''}
            />
          </div>

          {/* Prediction History Log */}
          <div className="glass-panel">
            <div className="panel-header">
              <div className="panel-title">
                <History size={18} />
                <span>Prediction History ({predictionLogs.length})</span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                  onClick={exportLogsAsCSV}
                  disabled={predictionLogs.length === 0}
                  title="Export to CSV"
                >
                  <Download size={14} /> Export
                </button>
                <button
                  type="button"
                  className="btn"
                  style={{ padding: '6px 12px', fontSize: '0.8rem' }}
                  onClick={clearLogs}
                  disabled={predictionLogs.length === 0}
                  title="Clear history"
                >
                  <Trash2 size={14} /> Clear
                </button>
              </div>
            </div>

            <div className="log-table-container">
              {predictionLogs.length > 0 ? (
                <table className="log-table">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Emotion</th>
                      <th>Confidence</th>
                      <th>Faces</th>
                    </tr>
                  </thead>
                  <tbody>
                    {predictionLogs.map((log) => (
                      <tr key={log.id}>
                        <td>{log.time}</td>
                        <td style={{ textTransform: 'capitalize', fontWeight: 600 }}>{log.emotion}</td>
                        <td>{log.confidence}%</td>
                        <td>{log.faceCount}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={{ padding: '24px', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
                  No predictions recorded yet. Start camera to begin logging.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      )}
    </div>
  );
}
