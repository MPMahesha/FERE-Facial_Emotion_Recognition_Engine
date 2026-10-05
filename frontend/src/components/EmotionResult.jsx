import React from 'react';
import { ScanFace } from 'lucide-react';

// ============================================================
// EMOTION VISUAL MAPPER
// ============================================================

const EMOTION_MAP = {
  angry: { emoji: '😠', label: 'Angry', color: '#ef4444' },
  disgust: { emoji: '🤢', label: 'Disgust', color: '#10b981' },
  fear: { emoji: '😨', label: 'Fear', color: '#a855f7' },
  happy: { emoji: '😄', label: 'Happy', color: '#f59e0b' },
  neutral: { emoji: '😐', label: 'Neutral', color: '#94a3b8' },
  sad: { emoji: '😢', label: 'Sad', color: '#3b82f6' },
  surprise: { emoji: '😲', label: 'Surprise', color: '#ec4899' },
};

// ============================================================
// EMOTION RESULT COMPONENT
// ============================================================

export default function EmotionResult({ primaryEmotion, confidence, hasFace, isSmoothingActive }) {
  if (!hasFace || !primaryEmotion) {
    return (
      <div className="emotion-card emotion-card-empty" role="status">
        <div className="emotion-mark">
          <ScanFace size={30} strokeWidth={1.7} />
        </div>
        <div className="emotion-info">
          <span className="emotion-label-title">Live prediction</span>
          <h2 className="emotion-name" style={{ color: '#94a3b8' }}>
            Waiting for a face
          </h2>
          <p className="emotion-confidence-text">
            Position your face in the camera view to begin analysis.
          </p>
        </div>
      </div>
    );
  }

  const emotionData = EMOTION_MAP[primaryEmotion.toLowerCase()] || {
    label: primaryEmotion,
    color: '#6366f1'
  };

  const percentScore = (confidence * 100).toFixed(1);

  return (
    <div className="emotion-card" style={{ borderColor: `${emotionData.color}40` }}>
      <div className="emotion-mark" style={{ color: emotionData.color }}>
        <ScanFace size={30} strokeWidth={1.7} />
      </div>

      <div className="emotion-info">
        <span className="emotion-label-title">
          Live prediction {isSmoothingActive && <span className="smooth-tag">SMOOTHED</span>}
        </span>
        <h2 className="emotion-name" style={{ color: emotionData.color }}>
          {emotionData.label}
        </h2>
        <p className="emotion-confidence-text">
          Confidence Score: <span className="emotion-confidence-val">{percentScore}%</span>
        </p>
      </div>
    </div>
  );
}
