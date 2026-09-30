import React from 'react';

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
      <div className="emotion-card">
        <div className="emotion-emoji">👤</div>
        <div className="emotion-info">
          <span className="emotion-label-title">Status</span>
          <h2 className="emotion-name" style={{ color: '#94a3b8' }}>
            No Face Detected
          </h2>
          <p className="emotion-confidence-text">
            Align your face within the camera frame for real-time recognition.
          </p>
        </div>
      </div>
    );
  }

  const emotionData = EMOTION_MAP[primaryEmotion.toLowerCase()] || {
    emoji: '🎭',
    label: primaryEmotion,
    color: '#6366f1'
  };

  const percentScore = (confidence * 100).toFixed(1);

  return (
    <div className="emotion-card" style={{ borderColor: `${emotionData.color}40` }}>
      {/* Background glow matching the emotion's tone */}
      <div
        className="emotion-card-glow"
        style={{ backgroundColor: emotionData.color }}
      ></div>

      <div className="emotion-emoji">
        {emotionData.emoji}
      </div>

      <div className="emotion-info">
        <span className="emotion-label-title">
          Detected Emotion {isSmoothingActive && '(Smoothed)'}
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
