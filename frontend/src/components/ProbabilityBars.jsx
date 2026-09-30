import React from 'react';

// ============================================================
// EMOTION COLORS & EMOJIS
// ============================================================

const EMOTIONS_CONFIG = [
  { key: 'happy', label: 'Happy', emoji: '😄', color: '#f59e0b' },
  { key: 'neutral', label: 'Neutral', emoji: '😐', color: '#94a3b8' },
  { key: 'surprise', label: 'Surprise', emoji: '😲', color: '#ec4899' },
  { key: 'sad', label: 'Sad', emoji: '😢', color: '#3b82f6' },
  { key: 'fear', label: 'Fear', emoji: '😨', color: '#a855f7' },
  { key: 'angry', label: 'Angry', emoji: '😠', color: '#ef4444' },
  { key: 'disgust', label: 'Disgust', emoji: '🤢', color: '#10b981' },
];

// ============================================================
// PROBABILITY BARS COMPONENT
// ============================================================

export default function ProbabilityBars({ probabilities = {}, topEmotion = '' }) {
  return (
    <div className="probability-list">
      {EMOTIONS_CONFIG.map(({ key, label, emoji, color }) => {
        const rawProb = probabilities[key] || 0;
        const percent = Math.min(Math.max((rawProb * 100), 0), 100).toFixed(1);
        const isTop = topEmotion && topEmotion.toLowerCase() === key;

        return (
          <div key={key} className={`prob-item ${isTop ? 'is-top' : ''}`}>
            <div className="prob-item-header">
              <span className="prob-item-label">
                <span>{emoji}</span>
                <span>{label}</span>
              </span>
              <span className="prob-item-val" style={{ color: isTop ? color : undefined }}>
                {percent}%
              </span>
            </div>

            <div className="prob-bar-track">
              <div
                className="prob-bar-fill"
                style={{
                  width: `${percent}%`,
                  backgroundColor: color,
                  boxShadow: isTop ? `0 0 10px ${color}80` : 'none'
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
