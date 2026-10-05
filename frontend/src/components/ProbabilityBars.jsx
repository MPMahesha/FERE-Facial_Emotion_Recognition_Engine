import React from 'react';

// ============================================================
// EMOTION COLORS
// ============================================================

const EMOTIONS_CONFIG = [
  { key: 'happy', label: 'Happy', color: '#bd7725' },
  { key: 'neutral', label: 'Neutral', color: '#77828b' },
  { key: 'surprise', label: 'Surprise', color: '#a45174' },
  { key: 'sad', label: 'Sad', color: '#547eaa' },
  { key: 'fear', label: 'Fear', color: '#7562a0' },
  { key: 'angry', label: 'Angry', color: '#b6524b' },
  { key: 'disgust', label: 'Disgust', color: '#43836b' },
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
                <span className="prob-item-dot" style={{ backgroundColor: color }} />
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
