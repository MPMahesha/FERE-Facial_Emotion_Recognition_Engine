import React from 'react';
import { Activity, Server, Cpu, Sparkles } from 'lucide-react';

// ============================================================
// HEADER COMPONENT
// ============================================================

export default function Header({ isApiOnline, isMockMode, modelFile, fps }) {
  return (
    <header className="header-card">
      <div className="header-left">
        <div className="header-logo">
          🎭
        </div>
        <div className="header-title-box">
          <h1>Facial Emotion Recogniser</h1>
          <p className="header-subtitle">
            P9 MSc Data Science Project &bull; Real-Time Browser Emotion Engine
          </p>
        </div>
      </div>

      <div className="header-badges">
        {/* API Status Badge */}
        <div className={`badge ${isApiOnline ? 'badge-online' : 'badge-offline'}`}>
          <span className="status-dot"></span>
          <Server size={14} />
          {isApiOnline ? 'API Connected (FastAPI)' : 'API Offline'}
        </div>

        {/* Model Mode Badge */}
        <div className={`badge ${isMockMode ? 'badge-mock' : 'badge-model'}`}>
          {isMockMode ? <Sparkles size={14} /> : <Cpu size={14} />}
          {isMockMode ? 'Mock Prediction Engine' : `PyTorch CNN: ${modelFile || 'Active'}`}
        </div>

        {/* Inference FPS Counter */}
        {isApiOnline && (
          <div className="badge badge-model">
            <Activity size={14} />
            {fps ? `${fps} FPS` : 'Standby'}
          </div>
        )}
      </div>
    </header>
  );
}
