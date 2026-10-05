import React, { useRef, useEffect, useCallback } from 'react';
import ReactWebcam from 'react-webcam';
import { Camera, CameraOff, Video, VideoOff, RefreshCw } from 'lucide-react';

// ============================================================
// WEBCAM STREAM & BOUNDING BOX OVERLAY COMPONENT
// ============================================================

export default function Webcam({
  isActive,
  onToggleCamera,
  faces = [],
  onCaptureFrame,
  captureInterval = 250,
  showBoundingBoxes = true
}) {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [facingMode, setFacingMode] = React.useState('user');

  // Video capture dimensions
  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: facingMode
  };

  // Periodic frame capture loop
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      if (webcamRef.current) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          onCaptureFrame(imageSrc);
        }
      }
    }, captureInterval);

    return () => clearInterval(interval);
  }, [isActive, captureInterval, onCaptureFrame]);

  // Draw face bounding boxes on the overlay canvas
  const drawBoundingBoxes = useCallback(() => {
    const canvas = canvasRef.current;
    const video = webcamRef.current?.video;

    if (!canvas || !video) return;

    const ctx = canvas.getContext('2d');
    const displayWidth = video.videoWidth || 640;
    const displayHeight = video.videoHeight || 480;

    canvas.width = displayWidth;
    canvas.height = displayHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!showBoundingBoxes || !faces || faces.length === 0) {
      return;
    }

    faces.forEach((face) => {
      const { x, y, width, height } = face.box;
      const emotion = face.emotion || 'Unknown';
      const conf = ((face.confidence || 0) * 100).toFixed(0);

      // Draw bounding box
      ctx.strokeStyle = '#1b8174';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(x, y, width, height, 8);
      ctx.stroke();

      // Draw label background tag
      const labelText = `${emotion.toUpperCase()} ${conf}%`;
      ctx.font = 'bold 14px "Plus Jakarta Sans", sans-serif';
      const textWidth = ctx.measureText(labelText).width;
      const tagHeight = 24;

      ctx.fillStyle = '#1b8174';
      ctx.beginPath();
      ctx.roundRect(x, Math.max(0, y - tagHeight - 4), textWidth + 16, tagHeight, 4);
      ctx.fill();

      // Draw label text
      ctx.fillStyle = '#ffffff';
      ctx.fillText(labelText, x + 8, Math.max(16, y - 8));
    });
  }, [faces, showBoundingBoxes]);

  useEffect(() => {
    drawBoundingBoxes();
  }, [drawBoundingBoxes]);

  const toggleFacingMode = () => {
    setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'));
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Camera size={20} />
          <span>Live Camera Feed</span>
        </div>
        <div className="webcam-controls">
          <button
            type="button"
            className={`btn ${isActive ? 'btn-danger' : 'btn-primary'}`}
            onClick={onToggleCamera}
          >
            {isActive ? <CameraOff size={16} /> : <Camera size={16} />}
            {isActive ? 'Stop Camera' : 'Start Camera'}
          </button>

          {isActive && (
            <button
              type="button"
              className="btn"
              onClick={toggleFacingMode}
              title="Flip Camera"
            >
              <RefreshCw size={16} />
            </button>
          )}
        </div>
      </div>

      <div className="webcam-container">
        {isActive ? (
          <>
            <ReactWebcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              videoConstraints={videoConstraints}
              className="webcam-feed"
              mirrored={facingMode === 'user'}
            />
            <canvas ref={canvasRef} className="webcam-canvas" />
          </>
        ) : (
          <div className="webcam-offline-placeholder">
            <VideoOff size={48} />
            <p>Camera is currently turned off.</p>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Click <strong>Start Camera</strong> above to begin real-time emotion recognition.
            </p>
          </div>
        )}
      </div>
      <p className="privacy-note">
        Camera frames are processed for live predictions only and are never stored.
      </p>
    </div>
  );
}
