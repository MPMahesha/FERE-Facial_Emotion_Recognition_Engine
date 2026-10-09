import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  Mic,
  MicOff,
  Square,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  Loader2,
  RotateCcw,
  Keyboard,
  Radio,
} from 'lucide-react';

// ============================================================
// TEXT EMOTION VISUALS
// NOTE: these six classes come from the DAIR.AI text dataset and
// are DIFFERENT from the seven FER-2013 facial classes.
// ============================================================

const TEXT_EMOTIONS = [
  { key: 'joy', label: 'Joy', emoji: '😄', color: '#bd7725' },
  { key: 'sadness', label: 'Sadness', emoji: '😢', color: '#547eaa' },
  { key: 'love', label: 'Love', emoji: '❤️', color: '#c0556e' },
  { key: 'anger', label: 'Anger', emoji: '😠', color: '#b6524b' },
  { key: 'fear', label: 'Fear', emoji: '😨', color: '#7562a0' },
  { key: 'surprise', label: 'Surprise', emoji: '😲', color: '#a45174' },
];

const getEmotionVisual = (key) =>
  TEXT_EMOTIONS.find((e) => e.key === key) || {
    label: key,
    emoji: '🧠',
    color: '#18796e',
  };

// ============================================================
// SPEECH & TEXT EMOTION COMPONENT
// Uses the browser Web Speech API for speech-to-text, then sends
// the transcript to the local FastAPI /predict-text endpoint.
// ============================================================

export default function SpeechEmotion({ apiBaseUrl = 'http://localhost:8000' }) {
  const SpeechRecognition =
    typeof window !== 'undefined'
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null;
  const speechSupported = Boolean(SpeechRecognition);

  // status: idle | recording | processing | success | error
  const [status, setStatus] = useState('idle');
  const [transcript, setTranscript] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [result, setResult] = useState(null);

  const recognitionRef = useRef(null);
  const shouldListenRef = useRef(false);
  const committedRef = useRef('');
  const interimRef = useRef('');
  const isAnalyzingRef = useRef(false);

  // Stop recognition cleanly if the component unmounts.
  useEffect(
    () => () => {
      shouldListenRef.current = false;
      if (recognitionRef.current) {
        try {
          recognitionRef.current.abort();
        } catch {
          /* ignore */
        }
      }
    },
    []
  );

  // ----------------------------------------------------------
  // Recording controls
  // ----------------------------------------------------------
  const startRecording = useCallback(() => {
    if (!speechSupported) return;
    if (status === 'recording' || isAnalyzingRef.current) return;

    setErrorMessage('');
    setResult(null);
    setStatus('recording');

    // Continue appending to any text the user already has in the box.
    committedRef.current = transcript.trim() ? transcript.trim() + ' ' : '';
    interimRef.current = '';
    shouldListenRef.current = true;

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.maxAlternatives = 1;
    recognitionRef.current = recognition;

    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const res = event.results[i];
        if (res.isFinal) {
          committedRef.current += res[0].transcript + ' ';
        } else {
          interim += res[0].transcript;
        }
      }
      interimRef.current = interim;
      setTranscript((committedRef.current + interim).replace(/\s+/g, ' ').trimStart());
    };

    recognition.onerror = (event) => {
      const code = event.error;
      if (code === 'no-speech') {
        // Non-fatal: user simply has not spoken yet.
        setErrorMessage('No speech detected yet - keep speaking into the microphone.');
        return;
      }

      const fatal = ['not-allowed', 'service-not-allowed', 'audio-capture', 'network'];
      if (fatal.includes(code)) {
        shouldListenRef.current = false;
        setStatus('error');
      }

      if (code === 'not-allowed' || code === 'service-not-allowed') {
        setErrorMessage(
          'Microphone permission was denied. Allow microphone access in your browser settings, then press record again.'
        );
      } else if (code === 'audio-capture') {
        setErrorMessage('No microphone was found. Please connect a microphone and try again.');
      } else if (code === 'network') {
        setErrorMessage(
          'Speech recognition network error. Browser speech recognition relies on an online service - check your internet connection.'
        );
      } else {
        setErrorMessage(`Speech recognition error: ${code}.`);
      }
    };

    recognition.onend = () => {
      // Flush any interim words that never became final, so words are not lost
      // when a session ends or restarts.
      if (interimRef.current.trim()) {
        committedRef.current += interimRef.current.trim() + ' ';
        interimRef.current = '';
        setTranscript(committedRef.current.replace(/\s+/g, ' ').trimStart());
      }

      // Chrome may end a "continuous" session after silence. Restart while the
      // user is still recording, but delay it slightly: calling start() too
      // quickly after end() can throw InvalidStateError and silently kill the
      // capture (which would leave the transcript stuck on the first phrase).
      if (shouldListenRef.current) {
        setTimeout(() => {
          if (!shouldListenRef.current) return;
          try {
            recognition.start();
          } catch {
            /* start() may throw if already starting; safe to ignore */
          }
        }, 300);
      } else if (recognitionRef.current === recognition) {
        setStatus((prev) => (prev === 'recording' ? 'idle' : prev));
      }
    };

    try {
      recognition.start();
    } catch (err) {
      shouldListenRef.current = false;
      setStatus('error');
      setErrorMessage(`Could not start speech recognition: ${err.message}`);
    }
  }, [SpeechRecognition, speechSupported, status, transcript]);

  const stopRecording = useCallback(() => {
    shouldListenRef.current = false;
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        /* ignore */
      }
    }
    setStatus((prev) => (prev === 'recording' ? 'idle' : prev));
  }, []);

  const resetAll = useCallback(() => {
    shouldListenRef.current = false;
    if (recognitionRef.current) {
      try {
        recognitionRef.current.abort();
      } catch {
        /* ignore */
      }
    }
    committedRef.current = '';
    interimRef.current = '';
    setTranscript('');
    setResult(null);
    setErrorMessage('');
    setStatus('idle');
  }, []);

  // ----------------------------------------------------------
  // Analyze the transcript with the backend NLP model
  // ----------------------------------------------------------
  const analyzeTranscript = useCallback(async () => {
    const text = transcript.trim();

    if (!text) {
      setStatus('error');
      setErrorMessage('The transcript is empty. Record your voice or type a sentence first.');
      return;
    }
    if (isAnalyzingRef.current) return;

    isAnalyzingRef.current = true;
    setStatus('processing');
    setErrorMessage('');
    setResult(null);

    try {
      const response = await fetch(`${apiBaseUrl}/predict-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = data && data.detail ? data.detail : `Request failed (HTTP ${response.status}).`;
        throw new Error(detail);
      }

      setResult(data);
      setStatus('success');
    } catch (err) {
      setStatus('error');
      setErrorMessage(
        err.message === 'Failed to fetch'
          ? 'Could not reach the backend API. Make sure the FastAPI server is running on ' + apiBaseUrl + '.'
          : err.message
      );
    } finally {
      isAnalyzingRef.current = false;
    }
  }, [apiBaseUrl, transcript]);

  const isRecording = status === 'recording';
  const isProcessing = status === 'processing';

  return (
    <div className="speech-layout">
      {/* ---------------- Recording & transcript ---------------- */}
      <div className="glass-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Radio size={20} />
            <span>Voice &amp; Text Emotion Recognition</span>
          </div>
          <span className="badge badge-model">
            <Sparkles size={14} />
            NLP: TF-IDF + Logistic Regression
          </span>
        </div>

        <p className="speech-intro">
          Speak into your microphone and the transcript appears below. You can edit it, then
          analyse the emotion expressed by the words. This uses a <strong>different</strong> model
          and a <strong>different</strong> set of six emotions from the facial analysis section.
        </p>

        {/* Microphone control */}
        <div className="mic-control">
          <button
            type="button"
            className={`mic-button ${isRecording ? 'is-recording' : ''}`}
            onClick={isRecording ? stopRecording : startRecording}
            disabled={!speechSupported || isProcessing}
            aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            title={isRecording ? 'Stop recording' : 'Start recording'}
          >
            {isRecording ? <Square size={26} /> : <Mic size={30} />}
          </button>

          <div className="mic-state">
            {!speechSupported && (
              <span className="mic-state-title" style={{ color: '#a3473f' }}>
                <MicOff size={16} /> Speech recognition not supported
              </span>
            )}
            {speechSupported && isRecording && (
              <span className="mic-state-title">Listening…</span>
            )}
            {speechSupported && !isRecording && (
              <span className="mic-state-title">
                {status === 'idle' || status === 'success' || status === 'error'
                  ? 'Press to start recording'
                  : 'Working…'}
              </span>
            )}
            <span className="mic-state-hint">
              {speechSupported
                ? 'Uses the browser Web Speech API (an online speech service).'
                : 'No speech API found - type your sentence manually below.'}
            </span>
          </div>
        </div>

        {/* Transcript box */}
        <div className="transcript-block">
          <label htmlFor="speech-transcript" className="transcript-label">
            <Keyboard size={15} /> Recognised transcript (editable)
          </label>
          <textarea
            id="speech-transcript"
            className="transcript-area"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder={
              speechSupported
                ? 'Your speech will appear here. You can also type or edit the text directly…'
                : 'Type the sentence you want to analyse…'
            }
            rows={4}
            maxLength={1000}
          />
          <div className="transcript-meta">
            <span>{transcript.trim().length} / 1000 characters</span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="speech-actions">
          <button
            type="button"
            className="btn btn-primary"
            onClick={analyzeTranscript}
            disabled={isProcessing || !transcript.trim()}
          >
            {isProcessing ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />}
            {isProcessing ? 'Analysing…' : 'Analyse Emotion'}
          </button>

          {isRecording && (
            <button type="button" className="btn btn-danger" onClick={stopRecording}>
              <Square size={15} /> Stop Recording
            </button>
          )}

          <button
            type="button"
            className="btn"
            onClick={resetAll}
          >
            <RotateCcw size={15} /> Retry / Clear
          </button>
        </div>

        {/* Status messages */}
        {status === 'error' && errorMessage && (
          <div className="speech-alert speech-alert-error" role="alert">
            <AlertCircle size={16} />
            <span>{errorMessage}</span>
          </div>
        )}

        {status === 'success' && (
          <div className="speech-alert speech-alert-success" role="status">
            <CheckCircle2 size={16} />
            <span>Transcript analysed successfully.</span>
          </div>
        )}

        <p className="privacy-note">
          Audio is handled by your browser's speech service and is not stored by this app.
          Only the editable transcript is sent to the local backend for classification.
        </p>
      </div>

      {/* ---------------- Results ---------------- */}
      <div className="glass-panel">
        <div className="panel-header">
          <div className="panel-title">
            <Sparkles size={18} />
            <span>Text Emotion Result</span>
          </div>
        </div>

        {result ? (
          <>
            <div
              className="emotion-card"
              style={{ borderColor: `${getEmotionVisual(result.emotion).color}40` }}
            >
              <div
                className="emotion-mark"
                style={{ color: getEmotionVisual(result.emotion).color, fontSize: '1.9rem' }}
              >
                {getEmotionVisual(result.emotion).emoji}
              </div>
              <div className="emotion-info">
                <span className="emotion-label-title">Predicted from text</span>
                <h2
                  className="emotion-name"
                  style={{ color: getEmotionVisual(result.emotion).color }}
                >
                  {getEmotionVisual(result.emotion).label}
                </h2>
                <p className="emotion-confidence-text">
                  Confidence:{' '}
                  <span className="emotion-confidence-val">
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </p>
              </div>
            </div>

            {result.analyzed_text && (
              <p className="analyzed-text">
                <span className="analyzed-text-label">Analysed text</span>
                “{result.analyzed_text}”
              </p>
            )}

            <div className="probability-list">
              {TEXT_EMOTIONS.map(({ key, label, color }) => {
                const prob = result.probabilities?.[key] || 0;
                const percent = Math.min(Math.max(prob * 100, 0), 100).toFixed(1);
                const isTop = result.emotion === key;
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
                          boxShadow: isTop ? `0 0 10px ${color}80` : 'none',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="speech-empty" role="status">
            <div className="emotion-mark">
              <Mic size={28} strokeWidth={1.7} />
            </div>
            <p>No analysis yet.</p>
            <p className="speech-empty-hint">
              Record or type a sentence, then press <strong>Analyse Emotion</strong> to see the
              predicted emotion and probability scores.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
