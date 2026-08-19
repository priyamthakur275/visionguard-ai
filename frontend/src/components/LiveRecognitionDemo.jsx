import React, { useState, useRef, useEffect } from 'react';
import { Camera, Upload, AlertOctagon, CheckCircle } from 'lucide-react';
import RobotSimulatorCanvas from './RobotSimulatorCanvas.jsx';
import { apiUrl, wsUrl } from '../api.js';

export default function LiveRecognitionDemo({ persons }) {
  const [activeTab, setActiveTab] = useState('camera');
  const [isStreaming, setIsStreaming] = useState(false);
  const [selectedTarget, setSelectedTarget] = useState('');
  const [robotData, setRobotData] = useState({ command: 'STOP', reason: 'Idle', distance_cm: -1, offset_px: 0 });
  const [latencyMs, setLatencyMs] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);

  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const wsRef = useRef(null);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsStreaming(true);
          initWebSocket();
        };
      }
    } catch (err) {
      alert(`Camera access failed: ${err.message}. You can use the 'Image Upload' tab instead.`);
    }
  };

  const stopCamera = () => {
    setIsStreaming(false);
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop());
      videoRef.current.srcObject = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
    setRobotData({ command: 'STOP', reason: 'Camera offline', distance_cm: -1, offset_px: 0 });
  };

  const initWebSocket = () => {
    const wsEndpoint = wsUrl('/ws/live');
    const ws = new WebSocket(wsEndpoint);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) return;

        setLatencyMs(data.inference_time_ms);
        if (data.robot) setRobotData(data.robot);
        drawBoundingBoxes(data.faces);
      } catch (e) {
        console.error(e);
      }
    };

    const sendLoop = () => {
      if (!isStreaming || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (videoRef.current && captureCanvasRef.current) {
        const canvas = captureCanvasRef.current;
        const ctx = canvas.getContext('2d');
        canvas.width = 480;
        canvas.height = 360;
        ctx.drawImage(videoRef.current, 0, 0, 480, 360);

        const b64 = canvas.toDataURL('image/jpeg', 0.6);
        wsRef.current.send(JSON.stringify({
          image: b64,
          target_name: selectedTarget || null
        }));
      }
      setTimeout(sendLoop, 65);
    };

    ws.onopen = () => {
      sendLoop();
    };
  };

  const drawBoundingBoxes = (faces) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !videoRef.current) return;
    const ctx = canvas.getContext('2d');
    canvas.width = videoRef.current.videoWidth || 640;
    canvas.height = videoRef.current.videoHeight || 480;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!faces) return;

    const scaleX = canvas.width / 480;
    const scaleY = canvas.height / 360;

    faces.forEach((f) => {
      const [x1, y1, x2, y2] = f.bbox;
      const rx1 = x1 * scaleX;
      const ry1 = y1 * scaleY;
      const rw = (x2 - x1) * scaleX;
      const rh = (y2 - y1) * scaleY;

      ctx.strokeStyle = f.is_known ? '#10b981' : '#ef4444';
      ctx.lineWidth = 3;
      ctx.strokeRect(rx1, ry1, rw, rh);

      const label = `${f.name} (${(f.similarity * 100).toFixed(0)}%) | ${f.distance_cm > 0 ? f.distance_cm + 'cm' : '--'}`;
      ctx.font = 'bold 13px Inter, sans-serif';
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = f.is_known ? '#10b981' : '#ef4444';
      ctx.fillRect(rx1, Math.max(0, ry1 - 22), tw + 10, 22);

      ctx.fillStyle = '#000000';
      ctx.fillText(label, rx1 + 5, Math.max(15, ry1 - 6));
    });
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const base64Str = reader.result;
      try {
        const res = await fetch(apiUrl('/api/recognize'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_base64: base64Str,
            target_name: selectedTarget || null
          })
        });
        const data = await res.json();
        setUploadResult({
          imageSrc: base64Str,
          data
        });
        if (data.robot) setRobotData(data.robot);
      } catch (err) {
        alert('Recognition failed: ' + err.message);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleEmergencyStop = () => {
    setSelectedTarget('');
    setRobotData({ command: 'STOP', reason: 'EMERGENCY STOP TRIGGERED', distance_cm: -1, offset_px: 0 });
  };

  return (
    <section id="demo" style={{ padding: '2rem 0 4rem 0' }}>
      <div className="container">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
          <div>
            <h2 style={{ fontSize: '1.75rem', marginBottom: '0.25rem' }}>Interactive AI Viewport & Telemetry</h2>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
              Real-time deep inference and automated robot navigation decisions.
            </p>
          </div>

          <div style={{ display: 'flex', background: 'var(--bg-surface-elevated)', borderRadius: '8px', padding: '4px', border: '1px solid var(--border-subtle)' }}>
            <button
              className={`btn ${activeTab === 'camera' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => { setActiveTab('camera'); setUploadResult(null); }}
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem', border: 'none' }}
            >
              <Camera size={16} /> Live Webcam
            </button>
            <button
              className={`btn ${activeTab === 'upload' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => { setActiveTab('upload'); stopCamera(); }}
              style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem', border: 'none' }}
            >
              <Upload size={16} /> Photo Upload Fallback
            </button>
          </div>
        </div>

        <div className="grid-2" style={{ gridTemplateColumns: '1.35fr 1fr', alignItems: 'start' }}>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Follow Target:</span>
                <select
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  style={{ width: '170px', padding: '0.4rem 0.65rem', fontSize: '0.8rem' }}
                >
                  <option value="">(None - Lockout)</option>
                  {persons?.map(p => (
                    <option key={p.identity_id} value={p.name}>{p.name}</option>
                  ))}
                </select>
              </div>

              <button className="btn btn-danger" onClick={handleEmergencyStop} style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}>
                <AlertOctagon size={16} /> EMERGENCY STOP
              </button>
            </div>

            {activeTab === 'camera' ? (
              <div style={{
                position: 'relative',
                background: '#000',
                borderRadius: '8px',
                aspectRatio: '4/3',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden'
              }}>
                <video ref={videoRef} playsInline muted style={{ width: '100%', height: '100%', objectFit: 'cover', display: isStreaming ? 'block' : 'none' }} />
                <canvas ref={overlayCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
                <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

                {!isStreaming && (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    <Camera size={48} style={{ opacity: 0.4, marginBottom: '1rem' }} />
                    <p style={{ fontWeight: 600, color: 'var(--text-main)', marginBottom: '0.25rem' }}>Browser Camera Offline</p>
                    <p style={{ fontSize: '0.85rem', maxWidth: '300px' }}>
                      Click 'Start Stream' to capture frames for live facial recognition.
                    </p>
                  </div>
                )}
              </div>
            ) : (
              <div style={{
                border: '2px dashed var(--border-subtle)',
                borderRadius: '8px',
                aspectRatio: '4/3',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'rgba(0,0,0,0.2)',
                position: 'relative',
                overflow: 'hidden'
              }}>
                {uploadResult ? (
                  <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                    <img src={uploadResult.imageSrc} alt="Uploaded frame" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                  </div>
                ) : (
                  <label style={{ cursor: 'pointer', textAlign: 'center', padding: '2rem', width: '100%' }}>
                    <Upload size={40} style={{ color: 'var(--primary)', marginBottom: '0.75rem' }} />
                    <p style={{ fontWeight: 600 }}>Click to Select Photo for Inference</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>JPG, PNG, WEBP files</p>
                    <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
                  </label>
                )}
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '1rem' }}>
              {activeTab === 'camera' ? (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {!isStreaming ? (
                    <button className="btn btn-success" onClick={startCamera}>
                      ▶ Start Camera Stream
                    </button>
                  ) : (
                    <button className="btn btn-outline" onClick={stopCamera}>
                      ■ Stop Stream
                    </button>
                  )}
                </div>
              ) : (
                <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
                  <Upload size={16} /> Choose Another Image
                  <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
                </label>
              )}

              {latencyMs && (
                <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Inference Latency: <strong style={{ color: 'var(--text-main)' }}>{latencyMs} ms</strong>
                </span>
              )}
            </div>

            <div style={{
              marginTop: '1.25rem',
              padding: '0.75rem 1rem',
              background: 'rgba(59, 130, 246, 0.08)',
              borderRadius: '8px',
              border: '1px solid rgba(59, 130, 246, 0.2)',
              fontSize: '0.75rem',
              color: 'var(--text-muted)',
              display: 'flex',
              gap: '0.5rem'
            }}>
              <CheckCircle size={16} style={{ color: 'var(--primary)', flexShrink: 0, marginTop: '2px' }} />
              <span>
                <strong>Privacy Protected:</strong> Video frames are processed strictly in server RAM and immediately discarded. No photos are written to disk during live demo streaming.
              </span>
            </div>
          </div>

          <RobotSimulatorCanvas robotData={robotData} />
        </div>
      </div>
    </section>
  );
}