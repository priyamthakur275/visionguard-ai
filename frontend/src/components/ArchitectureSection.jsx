import React from 'react';

export default function ArchitectureSection() {
  const steps = [
    { title: "1. Input Stream", desc: "640x480 BGR webcam frame capture with frame backpressure drops." },
    { title: "2. RetinaFace Detect", desc: "Locates faces, outputs (x1,y1,x2,y2) bounding boxes and 5 facial landmarks." },
    { title: "3. ArcFace 512-D", desc: "Extracts deep 512-dimensional feature vector, mapped onto unit hypersphere (L2 normalized)." },
    { title: "4. Cosine Matcher", desc: "Vectorized batch dot product against stacked gallery matrix. Threshold θ = 0.45." },
    { title: "5. Centroid Tracker", desc: "Nearest-neighbor spatial tracker paired with rolling majority-vote label smoothing." },
    { title: "6. Distance Estimator", desc: "Calculates d = (known_width * focal_length) / face_px_width pinhole geometry." },
    { title: "7. Safety Policy", desc: "Evaluates horizontal dead zone (±60px) and distance bands (50-100cm) for discrete commands." },
    { title: "8. Motor / Canvas", desc: "Line-delimited ASCII command dispatch over UART serial with 750ms watchdog timer." },
  ];

  return (
    <section id="architecture" style={{ padding: '4rem 0', background: '#080a0f', borderTop: '1px solid var(--border-subtle)' }}>
      <div className="container">
        <div style={{ textAlign: 'center', maxWidth: '750px', margin: '0 auto 3rem auto' }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>System Pipeline & Architecture</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            A decoupled asynchronous inference pipeline ensuring UI responsiveness and reliable real-time tracking.
          </p>
        </div>

        <div className="grid-4" style={{ gap: '1.25rem' }}>
          {steps.map((s, idx) => (
            <div key={idx} className="card" style={{ background: 'var(--bg-surface)' }}>
              <div className="mono" style={{ fontSize: '0.75rem', color: 'var(--primary)', fontWeight: 700, marginBottom: '0.35rem' }}>
                STAGE 0{idx + 1}
              </div>
              <h4 style={{ fontSize: '0.95rem', marginBottom: '0.4rem' }}>{s.title}</h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}