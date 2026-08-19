import React from 'react';

export default function TechStackSection() {
  const stack = [
    { cat: "Deep Learning", items: ["InsightFace Buffalo_L", "RetinaFace", "ArcFace 512-D", "ONNX Runtime"] },
    { cat: "Core & Vision", items: ["Python 3.10 / 3.11", "OpenCV (cv2)", "NumPy", "Scipy"] },
    { cat: "Web Services", items: ["FastAPI (ASGI)", "Uvicorn", "WebSockets", "Pydantic v2"] },
    { cat: "Frontend & UI", items: ["React 18", "Vite", "CustomTkinter (Desktop)", "HTML5 Canvas"] },
    { cat: "Microcontroller", items: ["Arduino C++", "PySerial UART", "750ms Watchdog", "ACK Protocol"] },
    { cat: "QA & CI/CD", items: ["Pytest (41 Tests)", "GitHub Actions", "Docker", "Vercel / Render"] },
  ];

  return (
    <section id="tech" style={{ padding: '4rem 0', background: '#080a0f', borderTop: '1px solid var(--border-subtle)' }}>
      <div className="container">
        <div style={{ textAlign: 'center', maxWidth: '750px', margin: '0 auto 3rem auto' }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>Technology & Tooling Stack</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Production libraries and frameworks utilized across desktop and web layers.
          </p>
        </div>

        <div className="grid-3">
          {stack.map((group, idx) => (
            <div key={idx} className="card">
              <h4 style={{ fontSize: '0.9rem', color: 'var(--primary)', marginBottom: '0.75rem' }}>{group.cat}</h4>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                {group.items.map((item, i) => (
                  <span key={i} className="badge mono" style={{ background: 'rgba(255,255,255,0.03)', fontSize: '0.75rem' }}>
                    {item}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}