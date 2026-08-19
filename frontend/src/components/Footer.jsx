import React from 'react';
import { Shield, Github } from 'lucide-react';

export default function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--border-subtle)',
      background: 'var(--bg-surface)',
      padding: '3rem 0 2rem 0',
      fontSize: '0.85rem',
      color: 'var(--text-muted)'
    }}>
      <div className="container">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem', marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <div style={{ width: '28px', height: '28px', borderRadius: '6px', background: 'var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fff' }}>
              <Shield size={16} />
            </div>
            <strong style={{ color: 'var(--text-main)', fontSize: '1rem' }}>VisionGuard AI</strong>
          </div>

          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <a href="https://github.com/priyamthakur275/visionguard-ai" target="_blank" rel="noreferrer" style={{ color: 'var(--text-muted)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <Github size={16} /> GitHub Repository
            </a>
            <a href="https://github.com/priyamthakur275/visionguard-ai/blob/main/LICENSE" target="_blank" rel="noreferrer" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>
              MIT License
            </a>
          </div>
        </div>

        <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem', display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
          <div>
            Created, hardened, and maintained by <strong>Priyam Thakur</strong>.
          </div>
          <div>
            Built with InsightFace, OpenCV, FastAPI, and React.
          </div>
        </div>
      </div>
    </footer>
  );
}