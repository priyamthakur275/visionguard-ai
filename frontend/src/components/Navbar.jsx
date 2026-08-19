import React from 'react';
import { Shield, Github } from 'lucide-react';

export default function Navbar({ health, onOpenEnroll }) {
  return (
    <header style={{
      borderBottom: '1px solid var(--border-subtle)',
      background: 'rgba(19, 23, 31, 0.85)',
      backdropFilter: 'blur(12px)',
      position: 'sticky',
      top: 0,
      zIndex: 50
    }}>
      <div className="container" style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '70px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{
            width: '38px',
            height: '38px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#fff'
          }}>
            <Shield size={22} />
          </div>
          <div>
            <div style={{ fontWeight: 800, fontSize: '1.15rem', letterSpacing: '-0.02em' }}>
              VisionGuard <span style={{ color: 'var(--primary)' }}>AI</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Face Recognition & Tracking System
            </div>
          </div>
        </div>

        <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <a href="#demo" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.875rem', fontWeight: 500 }}>Live Demo</a>
          <a href="#architecture" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.875rem', fontWeight: 500 }}>Architecture</a>
          <a href="#safety" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.875rem', fontWeight: 500 }}>Safety Invariants</a>
          <a href="#tech" style={{ color: 'var(--text-muted)', textDecoration: 'none', fontSize: '0.875rem', fontWeight: 500 }}>Tech Stack</a>
        </nav>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div className="badge">
            <span className={`badge-dot ${health?.status === 'healthy' ? 'online' : 'offline'}`} />
            <span className="mono">API: {health?.status === 'healthy' ? 'Online' : 'Checking'}</span>
          </div>

          <button className="btn btn-outline" onClick={onOpenEnroll} style={{ padding: '0.5rem 0.875rem' }}>
            ➕ Enroll Face
          </button>

          <a
            href="https://github.com/priyamthakur275/visionguard-ai"
            target="_blank"
            rel="noreferrer"
            className="btn btn-primary"
            style={{ padding: '0.5rem 0.875rem' }}
          >
            <Github size={16} /> GitHub
          </a>
        </div>
      </div>
    </header>
  );
}