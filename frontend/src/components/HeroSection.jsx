import React from 'react';
import { Play, ArrowRight, Video, Navigation, Database } from 'lucide-react';

export default function HeroSection({ stats }) {
  return (
    <section style={{ padding: '4rem 0 3rem 0', textAlign: 'center' }}>
      <div className="container" style={{ maxWidth: '900px' }}>
        <div style={{ display: 'inline-flex', marginBottom: '1.25rem' }}>
          <div className="badge" style={{ padding: '0.35rem 0.85rem', borderColor: 'rgba(59, 130, 246, 0.4)' }}>
            <span style={{ color: 'var(--primary)' }}>★ Portfolio Engineering Release</span>
            <span style={{ color: 'var(--text-dim)' }}>|</span>
            <span style={{ color: 'var(--text-muted)' }}>Maintained by Priyam Thakur</span>
          </div>
        </div>

        <h1 style={{ fontSize: '2.75rem', lineHeight: 1.15, marginBottom: '1.25rem' }}>
          Smart Face Recognition & Autonomous Robot Tracking System
        </h1>

        <p style={{ fontSize: '1.1rem', color: 'var(--text-muted)', lineHeight: 1.6, marginBottom: '2rem' }}>
          An asynchronous computer vision application combining <strong>InsightFace Buffalo_L</strong> deep feature extraction,
          monocular distance estimation, temporal bounding-box smoothing, and a fail-safe target following policy.
        </p>

        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap', marginBottom: '3.5rem' }}>
          <a href="#demo" className="btn btn-primary" style={{ padding: '0.85rem 1.75rem', fontSize: '1rem' }}>
            <Play size={18} /> Launch Live Web Demo
          </a>
          <a href="#architecture" className="btn btn-outline" style={{ padding: '0.85rem 1.75rem', fontSize: '1rem' }}>
            Explore Architecture <ArrowRight size={18} />
          </a>
        </div>

        <div className="grid-3" style={{ textAlign: 'left' }}>
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: 'var(--primary)' }}><Video size={20} /></div>
              <h4 style={{ fontSize: '0.9rem' }}>Feature Backbone</h4>
            </div>
            <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
              ArcFace 512-D
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              RetinaFace detection + cosine metric matching
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: 'var(--success)' }}><Navigation size={20} /></div>
              <h4 style={{ fontSize: '0.9rem' }}>Range Policy</h4>
            </div>
            <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
              50 – 100 cm
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              Pinhole geometric triangle similarity
            </div>
          </div>

          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.5rem' }}>
              <div style={{ color: 'var(--warning)' }}><Database size={20} /></div>
              <h4 style={{ fontSize: '0.9rem' }}>Registered Gallery</h4>
            </div>
            <div className="mono" style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--text-main)' }}>
              {stats?.total_persons ?? 0} Identities
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
              {stats?.total_embeddings ?? 0} indexed embedding vectors
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}