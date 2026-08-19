import React from 'react';
import { Activity } from 'lucide-react';

export default function SystemStatus({ health, stats }) {
  return (
    <section style={{ padding: '0 0 3rem 0' }}>
      <div className="container">
        <div className="card-elevated">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
            <Activity size={18} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>System Telemetry & Live Diagnostics</h3>
          </div>

          <div className="grid-4">
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Backend State</div>
              <div className="mono" style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '0.2rem' }}>
                <span className="badge-dot online" style={{ display: 'inline-block', marginRight: '6px' }} />
                {health?.status ? 'Operational' : 'Connecting...'}
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>ArcFace Threshold</div>
              <div className="mono" style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '0.2rem' }}>
                θ = {stats?.similarity_threshold ?? 0.45} (Cosine)
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Focal Length Geometry</div>
              <div className="mono" style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '0.2rem' }}>
                f = 615.0 px (W=14cm)
              </div>
            </div>

            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Safety Dead Zone</div>
              <div className="mono" style={{ fontSize: '0.95rem', fontWeight: 600, marginTop: '0.2rem' }}>
                ±{stats?.robot_settings?.dead_zone_px ?? 60} px Center
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}