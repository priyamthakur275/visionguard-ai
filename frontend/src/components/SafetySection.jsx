import React from 'react';
import { Lock, AlertTriangle, Radio, Activity, StopCircle, Clock } from 'lucide-react';

export default function SafetySection() {
  const invariants = [
    { icon: <Lock size={20} color="var(--danger)" />, title: "Unknown Lockout", desc: "Unknown faces are strictly barred from issuing movement commands under all conditions." },
    { icon: <AlertTriangle size={20} color="var(--warning)" />, title: "Target Loss Halt", desc: "Immediate transition to STOP if the designated target identity leaves the field of view." },
    { icon: <StopCircle size={20} color="var(--danger)" />, title: "Dedicated E-STOP", desc: "One-click hardware and software Emergency Stop overrides all navigation states immediately." },
    { icon: <Radio size={20} color="var(--primary)" />, title: "Stream Failure Monitor", desc: "Automatic safe halt if 10 consecutive video frame captures fail or camera is disconnected." },
    { icon: <Activity size={20} color="var(--primary)" />, title: "Worker Crash Isolation", desc: "Unhandled inference exceptions trigger safe STOP and isolate the main application loop." },
    { icon: <Clock size={20} color="var(--warning)" />, title: "750ms Watchdog Timer", desc: "Arduino firmware watchdog halts all motors if no valid command arrives within 750 milliseconds." },
  ];

  return (
    <section id="safety" style={{ padding: '4rem 0', borderTop: '1px solid var(--border-subtle)' }}>
      <div className="container">
        <div style={{ textAlign: 'center', maxWidth: '750px', margin: '0 auto 3rem auto' }}>
          <h2 style={{ fontSize: '1.75rem', marginBottom: '0.5rem' }}>Fail-Safe Safety Architecture</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Seven deterministic safety invariants enforced across software and microcontroller firmware.
          </p>
        </div>

        <div className="grid-3">
          {invariants.map((inv, idx) => (
            <div key={idx} className="card" style={{ display: 'flex', gap: '1rem', alignItems: 'flex-start' }}>
              <div style={{ padding: '0.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                {inv.icon}
              </div>
              <div>
                <h4 style={{ fontSize: '0.95rem', marginBottom: '0.25rem' }}>{inv.title}</h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{inv.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}