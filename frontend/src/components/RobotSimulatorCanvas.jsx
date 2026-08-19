import React, { useEffect, useRef } from 'react';

export default function RobotSimulatorCanvas({ robotData }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationId;

    const render = () => {
      ctx.clearRect(0, 0, 320, 320);

      // Radar Grid
      ctx.strokeStyle = '#1a2233';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 320; i += 32) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 320); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(320, i); ctx.stroke();
      }

      const cx = 160, cy = 260;

      // Safe Distance Arc Bands
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
      ctx.beginPath(); ctx.arc(cx, cy, 60, Math.PI, 0); ctx.stroke();

      ctx.strokeStyle = 'rgba(16, 185, 129, 0.5)';
      ctx.beginPath(); ctx.arc(cx, cy, 110, Math.PI, 0); ctx.stroke();

      ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
      ctx.beginPath(); ctx.arc(cx, cy, 160, Math.PI, 0); ctx.stroke();

      ctx.fillStyle = '#64748b';
      ctx.font = '10px JetBrains Mono';
      ctx.fillText('50cm', cx - 18, cy - 65);
      ctx.fillText('100cm', cx - 18, cy - 115);
      ctx.fillText('150cm', cx - 18, cy - 165);

      // Robot Chassis
      ctx.fillStyle = '#1e293b';
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(cx - 28, cy - 22, 56, 44, 6);
      ctx.fill();
      ctx.stroke();

      // Wheels
      const cmd = robotData?.command || 'STOP';
      ctx.fillStyle = cmd === 'LEFT' ? '#3b82f6' : '#0f172a';
      ctx.fillRect(cx - 36, cy - 20, 8, 40);

      ctx.fillStyle = cmd === 'RIGHT' ? '#3b82f6' : '#0f172a';
      ctx.fillRect(cx + 28, cy - 20, 8, 40);

      // Heading Arrow
      ctx.fillStyle = '#3b82f6';
      ctx.beginPath();
      ctx.moveTo(cx, cy - 28);
      ctx.lineTo(cx - 8, cy - 14);
      ctx.lineTo(cx + 8, cy - 14);
      ctx.closePath();
      ctx.fill();

      // Laser Tracking Ray & Target Dot
      if (robotData && robotData.distance_cm > 0) {
        const rayLen = Math.min(220, Math.max(40, robotData.distance_cm * 1.3));
        const angleOffset = ((robotData.offset_px || 0) / 480) * 0.7;
        const tx = cx + Math.sin(angleOffset) * rayLen;
        const ty = cy - Math.cos(angleOffset) * rayLen;

        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(cx, cy - 24);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = 'rgba(16, 185, 129, 0.2)';
        ctx.beginPath();
        ctx.arc(tx, ty, 14, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#10b981';
        ctx.beginPath();
        ctx.arc(tx, ty, 6, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter';
        ctx.fillText(`${robotData.distance_cm.toFixed(0)}cm`, tx + 10, ty + 4);
      }

      animationId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationId);
  }, [robotData]);

  const cmd = robotData?.command || 'STOP';
  const getCmdColor = () => {
    switch (cmd) {
      case 'FORWARD': return 'var(--success)';
      case 'BACKWARD': return 'var(--warning)';
      case 'LEFT':
      case 'RIGHT': return 'var(--primary)';
      default: return 'var(--danger)';
    }
  };

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 600 }}>2D Robot Telemetry Simulation</h3>
        <span className="badge mono" style={{ color: getCmdColor(), borderColor: getCmdColor() }}>
          {cmd}
        </span>
      </div>

      <div style={{
        background: '#07090e',
        borderRadius: '8px',
        border: '1px solid var(--border-subtle)',
        display: 'flex',
        justifyContent: 'center',
        overflow: 'hidden'
      }}>
        <canvas ref={canvasRef} width={320} height={320} style={{ maxWidth: '100%' }} />
      </div>

      <div style={{
        background: '#0a0e17',
        border: '1px solid var(--border-subtle)',
        borderRadius: '8px',
        padding: '0.85rem 1rem',
        fontSize: '0.8rem',
        lineHeight: 1.6,
        color: 'var(--text-muted)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Navigation Command:</span>
          <strong className="mono" style={{ color: getCmdColor() }}>{cmd}</strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Target Distance:</span>
          <strong className="mono" style={{ color: 'var(--text-main)' }}>
            {robotData?.distance_cm > 0 ? `${robotData.distance_cm.toFixed(1)} cm` : '--'}
          </strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Center Offset:</span>
          <strong className="mono" style={{ color: 'var(--text-main)' }}>
            {robotData?.offset_px !== undefined ? `${robotData.offset_px} px` : '0 px'}
          </strong>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.4rem', marginTop: '0.4rem' }}>
          <span>Policy Reason:</span>
          <span style={{ color: 'var(--text-main)', textAlign: 'right' }}>
            {robotData?.reason || 'No active target'}
          </span>
        </div>
      </div>
    </div>
  );
}