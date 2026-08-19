import React, { useState } from 'react';
import { X, Upload, Trash2 } from 'lucide-react';
import { apiUrl } from '../api.js';

export default function EnrollmentModal({ isOpen, onClose, onEnrolled, persons }) {
  const [name, setName] = useState('');
  const [consent, setConsent] = useState(false);
  const [files, setFiles] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState(null);

  if (!isOpen) return null;

  const handleFileChange = (e) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files);
      setFiles(prev => prev.concat(selected));
    }
  };

  const handleRemoveFile = (idx) => {
    setFiles(prev => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (files.length < 2) {
      setStatusMessage({ type: 'error', text: 'Please upload at least 2 clear facial photos.' });
      return;
    }

    setIsLoading(true);
    setStatusMessage(null);

    const formData = new FormData();
    formData.append('name', name.trim());
    formData.append('consent', consent);
    files.forEach(f => formData.append('images', f));

    try {
      const res = await fetch(apiUrl('/api/enroll'), {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setStatusMessage({ type: 'success', text: `Enrolled ${data.name} with ${data.images_enrolled} images.` });
        setName('');
        setFiles([]);
        setConsent(false);
        onEnrolled();
        setTimeout(() => {
          onClose();
        }, 1500);
      } else {
        setStatusMessage({ type: 'error', text: data.detail || 'Enrollment failed.' });
      }
    } catch (err) {
      setStatusMessage({ type: 'error', text: 'Network connection failed: ' + err.message });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeletePerson = async (pName) => {
    if (!confirm(`Delete ${pName} and all associated embeddings?`)) return;
    try {
      const res = await fetch(apiUrl(`/api/persons/${encodeURIComponent(pName)}`), { method: 'DELETE' });
      if (res.ok) {
        onEnrolled();
      }
    } catch (e) {
      alert('Delete failed: ' + e);
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.8)',
      backdropFilter: 'blur(8px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 100,
      padding: '1.5rem'
    }}>
      <div className="card-elevated" style={{ width: '100%', maxWidth: '650px', maxHeight: '90vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
          <h2 style={{ fontSize: '1.25rem' }}>Face Gallery & Identity Enrollment</h2>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        <div style={{ marginBottom: '1.5rem' }}>
          <h4 style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>Current Registered Gallery</h4>
          <div style={{ maxHeight: '120px', overflowY: 'auto', background: '#0a0d14', borderRadius: '8px', border: '1px solid var(--border-subtle)', padding: '0.5rem' }}>
            {persons && persons.length > 0 ? (
              persons.map(p => (
                <div key={p.identity_id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.4rem 0.5rem', fontSize: '0.8rem', borderBottom: '1px solid #1a2233' }}>
                  <div>
                    <strong>{p.name}</strong> <span style={{ color: 'var(--text-dim)' }}>({p.image_count} photos)</span>
                  </div>
                  <button onClick={() => handleDeletePerson(p.name)} style={{ background: 'transparent', border: 'none', color: 'var(--danger)', cursor: 'pointer' }}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            ) : (
              <div style={{ fontSize: '0.8rem', color: 'var(--text-dim)', textAlign: 'center', padding: '0.5rem' }}>No identities enrolled yet.</div>
            )}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Person Display Name *</label>
            <input
              type="text"
              placeholder="e.g. Priyam Thakur"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '0.35rem' }}>Upload Reference Photos (2–20 photos) *</label>
            <label style={{
              border: '2px dashed var(--border-subtle)',
              borderRadius: '8px',
              padding: '1.5rem',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              cursor: 'pointer',
              background: 'rgba(255,255,255,0.01)'
            }}>
              <Upload size={24} style={{ color: 'var(--primary)', marginBottom: '0.5rem' }} />
              <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Click to select reference photos</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>Each image must contain exactly one face</span>
              <input type="file" multiple accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
            </label>

            {files.length > 0 && (
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.75rem' }}>
                {files.map((f, idx) => (
                  <div key={idx} style={{ position: 'relative', width: '60px', height: '60px', borderRadius: '6px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
                    <img src={URL.createObjectURL(f)} alt="preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <button
                      type="button"
                      onClick={() => handleRemoveFile(idx)}
                      style={{ position: 'absolute', top: '2px', right: '2px', background: 'rgba(0,0,0,0.7)', border: 'none', color: '#fff', borderRadius: '50%', width: '16px', height: '16px', fontSize: '10px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start', marginBottom: '1.25rem' }}>
            <input
              type="checkbox"
              id="modalConsent"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              required
              style={{ width: 'auto', marginTop: '4px' }}
            />
            <label htmlFor="modalConsent" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              I confirm that consent has been obtained to process these photos to generate 512-D face recognition embeddings.
            </label>
          </div>

          {statusMessage && (
            <div style={{
              padding: '0.75rem',
              borderRadius: '6px',
              fontSize: '0.8rem',
              marginBottom: '1rem',
              background: statusMessage.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
              border: `1px solid ${statusMessage.type === 'success' ? 'var(--success)' : 'var(--danger)'}`,
              color: statusMessage.type === 'success' ? 'var(--success)' : 'var(--danger)'
            }}>
              {statusMessage.text}
            </div>
          )}

          <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
            <button type="button" className="btn btn-outline" onClick={onClose}>Cancel</button>
            <button type="submit" className="btn btn-primary" disabled={isLoading || files.length < 2 || !consent || !name.trim()}>
              {isLoading ? '⏳ Extracting ArcFace Vectors...' : '✅ Save & Enroll'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}