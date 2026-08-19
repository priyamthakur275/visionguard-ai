// API base URL configured via environment variable (e.g. https://visionguard-api.onrender.com)
// If empty, defaults to same-origin (useful for local Vite proxy or Docker/Render single-origin)
const rawBase = import.meta.env.VITE_API_URL || '';
export const API_BASE = rawBase.replace(/\/+$/, '');

export const apiUrl = (path) => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${cleanPath}`;
};

export const wsUrl = (path) => {
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  if (API_BASE) {
    const wsProto = API_BASE.startsWith('https:') ? 'wss:' : 'ws:';
    const host = API_BASE.replace(/^https?:\/\//, '');
    return `${wsProto}//${host}${cleanPath}`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}${cleanPath}`;
};