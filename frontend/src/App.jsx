import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar.jsx';
import HeroSection from './components/HeroSection.jsx';
import LiveRecognitionDemo from './components/LiveRecognitionDemo.jsx';
import SystemStatus from './components/SystemStatus.jsx';
import EnrollmentModal from './components/EnrollmentModal.jsx';
import ArchitectureSection from './components/ArchitectureSection.jsx';
import SafetySection from './components/SafetySection.jsx';
import TechStackSection from './components/TechStackSection.jsx';
import Footer from './components/Footer.jsx';
import { apiUrl } from './api.js';

export default function App() {
  const [health, setHealth] = useState(null);
  const [stats, setStats] = useState(null);
  const [persons, setPersons] = useState([]);
  const [isEnrollModalOpen, setIsEnrollModalOpen] = useState(false);

  const fetchStatusAndPersons = async () => {
    try {
      const [hRes, sRes, pRes] = await Promise.all([
        fetch(apiUrl('/health')),
        fetch(apiUrl('/api/stats')),
        fetch(apiUrl('/api/persons')),
      ]);

      if (hRes.ok) setHealth(await hRes.json());
      if (sRes.ok) setStats(await sRes.json());
      if (pRes.ok) {
        const pData = await pRes.json();
        setPersons(pData.persons || []);
      }
    } catch (e) {
      console.warn('Backend offline or initializing:', e.message);
    }
  };

  useEffect(() => {
    fetchStatusAndPersons();
    const interval = setInterval(fetchStatusAndPersons, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar health={health} onOpenEnroll={() => setIsEnrollModalOpen(true)} />
      <HeroSection stats={stats} />
      <LiveRecognitionDemo persons={persons} />
      <SystemStatus health={health} stats={stats} />
      <ArchitectureSection />
      <SafetySection />
      <TechStackSection />
      <Footer />

      <EnrollmentModal
        isOpen={isEnrollModalOpen}
        onClose={() => setIsEnrollModalOpen(false)}
        onEnrolled={fetchStatusAndPersons}
        persons={persons}
      />
    </div>
  );
}