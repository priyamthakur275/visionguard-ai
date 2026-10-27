import { useRef } from "react";
import { Cpu, Download, Github, Lock, MonitorSmartphone, Trash2, Upload, Usb } from "lucide-react";
import { useVision } from "../state/VisionContext.jsx";

const REPO = "https://github.com/priyamthakur275/visionguard-ai";

export function TopBar() {
  return (
    <header className="topbar">
      <a href="#top" className="brand" aria-label="VisionGuard home">
        <img src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" width="26" height="26" />
        VisionGuard
      </a>
      <nav aria-label="Sections">
        <a href="#robot">Robot</a>
        <a href="#space">Face map</a>
        <a href="#how">How it works</a>
        <a href="#hardware">Hardware</a>
        <a href={REPO} target="_blank" rel="noreferrer" className="gh"><Github size={16} /> Code</a>
      </nav>
    </header>
  );
}

export function People() {
  const { gallery, forget, exportData, importData, forgetAll } = useVision();
  const names = Object.keys(gallery);
  const fileRef = useRef(null);
  return (
    <div className="people">
      <div className="people-head">
        <h3>Remembered on this device</h3>
        <p><Lock size={13} /> Stored in your browser only, as numbers. Clearing site data removes them.</p>
      </div>
      {names.length === 0 ? (
        <p className="people-empty">No one yet. Name a face in the viewer to teach it.</p>
      ) : (
        <ul>
          {names.map((n) => (
            <li key={n}>
              <span className="person-initial">{n.slice(0, 1).toUpperCase()}</span>
              <span className="person-name">{n}</span>
              <span className="person-count">{gallery[n].length} faceprint{gallery[n].length === 1 ? "" : "s"}</span>
              <button className="icon-btn" onClick={() => forget(n)} aria-label={`Forget ${n}`}><Trash2 size={15} /></button>
            </li>
          ))}
        </ul>
      )}
      <div className="people-actions">
        <button className="link-btn" onClick={exportData} disabled={!names.length}><Download size={14} /> Export</button>
        <button className="link-btn" onClick={() => fileRef.current.click()}><Upload size={14} /> Import</button>
        {names.length > 0 && <button className="link-btn danger" onClick={forgetAll}><Trash2 size={14} /> Forget everyone</button>}
        <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={(e) => { if (e.target.files?.[0]) importData(e.target.files[0]); e.target.value = ""; }} />
      </div>
    </div>
  );
}

export function Hardware() {
  return (
    <section id="hardware" className="section">
      <div className="section-head">
        <h2>Same brain, three bodies</h2>
        <p>One Python core (detection, faceprints, tracking, distance, safety rules) runs in three places.</p>
      </div>
      <div className="bodies">
        <article>
          <MonitorSmartphone size={22} />
          <h3>This website</h3>
          <p>React + onnxruntime-web. The compact InsightFace pack (buffalo_sc) runs in a Web Worker. Static hosting, no server, no uploads.</p>
        </article>
        <article>
          <Cpu size={22} />
          <h3>Desktop app</h3>
          <p>Python, OpenCV and the full InsightFace pack (buffalo_l, ResNet-50). Enrol people, run live recognition from a USB webcam, packaged as a Windows .exe.</p>
        </article>
        <article>
          <Usb size={22} />
          <h3>Robot</h3>
          <p>Commands go over serial to an Arduino. The firmware stops the motors if no command arrives for 750 ms, so a crashed laptop can't leave it driving.</p>
        </article>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="footer">
      <p>
        VisionGuard by Priyam Thakur · <a href={REPO} target="_blank" rel="noreferrer">Source on GitHub</a>
      </p>
      <p className="fineprint">
        Models: InsightFace buffalo_sc (non-commercial research licence). Sample photo: NASA, public domain.
        Distance is an estimate from an average face width.
      </p>
    </footer>
  );
}
