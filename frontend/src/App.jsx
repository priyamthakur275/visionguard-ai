import { ShieldCheck } from "lucide-react";
import { VisionProvider } from "./state/VisionContext.jsx";
import Viewfinder from "./components/Viewfinder.jsx";
import RobotSection from "./components/RobotSection.jsx";
import HowItWorks from "./components/HowItWorks.jsx";
import FaceSpace from "./components/FaceSpace.jsx";
import { Footer, Hardware, People, TopBar } from "./components/Sections.jsx";

export default function App() {
  return (
    <VisionProvider>
      <TopBar />
      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <h1>
              A camera that knows <span className="nowrap">who it's</span> following.
            </h1>
            <p className="lede">
              VisionGuard finds every face, recognises the people you've named, estimates how far away they are, and
              steers a robot towards one of them.
            </p>
            <p className="privacy">
              <ShieldCheck size={18} />
              <span>
                <b>Runs entirely in this tab.</b> Your camera feed and photos never leave your device.
              </span>
            </p>
          </div>
          <Viewfinder />
          <div className="hero-people">
            <People />
          </div>
        </section>
        <RobotSection />
        <FaceSpace />
        <HowItWorks />
        <Hardware />
      </main>
      <Footer />
    </VisionProvider>
  );
}
