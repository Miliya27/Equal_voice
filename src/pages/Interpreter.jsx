import { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";

export default function Interpreter() {
  const webcamRef = useRef(null);

  const [prediction, setPrediction] = useState("");
  const [running, setRunning] = useState(false);
  const [muted, setMuted] = useState(false);
  const lastSpokenRef = useRef("");

  // 🔊 speak helper
  function speak(text) {
    if (!text || muted) return;

    if (lastSpokenRef.current === text) return;
    lastSpokenRef.current = text;

    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(u);
  }

  async function sendFrame() {
    if (!webcamRef.current) return;

    const shot = webcamRef.current.getScreenshot();
    if (!shot) return;

    try {
      const res = await fetch(shot);
      const blob = await res.blob();

      const form = new FormData();
      form.append("file", blob, "frame.jpg");

      const r = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: form,
      });

      const data = await r.json();

      let word = data.prediction || "";

      // ✅ MAP STOP → UNCHANGE
      if (word === "STOP") {
        word = "UNKNOWN";
      }

      setPrediction(word);
      speak(word);

    } catch (err) {
      console.log("API error:", err);
    }
  }

  // run loop
  useEffect(() => {
    if (!running) return;
    const id = setInterval(sendFrame, 1000);
    return () => clearInterval(id);
  }, [running, muted]);

  // stop speech when muted
  useEffect(() => {
    if (muted) speechSynthesis.cancel();
  }, [muted]);

  return (
    <div className="flex flex-col items-center gap-6 p-10">

      <h2 className="text-3xl font-bold">
        Live Sign Interpreter
      </h2>

      <div className="relative rounded-2xl overflow-hidden shadow-lg">

        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={560}
          className="bg-black"
          videoConstraints={{ facingMode: "user" }}
        />

        {/* subtitle */}
        <div className="absolute bottom-4 left-0 right-0 text-center">
          <span className="bg-black/70 text-white px-6 py-2 rounded-lg text-2xl font-semibold">
            {prediction || "—"}
          </span>
        </div>

      </div>

      {/* CONTROL BUTTON ROW */}
      <div className="flex gap-3">

        {/* Start / Stop */}
        <button
          onClick={() => setRunning(!running)}
          className={`px-5 py-2 text-sm font-semibold rounded-md transition active:scale-95
            ${running
              ? "bg-red-600 hover:bg-red-700 text-white"
              : "bg-indigo-600 hover:bg-indigo-700 text-white"
            }`}
        >
          {running ? "Stop" : "Start"}
        </button>

        {/* Mute Toggle */}
        <button
          onClick={() => setMuted(!muted)}
          className="px-4 py-2 text-sm rounded-md border border-gray-300 hover:bg-gray-100 transition active:scale-95"
        >
          {muted ? " 🔇" : " 🔊 "}
        </button>

      </div>

    </div>
  );
}
