import { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";

export default function Learning() {

  const signs = [
    "YES","THANKYOU","NO","HELP","PLEASE",
    "HELLO","STOP","NAME","SORRY",
    "UP","DOWN","FOOD"
  ];
const signVideos = {
  YES: "/videos/yes.mp4",
  THANKYOU: "/videos/thankyou.mp4",
  NO: "/videos/no.mp4",
  HELP: "/videos/help.mp4",
  PLEASE: "/videos/please.mp4",
  HELLO: "/videos/hello.mp4",
  STOP: "/videos/stop.mp4",
  NAME: "/videos/name.mp4",
  SORRY: "/videos/sorry.mp4",
  UP: "/videos/up.mp4",
  DOWN: "/videos/down.mp4",
  FOOD: "/videos/food.mp4"
};

  const webcamRef = useRef(null);

  const [video, setVideo] = useState(null);
  const [current, setCurrent] = useState("");

  const [practiceTarget, setPracticeTarget] = useState(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState(null);

  const [correctCount, setCorrectCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  // ✅ NEW — sound toggle
  const [soundOn, setSoundOn] = useState(true);

  // 🔊 voice (respects toggle)
  const speak = (text) => {
    if (!soundOn) return;
    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
  };

  // ✅ SEND FRAME TO BACKEND
  const sendFrame = async () => {
    if (!webcamRef.current) return;

    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    const blob = await (await fetch(imageSrc)).blob();
    const formData = new FormData();
    formData.append("file", blob, "frame.jpg");

    try {
      const r = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
      });

      if (!r.ok) {
        console.log("server reject", r.status);
        return;
      }

      const data = await r.json();
      if (!data.prediction) return;

      let label = data.prediction;

      // map STOP → UNKNOWN
      if (label === "STOP") {
        label = "UNKNOWN";
      }

      setResult({
        topLabel: label,
        topConf: 1.0
      });

      speak(label);

      if (practiceTarget) {
          const isCorrect =
        label === practiceTarget ||
        (practiceTarget === "STOP" && label === "UNKNOWN");

        setFeedback(isCorrect ? "correct" : "wrong");

        setTotalCount(t => t + 1);
        if (isCorrect) setCorrectCount(c => c + 1);
      }

    } catch (e) {
      console.log("predict error", e);
    }
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(sendFrame, 900);
    return () => clearInterval(id);
  }, [running, practiceTarget, soundOn]);

  return (
    <div className="p-10 max-w-7xl mx-auto">

      <h1 className="text-4xl font-bold text-center mb-12">
        Communication Signs Learning
      </h1>

      {/* ✅ SOUND TOGGLE BUTTON */}
      <div className="flex justify-center mb-6">
        <button
          onClick={() => {
            setSoundOn(s => !s);
            window.speechSynthesis.cancel();
          }}
          className="border px-4 py-2 rounded-lg hover:bg-gray-100 transition"
        >
          {soundOn ? "🔊 Sound ON" : "🔇 Sound OFF"}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-8">
        {signs.map((s, i) => (
          <div key={i}
            className="bg-white border rounded-2xl p-6 shadow-sm flex flex-col items-center gap-4">

            <div className="text-2xl font-extrabold">{s}</div>

            <div className="flex gap-2 w-full">

              <button
                onClick={() => {
                  setVideo(signVideos[s]);

                  setCurrent(s);
                }}
                className="flex-1 bg-indigo-600 text-white py-2 rounded-md">
                View
              </button>

              <button
                onClick={() => {
                  setPracticeTarget(s);
                  setRunning(true);
                  setFeedback(null);
                  setResult(null);
                  setCorrectCount(0);
                  setTotalCount(0);
                }}
                className="flex-1 bg-gray-200 py-2 rounded-md">
                Practice
              </button>

            </div>
          </div>
        ))}
      </div>

      {practiceTarget && (
        <div className="mt-12 flex flex-col items-center gap-6">

          <h2 className="text-2xl font-semibold">
            Practice: {practiceTarget}
          </h2>

          {!result && running &&
            <div className="text-blue-600">
              🔍 Reading your sign...
            </div>
          }

          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            className="rounded-xl shadow"
          />

          {result &&
            <div className="text-lg text-center">
              Detected: <b>{result.topLabel}</b>
            </div>
          }

          {feedback === "correct" &&
            <div className="text-green-600 text-xl font-bold">
              ✅ Correct — great job!
            </div>
          }

          {feedback === "wrong" &&
            <div className="text-red-600 text-xl font-bold">
              ❌ Try again
            </div>
          }

          {totalCount > 0 && (
            <div className="text-lg font-semibold">
              Accuracy: {Math.round((correctCount / totalCount) * 100)}%
              <div className="text-sm text-gray-500">
                {correctCount} correct / {totalCount} attempts
              </div>
            </div>
          )}

          <button
            onClick={() => {
              setRunning(false);
              setPracticeTarget(null);
              window.speechSynthesis.cancel();
            }}
            className="px-6 py-3 border rounded">
            Stop Practice
          </button>

        </div>
      )}

      {video && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center">
          <div className="bg-white rounded-xl p-6 w-[90%] max-w-3xl relative">
            <button
              onClick={() => setVideo(null)}
              className="absolute top-3 right-3 bg-red-500 text-white px-3 py-1 rounded">
              X
            </button>

            <h2 className="text-xl font-bold mb-4">
              Learning: {current}
            </h2>

            <div className="aspect-video">
              <iframe src={video} className="w-full h-full rounded-lg" allowFullScreen />
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
