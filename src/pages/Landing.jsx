import { useState } from "react";

function PipelineCard() {
  const items = [
    "LIVE AI PIPELINE",
    "Webcam capture",
    "Hand landmark detection",
    "ML sign classification",
    "Instant speech output"
  ];

  const [active, setActive] = useState(null);

  return (
    <div className="bg-indigo-50 dark:bg-indigo-950 border dark:border-indigo-800 rounded-3xl p-10 shadow-sm">
      <div className="space-y-4">

        {items.map((text, i) => (
          <div
            key={i}
            onMouseEnter={() => setActive(i)}
            onMouseLeave={() => setActive(null)}
            className={`transition-all duration-150
              ${i === 0
                ? active === i
                  ? "text-indigo-700 dark:text-indigo-300 font-extrabold tracking-widest"
                  : "text-indigo-600 dark:text-indigo-400 font-bold tracking-widest"
                : active === i
                  ? "font-bold text-gray-900 dark:text-white"
                  : "font-normal text-gray-700 dark:text-gray-300"
              }
            `}
          >
            {text}
          </div>
        ))}

      </div>
    </div>
  );
}

export default function Landing({ setPage }) {
  return (
    <div className="max-w-6xl mx-auto px-8 py-16">

      {/* HERO */}
      <div className="grid md:grid-cols-2 gap-16 items-center">

        <div className="space-y-7">

          <div className="text-sm uppercase tracking-widest text-indigo-600 dark:text-indigo-400 font-bold">
            EqualVoice
          </div>

          <h1 className="text-5xl font-extrabold leading-tight text-gray-900 dark:text-white">
            Sign. Show.
            <br />
            <span className="underline decoration-indigo-500">
              Speak.
            </span>
          </h1>

          <p className="text-gray-600 dark:text-gray-300 text-lg leading-relaxed">
            Your hands already know the language.
            EqualVoice helps the world hear it.
          </p>

          <div className="flex gap-5 pt-3">

            <button
              onClick={() => setPage("interpreter")}
              className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 rounded-lg font-semibold transition"
            >
              Start Interpreting →
            </button>

            <button
              onClick={() => setPage("learning")}
              className="text-indigo-600 dark:text-indigo-400 font-semibold hover:underline"
            >
              Learn Signs First
            </button>
            <button
  onClick={() => setPage("textsign")}
  className="text-indigo-600 dark:text-indigo-400 font-semibold hover:underline"
>
  Text → Sign Language
</button>

          </div>

        </div>

        <PipelineCard />

      </div>

      {/* STORY BLOCK */}
      <div className="mt-28 grid md:grid-cols-2 gap-16 items-center">

        <div className="bg-gray-100 dark:bg-gray-800 border dark:border-gray-700 rounded-3xl p-10 space-y-5">

          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            Why this exists
          </div>

          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            Not everyone understands sign language —
            but everyone understands speech.
            EqualVoice bridges that gap using only
            your camera and AI hand tracking.
          </p>

        </div>

        <div className="space-y-7">

          <Feature  title="Real-time" text="Predictions update live from webcam" />
          <Feature  title="Learn + Practice" text="Built-in sign learning mode" />
          <Feature  title="Accessible" text="Voice + subtitles + mute controls" />

        </div>

      </div>

      {/* CTA */}
      <div className="mt-28 text-center space-y-6">

        <div className="text-3xl font-bold text-gray-900 dark:text-white">
          Try it with your hands — not a mouse.
        </div>

        <button
          onClick={() => setPage("interpreter")}
          className="bg-black dark:bg-white dark:text-black text-white px-8 py-3 rounded-xl font-semibold hover:opacity-90 transition"
        >
          Launch Interpreter
        </button>

      </div>

    </div>
  );
}

function Feature({ icon, title, text }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="text-2xl">{icon}</div>
      <div>
        <div className="font-semibold text-gray-900 dark:text-white">
          {title}
        </div>
        <div className="text-gray-600 dark:text-gray-400 text-sm">
          {text}
        </div>
      </div>
    </div>
  );
}
