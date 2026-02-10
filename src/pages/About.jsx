export default function About() {
  return (
    <div className="max-w-5xl mx-auto px-8 py-12 space-y-12">

      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold text-gray-900 dark:text-white">
          About EqualVoice
        </h1>
        <p className="text-gray-600 dark:text-gray-300 text-lg">
          Bridging communication gaps using AI-powered sign language tools
        </p>
      </div>

      {/* Mission */}
      <section className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-2xl p-8 shadow-sm">
        <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
          Our Mission
        </h2>
        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
          EqualVoice is built to make communication more inclusive for the
          deaf and hard-of-hearing community. We combine computer vision,
          machine learning, and real-time interpretation tools to convert
          sign language into readable and audible output.
        </p>
      </section>

      {/* Feature Blocks */}
      <section className="grid md:grid-cols-3 gap-6">

        {[
          ["", "Live Interpreter", "Real-time sign to speech"],
          ["", "Learning Mode", "Guided sign learning"],
          ["", "Voice Output", "Instant spoken results"]
        ].map(([icon, title, desc], i) => (
          <div
            key={i}
            className="bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl p-6 shadow-sm"
          >
            <div className="text-2xl mb-2">{icon}</div>
            <h3 className="font-semibold text-lg text-gray-900 dark:text-white">
              {title}
            </h3>
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              {desc}
            </p>
          </div>
        ))}

      </section>

      {/* Tech */}
      <section className="bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-2xl p-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
          Technology Used
        </h2>

        <ul className="grid md:grid-cols-2 gap-3 text-gray-700 dark:text-gray-300">
          <li>React + Tailwind</li>
          <li>FastAPI</li>
          <li>PyTorch</li>
          <li>MediaPipe</li>
          <li>OpenCV</li>
          <li>Web Speech API</li>
        </ul>
      </section>

      <div className="text-center text-sm text-gray-500 dark:text-gray-400">
        Built for inclusive communication.
      </div>

    </div>
  );
}
