import StickmanCanvas from "../components/StickmanCanvas";

export default function TextToSign() {
  return (
    <div className="max-w-6xl mx-auto px-8 py-16">

      {/* ✅ Center EVERYTHING */}
      <div className="flex flex-col items-center text-center space-y-6">

        <h1 className="text-4xl font-bold">
          Text → Sign Language
        </h1>

        <p className="text-gray-600 dark:text-gray-400">
          Type a supported word and view its sign animation.
        </p>

        <StickmanCanvas />

      </div>

    </div>
  );
}
