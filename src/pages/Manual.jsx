import { useEffect, useMemo, useRef, useState } from "react";

const SIGNS = [
  { word: "HELLO", src: "/videos/hello.mp4" },
  { word: "THANK YOU", src: "/videos/thankyou.mp4" },
  { word: "YES", src: "/videos/yes.mp4" },
  { word: "NO", src: "/videos/no.mp4" },
  { word: "HELP", src: "/videos/help.mp4" },
  { word: "WATER", src: "/videos/water.mp4" },
  { word: "EAT", src: "/videos/eat.mp4" },
  { word: "DRINK", src: "/videos/drink.mp4" },
  { word: "HOME", src: "/videos/home.mp4" },
  { word: "SCHOOL", src: "/videos/school.mp4" },
  { word: "WORK", src: "/videos/work.mp4" },
  { word: "BOOK", src: "/videos/book.mp4" },
  { word: "READ", src: "/videos/read.mp4" },
  { word: "WRITE", src: "/videos/write.mp4" },
  { word: "GO", src: "/videos/go.mp4" },
  { word: "COME", src: "/videos/come.mp4" },
  { word: "STOP", src: "/videos/stop.mp4" },
  { word: "WAIT", src: "/videos/wait.mp4" },
  { word: "FINISH", src: "/videos/finish.mp4" },
  { word: "DAY", src: "/videos/day.mp4" },
  { word: "FATHER", src: "/videos/father.mp4" },
  { word: "MOTHER", src: "/videos/mother.mp4" },
  { word: "NIGHT", src: "/videos/night.mp4" },
  { word: "GOODBYE", src: "/videos/goodbye.mp4" },
  { word: "SAD", src: "/videos/sad.mp4" },
  { word: "HAPPY", src: "/videos/happy.mp4" },
  { word: "HOW", src: "/videos/how.mp4" },
  { word: "WHY", src: "/videos/why.mp4" },
  { word: "WHO", src: "/videos/who.mp4" },
  { word: "WHEN", src: "/videos/when.mp4" },
  { word: "WHERE", src: "/videos/where.mp4" },
  { word: "WHAT", src: "/videos/what.mp4" },
  { word: "NAME", src: "/videos/name.mp4" },
  { word: "LOVE", src: "/videos/love.mp4" },
  { word: "GOOD", src: "/videos/good.mp4" },
  { word: "BAD", src: "/videos/bad.mp4" },
  { word: "SORRY", src: "/videos/sorry.mp4" },
  { word: "FRIEND", src: "/videos/friend.mp4" },
  { word: "PLEASE", src: "/videos/please.mp4" },
  { word: "DRAW", src: "/videos/draw.mp4" },
];

export default function Manual() {
  const [query, setQuery] = useState("");
  const [openIndex, setOpenIndex] = useState(null);
  const startX = useRef(0);

  const filtered = useMemo(
    () =>
      SIGNS.filter(s =>
        s.word.toLowerCase().includes(query.toLowerCase())
      ),
    [query]
  );

  const open = i => setOpenIndex(i);
  const close = () => setOpenIndex(null);

  const next = () =>
    setOpenIndex(i => (i + 1) % filtered.length);

  const prev = () =>
    setOpenIndex(i => (i - 1 + filtered.length) % filtered.length);

  // ✅ keyboard accessibility
  useEffect(() => {
    const fn = e => {
      if (openIndex === null) return;
      if (e.key === "Escape") close();
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
    };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [openIndex, filtered.length]);

  return (
    <section className="max-w-6xl mx-auto px-6 py-10">

      {/* Header */}
      <h1 className="text-2xl font-semibold mb-6">
        ASL Sign Manual
      </h1>

      {/* Toolbar */}
      <div className="flex flex-wrap gap-4 mb-8 items-center">
        <input
          className="
            border border-line dark:border-slate-700
            bg-white dark:bg-slate-900
            rounded-lg px-3 py-2 w-64
            focus:outline-none focus:ring-2 focus:ring-brand
          "
          placeholder="Search sign..."
          onChange={e => setQuery(e.target.value)}
        />

        <div className="text-sm text-gray-600 dark:text-gray-400">
          {filtered.length} signs
        </div>
      </div>

      {/* Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-5">
        {filtered.map((s, i) => (
          <button
            key={s.word}
            onClick={() => open(i)}
            className="
              text-left
              border border-line dark:border-slate-800
              bg-white dark:bg-slate-900
              rounded-xl p-3
              hover:shadow-md hover:-translate-y-1
              transition
              focus:outline-none focus:ring-2 focus:ring-brand
            "
          >
            <video
              src={s.src}
              autoPlay
              muted
              loop
              playsInline
              className="
                w-full h-36 object-contain
                bg-black rounded-lg
              "
            />

            <div className="mt-2 text-sm font-semibold">
              {s.word}
            </div>
          </button>
        ))}
      </div>

      {/* Modal Viewer */}
      {openIndex !== null && (
        <div
          className="fixed inset-0 bg-black/90 flex items-center justify-center z-50"
          onClick={close}
          onTouchStart={e => (startX.current = e.changedTouches[0].screenX)}
          onTouchEnd={e => {
            const dx = e.changedTouches[0].screenX - startX.current;
            if (dx > 60) prev();
            if (dx < -60) next();
          }}
        >
          <div
            className="relative max-w-3xl w-full px-6"
            onClick={e => e.stopPropagation()}
          >
            <video
              src={filtered[openIndex].src}
              controls
              autoPlay
              loop
              className="w-full rounded-lg bg-black"
            />

            <div className="text-white text-xl font-semibold mt-4 text-center">
              {filtered[openIndex].word}
            </div>

            {/* Controls */}
            <button
              onClick={prev}
              className="absolute left-2 top-1/2 text-white text-3xl"
            >
              ❮
            </button>

            <button
              onClick={next}
              className="absolute right-2 top-1/2 text-white text-3xl"
            >
              ❯
            </button>

            <button
              onClick={close}
              className="absolute top-2 right-2 text-white text-2xl"
            >
              ✕
            </button>
          </div>
        </div>
      )}

    </section>
  );
}
