import { useEffect, useState } from "react";

export default function Navbar({ setPage }) {
  const [dark, setDark] = useState(false);

  // apply theme class to html root
  useEffect(() => {
    const root = document.documentElement;
    if (dark) root.classList.add("dark");
    else root.classList.remove("dark");
  }, [dark]);

  return (
    <div className="flex justify-between items-center px-8 py-4 border-b bg-white dark:bg-gray-900 dark:text-white">

      {/* Brand */}
      <button
        onClick={() => setPage("home")}
        className="font-bold text-xl hover:opacity-70 transition"
      >
        EqualVoice
      </button>

      <div className="flex gap-6 items-center">

        <button onClick={() => setPage("interpreter")}>Interpreter</button>
        <button onClick={() => setPage("textsign")}>
  Text → Sign
</button>
        <button onClick={() => setPage("learning")}>Learning Mode</button>
        <button onClick={() => setPage("manual")}>Manual</button>
        <button onClick={() => setPage("qr")}>QR</button>
        <button onClick={() => setPage("about")}>About</button>
      

        {/* 🌙 Theme Toggle */}
        <button
          onClick={() => setDark(!dark)}
          className="border rounded-lg px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
          title="Toggle theme"
        >
          {dark ? "☀️" : "🌙"}
        </button>

      </div>
    </div>
  );
}
