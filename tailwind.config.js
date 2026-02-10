export default {
  darkMode: "class",   // ✅ important
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: "#2563eb",
        ink: "#0f172a",
        soft: "#f8fafc",
        line: "#e5e7eb"
      }
    },
  },
  plugins: [],
}
