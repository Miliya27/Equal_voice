import { useState } from "react";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import Landing from "./pages/Landing";
import Interpreter from "./pages/Interpreter";
import Manual from "./pages/Manual";
import QR from "./pages/QR";
import About from "./pages/About";
import Learning from "./pages/Learning";
import TextToSign from "./pages/TextToSign";

export default function App() {
  const [page, setPage] = useState("home");
 
  const render = () => {
    switch (page) {
      case "interpreter": return <Interpreter />;

      case "learning": return <Learning />;
      case "manual": return <Manual />;
      case "qr": return <QR />;
      case "about": return <About />;
      case "textsign": return <TextToSign />;  
      default: return <Landing setPage={setPage} />;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100">

      <Navbar setPage={setPage} />

      <main className="flex-1">{render()}</main>
      <Footer />
    </div>
  );
}
