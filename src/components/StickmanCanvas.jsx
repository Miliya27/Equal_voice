import { useRef, useEffect, useState } from "react";

export default function StickmanCanvas() {
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  const [status, setStatus] = useState("Ready");
  const [word, setWord] = useState("");

  const restingFrame = {
    FACE: { MOUTH: [150, 115] },
    TORSO: { TL: [50,200], TR: [250,200], BL: [70,450], BR: [230,450] },
    L_ARM: {
      SHOULDER:[50,200],
      ELBOW:[30,280],
      WRIST:[40,350],
      FINGERS:[[35,360],[38,365],[42,365],[45,360],[40,340]]
    },
    R_ARM: {
      SHOULDER:[250,200],
      ELBOW:[280,250],
      WRIST:[260,300],
      FINGERS:[[260,390],[255,305],[260,308],[265,308],[270,305]]
    }
  };

  function drawStickman(ctx, frame) {
  ctx.clearRect(0,0,600,500);

  ctx.save();

  // ✅ shift drawing to center of canvas
  ctx.translate(150, 0);

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "red";
  ctx.lineWidth = 6;

  // head
  ctx.beginPath();
  ctx.ellipse(150, 80, 40, 55, 0, 0, Math.PI * 2);
  ctx.stroke();

  // torso
  if(frame.TORSO){
    ctx.beginPath();
    ctx.moveTo(...frame.TORSO.TL);
    ctx.lineTo(...frame.TORSO.TR);
    ctx.lineTo(...frame.TORSO.BR);
    ctx.lineTo(...frame.TORSO.BL);
    ctx.closePath();
    ctx.stroke();
  }

  const drawArm = arm => {
    if(!arm) return;

    ctx.beginPath();
    ctx.moveTo(...arm.SHOULDER);
    ctx.lineTo(...arm.ELBOW);
    ctx.lineTo(...arm.WRIST);
    ctx.stroke();

    arm.FINGERS.forEach(pt=>{
      ctx.beginPath();
      ctx.moveTo(...arm.WRIST);
      ctx.lineTo(...pt);
      ctx.stroke();
    });
  };

  drawArm(frame.L_ARM);
  drawArm(frame.R_ARM);

  ctx.restore();
}


  useEffect(() => {
    const ctx = canvasRef.current.getContext("2d");
    drawStickman(ctx, restingFrame);
  }, []);

  async function playSign(inputWord) {
    const clean = inputWord.toLowerCase().trim();
    if (!clean) return;

    if (intervalRef.current) clearInterval(intervalRef.current);

    try {
      setStatus("Loading...");

      const data = await import(
        /* @vite-ignore */
        `./stickman/${clean}.json`
      );

      const frames = data.default.frames;
      const ctx = canvasRef.current.getContext("2d");

      let i = 0;

      intervalRef.current = setInterval(() => {
        if (i < frames.length) {
          drawStickman(ctx, frames[i]);
          i++;
        } else {
          clearInterval(intervalRef.current);
          setStatus("Done");
          setTimeout(() => drawStickman(ctx, restingFrame), 700);
        }
      }, 500);

    } catch {
      setStatus("Word not found");
    }
  }

  return (
    <div className="flex flex-col items-center justify-center text-center space-y-6 w-full">

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        width={600}
        height={500}
        className="mx-auto"
      />

      {/* Controls */}
      <div className="flex justify-center items-center gap-3">
        <input
          value={word}
          onChange={e => setWord(e.target.value)}
          onKeyDown={e => e.key === "Enter" && playSign(word)}
          placeholder="Type a word (yes, hello, please...)"
          className="border px-4 py-2 rounded-lg dark:bg-gray-800 dark:border-gray-700 w-72 text-center"
        />

        <button
          onClick={() => playSign(word)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg"
        >
          Play Sign
        </button>
      </div>

      {/* Status */}
      <div className="text-sm text-gray-600 dark:text-gray-400 text-center">
        {status}
      </div>

    </div>
  );
}
