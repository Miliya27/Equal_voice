import os
import csv
import argparse
import cv2
import numpy as np

from hand_tracker import HandTracker
from utils import normalize_vec126

def ensure_dirs():
    os.makedirs("data/raw", exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label name, e.g., HELLO")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--outfile", default="data/raw/dataset.csv")
    args = parser.parse_args()

    ensure_dirs()
    tracker = HandTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("\n✅ Instructions:")
    print("1) Put your hands in camera")
    print("2) Press 's' to start collecting")
    print("3) Press 'q' to quit\n")

    collecting = False
    collected = 0

    file_exists = os.path.exists(args.outfile)

    with open(args.outfile, "a", newline="") as f:
        writer = csv.writer(f)

        # write header once
        if not file_exists:
            header = ["label"] + [f"f{i}" for i in range(126)]
            writer.writerow(header)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            vec126, annotated, info = tracker.process(frame, draw=True)

            # normalize (important for accuracy)
            vec126 = normalize_vec126(vec126)

            # UI text
            cv2.putText(annotated, f"Label: {args.label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(annotated, f"Collected: {collected}/{args.samples}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)

            if not collecting:
                cv2.putText(annotated, "Press 's' to START", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

            cv2.imshow("Collect Dataset", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                collecting = True

            if collecting and collected < args.samples:
                # save only when at least one hand is detected (recommended)
                if info["found_left"] or info["found_right"]:
                    row = [args.label] + vec126.tolist()
                    writer.writerow(row)
                    collected += 1

            if collected >= args.samples:
                print(f"✅ Done! Collected {args.samples} samples for {args.label}")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
