import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn

from hand_tracker import HandTracker
from utils import normalize_vec126, majority_vote

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/models/best_model.pth")
    parser.add_argument("--smooth", type=int, default=10, help="smoothing window")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu")
    classes = checkpoint["classes"]
    num_classes = len(classes)

    model = MLP(126, num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tracker = HandTracker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vec126, annotated, info = tracker.process(frame, draw=True)
        vec126 = normalize_vec126(vec126)

        x = torch.tensor(vec126, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            idx = int(np.argmax(probs))
            pred_label = classes[idx]
            conf = float(probs[idx])

        # smoothing
        history.append(pred_label)
        if len(history) > args.smooth:
            history.pop(0)
        smooth_label = majority_vote(history)

        cv2.putText(annotated, f"Pred: {smooth_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(annotated, f"Conf: {conf:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        cv2.imshow("Live Sign Prediction", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
