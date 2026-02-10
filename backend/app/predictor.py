import torch
import numpy as np

from app.model_loader import get_model, get_classes
from ml.src.hand_tracker import HandTracker
from ml.src.utils import normalize_vec126

tracker = HandTracker()
CLASSES = get_classes()


def predict_from_frame(frame_bgr):

    model = get_model()

    # ✅ use your tracker correctly
    vec126, annotated, info = tracker.process(frame_bgr, draw=False)

    vec126 = normalize_vec126(vec126)

    x = torch.tensor(vec126, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        idx = int(torch.argmax(logits, dim=1))

    return CLASSES[idx]
