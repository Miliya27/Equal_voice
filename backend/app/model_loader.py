from pathlib import Path
import torch
import torch.nn as nn


# ✅ SAME MODEL STRUCTURE AS TRAIN.PY
class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "ml" / "data" / "models" / "best_model.pth"

print("Loading model from:", MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

classes = checkpoint["classes"]
num_classes = len(classes)

model = MLP(126, num_classes)   # 126 features — from your training
model.load_state_dict(checkpoint["model_state"])
model.eval()


def get_model():
    return model


def get_classes():
    return classes
