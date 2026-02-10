import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

torch.set_num_threads(1)  
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== LOAD MODEL =====

ckpt = torch.load("models/model.pt", map_location="cpu")
state_dict = ckpt["model_state"]

classes = [
    "YES","THANKYOU","NO","HELP","PLEASE",
    "HELLO","UNKNOWN","NAME","SORRY",
    "UP","DOWN","FOOD","IDLE"
]

model = nn.Sequential(
    nn.Linear(126,256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128,len(classes))
)

fixed = {k.replace("net.",""):v for k,v in state_dict.items()}
model.load_state_dict(fixed, strict=True)
model.eval()

print("✅ Model loaded")

# ===== API =====

class PredictReq(BaseModel):
    features: list

@app.post("/predict")
def predict(req: PredictReq):

    x = np.array(req.features, dtype=np.float32)

    if x.shape[0] != 126:
        return {"error": "need 126 features"}

    t = torch.tensor(x).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].numpy()

    idx = np.argsort(-probs)

    return {
        "topLabel": classes[idx[0]],
        "topConf": float(probs[idx[0]]),
        "secondLabel": classes[idx[1]],
        "secondConf": float(probs[idx[1]])
    }
