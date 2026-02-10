import os
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_csv(path):
    X, y = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            y.append(row[0])
            feats = np.array(row[1:], dtype=np.float32)
            X.append(feats)
    return np.array(X, dtype=np.float32), np.array(y)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/dataset.csv")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs("data/models", exist_ok=True)

    X, y = load_csv(args.data)
    print("Loaded:", X.shape, y.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print("Classes:", list(le.classes_))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(126, num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def make_loader(Xa, ya):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(Xa), torch.tensor(ya, dtype=torch.long)
        )
        return torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True)

    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)

    best_acc = 0.0
    best_path = "data/models/best_model.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)

        acc = correct / total if total else 0
        print(f"Epoch {epoch}/{args.epochs} | loss={total_loss:.3f} | val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": le.classes_.tolist()
            }, best_path)

    print(f"\n✅ Training done. Best val_acc={best_acc:.3f}")
    print("Saved:", best_path)

if __name__ == "__main__":
    main()
