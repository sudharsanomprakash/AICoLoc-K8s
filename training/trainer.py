import json
import torch
from ai_engine.model import SimpleModel
import torch.nn as nn
import torch.optim as optim

def load_training_data(path):
    X, y = [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            for node, metrics in record["data"].items():
                if metrics["cpu"] >= 0:
                    features = [metrics["cpu"] / 100, metrics["mem"] / 100, 0, 0]
                    label = 1.0 if metrics["cpu"] < 80 else 0.0
                    X.append(features)
                    y.append(label)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def train_model(X, y):
    model = SimpleModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    return model

if __name__ == "__main__":
    X, y = load_training_data("results.json")
    model = train_model(X, y)
    torch.save(model.state_dict(), "model.pth")
