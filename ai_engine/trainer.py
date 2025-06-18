# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SimpleModel

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define training samples
# Simulate real-world co-location scoring
# Feature vector: [cpu, mem, is_db, is_frontend, is_backend, is_redis]
X = []
y = []

# Rule-based scoring for training data
# Higher score for co-location of same tier or affinity

def generate_sample(cpu, mem, is_db, is_frontend, is_backend, is_redis):
    features = [cpu, mem, is_db, is_frontend, is_backend, is_redis]
    # Scoring logic: lower usage and co-location
    score = 1 - 0.5 * cpu - 0.3 * mem
    if is_frontend:
        score += 0.2 * is_frontend
    if is_backend:
        score += 0.2 * is_backend
    if is_db:
        score += 0.2 * is_db
    return features, min(score, 1.0)

# Simulate combinations
for _ in range(1000):
    cpu = np.random.uniform(0.01, 0.9)
    mem = np.random.uniform(0.01, 0.8)
    for role in [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0)]:
        feat, score = generate_sample(cpu, mem, *role)
        X.append(feat)
        y.append([score])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Convert to tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Initialize model
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train
for epoch in range(200):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print(f"Training complete. Final loss: {loss.item():.4f}")
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")

