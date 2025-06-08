# save_model.py
import torch
from model import SimplePolicyNet

# Match this with your inferencer.py and model.py structure
input_size = 3      # CPU, memory, network metrics
hidden_size = 16
output_size = 1     # Just a single decision score

# Initialize model
model = SimplePolicyNet(input_size, hidden_size, output_size)

# Save dummy weights
torch.save(model.state_dict(), "model.pth")
print("Dummy model.pth created successfully.")
