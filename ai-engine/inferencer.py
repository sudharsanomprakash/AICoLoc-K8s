from flask import Flask, request, jsonify
import torch
import numpy as np
from model import SimplePolicyNet

app = Flask(__name__)
model = SimplePolicyNet(3, 16, 1)
model.load_state_dict(torch.load("model.pth"))  # Pretrained weights
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    inputs = torch.tensor(data['metrics'], dtype=torch.float32)
    with torch.no_grad():
        result = model(inputs).numpy().tolist()
    return jsonify({"decision": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
