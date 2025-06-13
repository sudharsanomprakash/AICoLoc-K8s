from flask import Flask, request, jsonify
import torch
import numpy as np
from model import SimplePolicyNet

app = Flask(__name__)

# Load the model
model = SimplePolicyNet(3, 16, 1)
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print(f"[AI] Received metrics: {data['metrics']}", flush=True)

        # Convert input to tensor
        inputs = torch.tensor(data['metrics'], dtype=torch.float32)

        # Inference
        with torch.no_grad():
            result = model(inputs)

        # Format result
        value = result.item() if result.numel() == 1 else result.numpy().tolist()
        print(f"[AI] Returning decision: {value}", flush=True)

        return jsonify({"decision": value})
    except Exception as e:
        print(f"[AI] ERROR: {str(e)}", flush=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
