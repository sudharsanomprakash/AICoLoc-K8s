from flask import Flask, request, jsonify
from prometheus_api_client import PrometheusConnect
import torch
from model import SimpleModel
from metrics import collect_node_metrics
from pod_context import get_pod_context_features

app = Flask(__name__)
prom = PrometheusConnect(url="http://prometheus.monitoring.svc:9090", disable_ssl=True)

model = SimpleModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    pod = data["pod"]
    node_names = data["nodeNames"]

    features = []
    for node in node_names:
        metric_vec = collect_node_metrics(prom, node)        # 4 features
        pod_vec = get_pod_context_features(pod)              # 2 features
        features.append(metric_vec + pod_vec)                # Total: 6 features

    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        scores = model(X).squeeze().tolist()

    ranked_nodes = [node for _, node in sorted(zip(scores, node_names), reverse=True)]
    return jsonify({"recommended": ranked_nodes[:3]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
