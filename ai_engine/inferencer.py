import logging
from flask import Flask, request, jsonify
from prometheus_api_client import PrometheusConnect
import torch
from model import SimpleModel
from metrics import collect_node_metrics
from pod_context import get_pod_context_features

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("aicoloc-ai")

app = Flask(__name__)
log.info("Starting AICoLoc AI Engine")

# Connect to Prometheus inside cluster
prom = PrometheusConnect(url="http://prometheus-operated.default.svc.cluster.local:9090", disable_ssl=True)

# Load trained model
model = SimpleModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()
log.info("Model loaded successfully")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        pod = data.get("pod", {})
        node_names = data.get("nodeNames", [])
        pod_name = pod.get("metadata", {}).get("name", "unknown")

        log.info(f"Received scheduling request for pod: {pod_name}")
        log.info(f"Candidate nodes: {node_names}")

        features = []
        for node in node_names:
            metric_vec = collect_node_metrics(prom, node)
            pod_vec = get_pod_context_features(pod)
            full_vec = metric_vec + pod_vec
            features.append(full_vec)
            log.info(f"Node {node} → Features: {full_vec}")

        X = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            scores = model(X).squeeze().tolist()

        ranked_nodes = [node for _, node in sorted(zip(scores, node_names), reverse=True)]
        log.info(f" Node scores: {dict(zip(node_names, scores))}")
        log.info(f" Top nodes: {ranked_nodes[:3]}")

        return jsonify({"recommended": [ranked_nodes[0]]})
    
    except Exception as e:
        log.error(f"Error in /recommend: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

