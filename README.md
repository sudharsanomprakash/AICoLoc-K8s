# AICoLoc-K8s

AICoLoc-K8s is an AI-driven scheduler extender for Kubernetes that improves pod placement by learning from live cluster metrics and service communication patterns. It enables adaptive co-location of microservices based on real-time CPU, memory, and traffic data to reduce inter-service latency and optimize resource utilization.

# Features

- Scheduler Extender integrated with default Kubernetes scheduling via `/filter` API.
- Prometheus-based real-time metric collection (CPU, memory, network).
- AI Decision Engine using trained model to score node suitability for incoming pods.
- Tier-aware co-location: intelligently places frontend, backend, and DB services together.
- Improves latency, CPU utilization, and node packing efficiency.

# Architecture

<img width="1089" alt="Image" src="https://github.com/user-attachments/assets/8c3e8c13-bdfb-4be1-84c3-38cb0fe46700" />


# System Environment
Cluster: 3-node Kubernetes (RKE2) cluster on KubeVirt VMs
Nodes: Each with 8 vCPU, 16GB RAM
OS: Rocky Linux 9.4
Kubernetes Version: v1.32.5+rke2r1
Prometheus Stack: kube-prometheus-stack v0.83.0
Model Deployment: Flask-based AI microservice (aicoloc-ai-engine)

# Getting Started

1. Clone the Repository

git clone https://github.com/your-org/AICoLoc-K8s.git
cd AICoLoc-K8s/k8s_manifests

2. Deploy the AI Engine

kubectl apply -f ai_engine.yaml

3. Deploy Scheduler Extender

kubectl apply -f scheduler_extender.yaml

4. Configure Kubernetes to Use Extender
Make sure your scheduler configuration includes the extender in the --config file:

extenders:
  - urlPrefix: "http://aicoloc-extender.default.svc.cluster.local:8080"
    filterVerb: "filter"
    enableHTTPS: false
    
5. Metrics Configuration
Ensure Prometheus is running and scraping node metrics:

kubectl get svc -n monitoring
kubectl port-forward svc/prometheus-operated 9090:9090 -n monitoring

You can query metrics like:
- node_cpu_seconds_total
- node_memory_Active_bytes
- container_network_transmit_bytes_total

