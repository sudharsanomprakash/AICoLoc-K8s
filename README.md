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
