from prometheus_api_client import PrometheusConnect
from kubernetes import client, config
import random
import logging

# Prometheus setup
prom = PrometheusConnect(url="http://prometheus-operated.default.svc.cluster.local:9090", disable_ssl=True)

# Kubernetes config (works inside cluster)
try:
    config.load_incluster_config()
except:
    config.load_kube_config()
v1 = client.CoreV1Api()

# Resolve node name → IP:9100 for Prometheus query
def get_node_instance_address(node_name):
    try:
        node = v1.read_node(node_name)
        for addr in node.status.addresses:
            if addr.type == "InternalIP":
                return f"{addr.address}:9100"
    except Exception as e:
        logging.warning(f"Could not resolve IP for node {node_name}: {e}")
    return None

def collect_node_metrics(prom, node_name):
    try:
        instance = get_node_instance_address(node_name)
        if not instance:
            raise Exception("No valid instance address found")

        cpu_query = (
            f"100 - (avg by(instance) (rate(node_cpu_seconds_total{{mode='idle',instance='{instance}'}}[1m])) * 100)"
        )
        mem_query = (
            f"(1 - (node_memory_MemAvailable_bytes{{instance='{instance}'}} / "
            f"node_memory_MemTotal_bytes{{instance='{instance}'}})) * 100"
        )

        logging.info(f"[PROMQL] CPU Query: {cpu_query}")
        logging.info(f"[PROMQL] MEM Query: {mem_query}")

        cpu_result = prom.custom_query(cpu_query)
        mem_result = prom.custom_query(mem_query)

        cpu = float(cpu_result[0]['value'][1]) if cpu_result else random.uniform(70, 90)
        mem = float(mem_result[0]['value'][1]) if mem_result else random.uniform(60, 80)

        logging.info(f"[PROM] {node_name}: CPU={cpu:.2f}%, MEM={mem:.2f}%")
        return [cpu / 100, mem / 100]

    except Exception as e:
        logging.warning(f"Error collecting metrics for {node_name}: {e}")
        return [random.uniform(0.7, 0.9), random.uniform(0.6, 0.8)]

