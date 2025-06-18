from prometheus_api_client import PrometheusConnect
import time
import json

prom = PrometheusConnect(url="http://prometheus.monitoring.svc:9090", disable_ssl=True)
nodes = ["node1", "node2", "node3"]  # replace with your actual node names

def capture_metrics():
    data = {}
    for node in nodes:
        try:
            cpu_query = f"100 - (avg by(instance) (rate(node_cpu_seconds_total{{mode='idle',instance='{node}'}}[1m])) * 100)"
            mem_query = f"(1 - (node_memory_MemAvailable_bytes{{instance='{node}'}} / node_memory_MemTotal_bytes{{instance='{node}'}})) * 100"
            cpu = float(prom.custom_query(cpu_query)[0]['value'][1])
            mem = float(prom.custom_query(mem_query)[0]['value'][1])
            data[node] = {"cpu": cpu, "mem": mem}
        except Exception as e:
            print(f"Error: {e}")
            data[node] = {"cpu": -1, "mem": -1}

    timestamp = time.time()
    with open("results.json", "a") as f:
        json.dump({"timestamp": timestamp, "data": data}, f)
        f.write("\n")

if __name__ == "__main__":
    while True:
        capture_metrics()
        time.sleep(60)
