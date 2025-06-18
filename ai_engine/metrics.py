def collect_node_metrics(prom, node):
    try:
        cpu_query = f"100 - (avg by(instance) (rate(node_cpu_seconds_total{{mode='idle',instance='{node}'}}[1m])) * 100)"
        mem_query = f"(1 - (node_memory_MemAvailable_bytes{{instance='{node}'}} / node_memory_MemTotal_bytes{{instance='{node}'}})) * 100"
        net_query = f"rate(node_network_receive_bytes_total{{instance='{node}'}}[1m])"
        pod_query = f"count by(node) (kube_pod_info{{node='{node}'}})"

        cpu = float(prom.custom_query(cpu_query)[0]['value'][1])
        mem = float(prom.custom_query(mem_query)[0]['value'][1])
        net = float(prom.custom_query(net_query)[0]['value'][1])
        pods = float(prom.custom_query(pod_query)[0]['value'][1])

        return [cpu / 100, mem / 100, net / 1e9, pods / 100]

    except Exception as e:
        print(f"Error collecting metrics for {node}: {e}")
        return [1.0, 1.0, 1.0, 1.0]
