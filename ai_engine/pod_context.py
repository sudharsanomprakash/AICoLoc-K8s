def get_pod_context_features(pod):
    try:
        name = pod.get("metadata", {}).get("name", "").lower()
        labels = pod.get("metadata", {}).get("labels", {})

        is_db = "db" in name or labels.get("role") == "database"
        is_frontend = "frontend" in name or labels.get("tier") == "frontend"
        is_backend = "backend" in name or labels.get("tier") == "backend"

        return [
            1.0 if is_db else 0.0,
            1.0 if is_frontend else 0.0,
            1.0 if is_backend else 0.0
        ]
    except Exception as e:
        print(f"Error parsing pod context: {e}")
        return [0.0, 0.0, 0.0]

