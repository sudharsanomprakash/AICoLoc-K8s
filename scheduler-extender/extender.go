package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"time"
)

type ExtenderArgs struct {
	Pod   Pod      `json:"pod"`
	Nodes NodeList `json:"nodes"`
}

type Pod struct {
	Metadata Metadata `json:"metadata"`
}

type Metadata struct {
	Name string `json:"name"`
}

type NodeList struct {
	Items []Node `json:"items"`
}

type Node struct {
	Metadata Metadata `json:"metadata"`
}

type ExtenderFilterResult struct {
	Nodes      NodeList          `json:"nodes"`
	NodeNames  []string          `json:"nodeNames,omitempty"`
	FailedNodes map[string]string `json:"failedNodes,omitempty"`
	Error      string            `json:"error,omitempty"`
}

func queryPrometheus(query string) (float64, error) {
	promURL := os.Getenv("PROM_URL")
	if promURL == "" {
		promURL = "http://prometheus-operated.default.svc.cluster.local:9090"
	}

	fullURL := fmt.Sprintf("%s/api/v1/query?query=%s", promURL, url.QueryEscape(query))
	resp, err := http.Get(fullURL)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	body, _ := io.ReadAll(resp.Body)
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, err
	}

	data, ok := result["data"].(map[string]interface{})
	if !ok || data["result"] == nil {
		return 0, fmt.Errorf("invalid data format")
	}

	results := data["result"].([]interface{})
	if len(results) == 0 {
		return 0, nil // Treat as 0 if no data
	}
	value := results[0].(map[string]interface{})["value"].([]interface{})[1].(string)

	var val float64
	fmt.Sscanf(value, "%f", &val)
	return val, nil
}

func getMetricsForNode(nodeName string) ([]float32, error) {
	cpuQ := fmt.Sprintf(`avg(container_cpu_usage_seconds_total{node="%s"})`, nodeName)
	memQ := fmt.Sprintf(`avg(container_memory_usage_bytes{node="%s"})`, nodeName)
	netQ := fmt.Sprintf(`avg(container_network_transmit_bytes_total{node="%s"})`, nodeName)

	cpu, _ := queryPrometheus(cpuQ)
	mem, _ := queryPrometheus(memQ)
	net, _ := queryPrometheus(netQ)

	return []float32{float32(cpu), float32(mem), float32(net)}, nil
}

func getBestNodeAI(metricsMap map[string][]float32, nodeNames []string) (string, error) {
	aiURL := os.Getenv("AI_ENGINE_URL")
	if aiURL == "" {
		aiURL = "http://ai-engine.default.svc.cluster.local:5000/predict"
	}

	// Average metrics across nodes for now
	metrics := []float32{0, 0, 0}
	for _, m := range metricsMap {
		metrics[0] += m[0]
		metrics[1] += m[1]
		metrics[2] += m[2]
	}
	n := float32(len(metricsMap))
	for i := range metrics {
		metrics[i] /= n
	}

	body, _ := json.Marshal(map[string][]float32{"metrics": metrics})
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Post(aiURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("AI engine unreachable: %v", err)
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("Invalid AI engine response: %v", err)
	}

	decision := int(result["decision"].(float64)) % len(nodeNames)
	return nodeNames[decision], nil
}

func filterHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Only POST allowed", http.StatusMethodNotAllowed)
		return
	}

	var args ExtenderArgs
	if err := json.NewDecoder(r.Body).Decode(&args); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	allNodes := args.Nodes.Items
	nodeNames := extractNodeNames(allNodes)
	log.Printf("Incoming pod: %s", args.Pod.Metadata.Name)
	log.Printf("Candidate nodes: %v", nodeNames)

	metricsMap := map[string][]float32{}
	for _, n := range nodeNames {
		m, err := getMetricsForNode(n)
		if err != nil {
			log.Printf("Failed to fetch metrics for %s: %v", n, err)
		}
		metricsMap[n] = m
		log.Printf("Metrics for %s: %v", n, m)
	}

	selectedNodeName, err := getBestNodeAI(metricsMap, nodeNames)
	if err != nil {
		log.Printf("AI error: %v", err)
		http.Error(w, fmt.Sprintf("AI error: %v", err), http.StatusInternalServerError)
		return
	}

	var selected []Node
	for _, n := range allNodes {
		if n.Metadata.Name == selectedNodeName {
			selected = append(selected, n)
			break
		}
	}

	log.Printf("Filtered node: %v", selectedNodeName)
	result := ExtenderFilterResult{Nodes: NodeList{Items: selected}}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func extractNodeNames(nodes []Node) []string {
	names := []string{}
	for _, n := range nodes {
		names = append(names, n.Metadata.Name)
	}
	return names
}

func healthzHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "ok")
}

func main() {
	log.SetOutput(os.Stdout)
	http.HandleFunc("/filter", filterHandler)
	http.HandleFunc("/healthz", healthzHandler)
	log.Println("Scheduler extender listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
