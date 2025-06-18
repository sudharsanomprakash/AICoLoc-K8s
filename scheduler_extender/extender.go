package main

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)

type ExtenderArgs struct {
	Pod   map[string]interface{} `json:"pod"`
	Nodes struct {
		Items []struct {
			Metadata struct {
				Name string `json:"name"`
			} `json:"metadata"`
		} `json:"items"`
	} `json:"nodes"`
}

type ExtenderFilterResult struct {
	Nodes       struct {
		Items []struct {
			Metadata struct {
				Name string `json:"name"`
			} `json:"metadata"`
		} `json:"items"`
	} `json:"nodes"`
	NodeNames   []string `json:"nodeNames,omitempty"`
	Recommended []string `json:"recommended,omitempty"`
	Error       string   `json:"error,omitempty"`
}

func handler(w http.ResponseWriter, r *http.Request) {
	var args ExtenderArgs
	if err := json.NewDecoder(r.Body).Decode(&args); err != nil {
		log.Printf("Error decoding extender request: %v", err)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	podName := "unknown"
	if m, ok := args.Pod["metadata"].(map[string]interface{}); ok {
		if n, exists := m["name"].(string); exists {
			podName = n
		}
	}
	log.Printf("Scheduling request for pod: %s", podName)

	nodeNames := []string{}
	for _, item := range args.Nodes.Items {
		nodeNames = append(nodeNames, item.Metadata.Name)
	}
	log.Printf("Candidate nodes: %v", nodeNames)

	// Prepare AI engine payload
	payload := map[string]interface{}{
		"pod":       args.Pod,
		"nodeNames": nodeNames,
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post("http://aicoloc-ai-engine.default.svc.cluster.local:5000/recommend", "application/json", bytes.NewBuffer(body))
	if err != nil {
		log.Printf("Error calling AI engine: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	respBody, _ := ioutil.ReadAll(resp.Body)
	var result ExtenderFilterResult
	if err := json.Unmarshal(respBody, &result); err != nil {
		log.Printf("Error decoding AI engine response: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	log.Printf("AI engine recommended nodes: %v", result.Recommended)

	// Build filtered node list for scheduler
	filtered := ExtenderFilterResult{}
	for _, name := range result.Recommended {
		for _, item := range args.Nodes.Items {
			if item.Metadata.Name == name {
				filtered.Nodes.Items = append(filtered.Nodes.Items, item)
			}
		}
	}

	json.NewEncoder(w).Encode(filtered)
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	log.Printf("🚀 AICoLoc Scheduler Extender running on port %s", port)
	http.HandleFunc("/filter", handler)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

