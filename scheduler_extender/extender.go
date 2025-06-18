package main

import (
    "bytes"
    "encoding/json"
    "log"
    "net/http"
    "os"
)

type ExtenderArgs struct {
    Pod   map[string]interface{} `json:"pod"`
    Nodes struct {
        Items []map[string]interface{} `json:"items"`
    } `json:"nodes"`
}

type ExtenderResponse struct {
    Nodes struct {
        Items []map[string]interface{} `json:"items"`
    } `json:"nodes"`
}

func handler(w http.ResponseWriter, r *http.Request) {
    var args ExtenderArgs
    if err := json.NewDecoder(r.Body).Decode(&args); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    nodeNames := []string{}
    for _, node := range args.Nodes.Items {
        name := node["metadata"].(map[string]interface{})["name"].(string)
        nodeNames = append(nodeNames, name)
    }

    aiPayload := map[string]interface{}{
        "pod":       args.Pod,
        "nodeNames": nodeNames,
    }

    payload, _ := json.Marshal(aiPayload)
    aiURL := os.Getenv("AI_ENGINE_URL")
    resp, err := http.Post(aiURL+"/recommend", "application/json", bytes.NewBuffer(payload))
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    defer resp.Body.Close()

    var result struct {
        Recommended []string `json:"recommended"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    filtered := []map[string]interface{}{}
    for _, n := range args.Nodes.Items {
        name := n["metadata"].(map[string]interface{})["name"].(string)
        for _, r := range result.Recommended {
            if name == r {
                filtered = append(filtered, n)
            }
        }
    }

    response := ExtenderResponse{}
    response.Nodes.Items = filtered
    json.NewEncoder(w).Encode(response)
}

func main() {
    http.HandleFunc("/filter", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
