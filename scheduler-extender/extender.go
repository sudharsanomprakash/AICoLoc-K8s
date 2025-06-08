package main
import (
    "encoding/json"
    "net/http"
    "log"
)
type ExtenderArgs struct {
    Pod string `json:"pod"`
    Nodes []string `json:"nodes"`
}
type ExtenderResponse struct {
    Nodes []string `json:"nodes"`
}
func handler(w http.ResponseWriter, r *http.Request) {
    var args ExtenderArgs
    _ = json.NewDecoder(r.Body).Decode(&args)
    json.NewEncoder(w).Encode(ExtenderResponse{Nodes: args.Nodes})
}
func main() {
    http.HandleFunc("/filter", handler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
