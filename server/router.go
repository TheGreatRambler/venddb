package server

import (
	"net/http"

	"github.com/gorilla/mux"
)

type VendRouter struct {
	Server *VendServer
}

func NewVendRouter(server *VendServer) *VendRouter {
	return &VendRouter{
		Server: server,
	}
}

// Add all routes
// TODO will use SolidJS generated website
func (vr *VendRouter) AddRouter(r *mux.Router) {
	r.HandleFunc("/", vr.handleHome).Methods("GET")
	r.HandleFunc("/health", vr.handleHealth).Methods("GET")
	r.HandleFunc("/api/items", vr.handleListItems).Methods("GET")
	r.HandleFunc("/api/items", vr.handleCreateItem).Methods("POST")
}

func (vr *VendRouter) handleHome(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("Vend DB API"))
}

func (vr *VendRouter) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

func (vr *VendRouter) handleListItems(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`[]`))
}

func (vr *VendRouter) handleCreateItem(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	w.Write([]byte(`{"id":"1"}`))
}
