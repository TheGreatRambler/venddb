package server

import (
	"github.com/gorilla/mux"
)

// VendServer holds the HTTP server and router
type VendServer struct {
	router *VendRouter
}

// NewVendServer creates a new VendServer with routes configured
func NewVendServer() *VendServer {
	return &VendServer{}
}

func (s *VendServer) AddRouter(r *mux.Router) {
	vend_router := NewVendRouter(s)
	vend_router.AddRouter(r)
	s.router = vend_router
}
