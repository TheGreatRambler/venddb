package vend

import (
	"log"

	"github.com/gorilla/mux"
	"github.com/joho/godotenv"
)

// VendServer holds the HTTP server and router
type VendServer struct {
	vend          *Vend
	router        *VendRouter
	images_dir    string
	output_dir    string
	vend_detector *YOLODetector
}

// NewVendServer creates a new VendServer with routes configured.
// Call VendServer.Shutdown() on exit to clean up libvips.
func NewVendServer(vend *Vend) *VendServer {
	s := &VendServer{vend: vend}

	if err := godotenv.Load(); err != nil {
		log.Println("No .env file found, reading environment directly")
	}

	s.images_dir = vend.config.VendingMachineImagesDir
	s.output_dir = vend.config.VendingMachineOutputDir

	return s
}

// Creates the YOLO vending machine detector
func (s *VendServer) StartVendDetector(yolo_schema string) error {
	detector, err := LoadYOLODetector(yolo_schema, false)
	if err != nil {
		return err
	}
	s.vend_detector = detector
	return nil
}

// Shutdown releases all resources used by the VendServer.
func (s *VendServer) Shutdown() {
	s.vend_detector.Close()
}

func (s *VendServer) AddRouter(r *mux.Router) {
	vend_router := NewVendRouter(s)
	vend_router.AddRouter(r)
	s.router = vend_router
}
