package vend

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/gorilla/mux"
	"github.com/tidwall/jsonc"
)

// Vend orchestrates everything.
type Vend struct {
	server      *VendServer
	http_server *http.Server
	config      *VendConfig
}

type VendConfig struct {
	YoloModelsDir           string `json:"yolo_models_dir"`
	VendingMachineImagesDir string `json:"vending_machine_images_dir"`
	VendingMachineOutputDir string `json:"vending_machine_output_dir"`
	VendingMachineModel     string `json:"vending_machine_model"`
	Japan                   struct {
		AskulImagesDir string `json:"askul_images_dir"`
		Askul          struct {
			Model             string   `json:"model"`
			CategoryURLPrefix []string `json:"category_url_prefix"`
			OutputDir         string   `json:"output_dir"`
		} `json:"askul"`
	} `json:"japan"`
}

func NewVend() *Vend {
	// Load libvips.
	// Used by both server and scraping tasks.
	vips.LoggingSettings(nil, vips.LogLevelCritical)
	vips.Startup(nil)

	return &Vend{}
}

// LoadConfig reads a JSONC config file and populates config.
func (v *Vend) LoadConfig(path string) error {
	config := &VendConfig{}
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	err = json.Unmarshal(jsonc.ToJSON(data), config)
	if err != nil {
		return err
	}

	v.config = config
	return nil
}

// Start registers routes and begins listening on addr (e.g. ":8080").
func (v *Vend) StartServer(addr string) error {
	if v.config == nil {
		return fmt.Errorf("config must be loaded prior to starting server")
	}

	v.server = NewVendServer(v)

	r := mux.NewRouter()
	v.server.AddRouter(r)
	v.http_server = &http.Server{
		Addr:    addr,
		Handler: r,
	}

	return v.http_server.ListenAndServe()
}

// Stop gracefully shuts down the HTTP server and releases VendServer resources.
func (v *Vend) StopServer(ctx context.Context) error {
	err := v.http_server.Shutdown(ctx)
	v.server.Shutdown()
	vips.Shutdown()

	return err
}

// Config returns the loaded VendConfig.
func (v *Vend) Config() *VendConfig {
	return v.config
}

// Server returns the underlying VendServer for direct configuration.
func (v *Vend) Server() *VendServer {
	return v.server
}
