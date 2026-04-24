package vend

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"

	"github.com/davidbyttow/govips/v2/vips"
	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/gorilla/mux"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/tidwall/jsonc"
)

// Vend orchestrates everything.
type Vend struct {
	db          *pgxpool.Pool
	server      *VendServer
	http_server *http.Server
	config      *VendConfig
}

type VendConfig struct {
	ClipModelsDir           string `json:"clip_models_dir"`
	Siglip2ModelsDir        string `json:"siglip2_models_dir"`
	ConvNextV2ModelsDir     string `json:"convnextv2_models_dir"`
	YoloModelsDir           string `json:"yolo_models_dir"`
	VendingMachineImagesDir string `json:"vending_machine_images_dir"`
	VendingMachineOutputDir string `json:"vending_machine_output_dir"`
	VendingMachineModel     string `json:"vending_machine_model"`
	ScrapeRefreshDays       int    `json:"scrape_refresh_days"`
	PostgresPort            int    `json:"postgres_port"`
	MilvusPort              int    `json:"milvus_port"`
	EmbedServicePort        int    `json:"embed_service_port"`
	ExportWorkers           int    `json:"export_workers"`
	EmbedWorkers            int    `json:"embed_workers"`
	Japan                   struct {
		Askul struct {
			OutputDir         string   `json:"output_dir"`
			ImagesDir         string   `json:"images_dir"`
			Model             string   `json:"model"`
			CategoryURLPrefix []string `json:"category_url_prefix"`
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

// StartDB connects to postgres and runs any pending migrations.
func (v *Vend) StartDB(migrations_fs fs.FS) error {
	pool, err := pgxpool.New(context.Background(),
		fmt.Sprintf("postgres://vend:vend@postgres:%d/vend", v.config.PostgresPort))
	if err != nil {
		return fmt.Errorf("connect db: %w", err)
	}
	v.db = pool

	conn := stdlib.OpenDBFromPool(pool)
	driver, err := postgres.WithInstance(conn, &postgres.Config{})
	if err != nil {
		return fmt.Errorf("migrate driver: %w", err)
	}
	src, err := iofs.New(migrations_fs, ".")
	if err != nil {
		return fmt.Errorf("migrate source: %w", err)
	}
	m, err := migrate.NewWithInstance("iofs", src, "postgres", driver)
	if err != nil {
		return fmt.Errorf("migrate init: %w", err)
	}
	if err := m.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("migrate up: %w", err)
	}
	return nil
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
