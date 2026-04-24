package vend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

// StartServices runs the scraper, exporter, and embedder loops concurrently.
func (v *Vend) StartServices() {
	log.Println("services: starting all services")

	if err := v.resetInProgressFlags(); err != nil {
		log.Printf("services: reset in-progress flags: %v", err)
	}

	go v.StartScrapers()
	go v.StartExporters()
	go v.StartEmbedders()
}

// resetInProgressFlags clears stale is_* flags left behind by a previous crash or shutdown.
// Without this, rows claimed by a killed worker remain locked forever.
func (v *Vend) resetInProgressFlags() error {
	ctx := context.Background()

	if _, err := v.db.Exec(ctx,
		`UPDATE source SET is_updating = false WHERE is_updating = true`); err != nil {
		return fmt.Errorf("reset source.is_updating: %w", err)
	}

	if _, err := v.db.Exec(ctx,
		`UPDATE drink SET is_exporting = false WHERE is_exporting = true`); err != nil {
		return fmt.Errorf("reset drink.is_exporting: %w", err)
	}

	if _, err := v.db.Exec(ctx,
		`UPDATE drink SET is_embedding = false WHERE is_embedding = true`); err != nil {
		return fmt.Errorf("reset drink.is_embedding: %w", err)
	}

	return nil
}

// StartScrapers loops, updating sources if stale.
func (v *Vend) StartScrapers() {
	log.Printf("StartScrapers: starting all scrapers")

	for {
		if err := v.runScraperOnce(); err != nil {
			log.Printf("scraper: %v", err)
		}
		time.Sleep(10 * time.Second)
	}
}

func (v *Vend) runScraperOnce() error {
	ctx := context.Background()

	cutoff := time.Now().Add(
		-time.Duration(v.config.ScrapeRefreshDays) * 24 * time.Hour)

	// Check if this source has ever been scraped.
	var is_new_row bool
	if err := v.db.QueryRow(ctx,
		`SELECT NOT EXISTS(SELECT 1 FROM source WHERE id = 'askul')`,
	).Scan(&is_new_row); err != nil {
		return fmt.Errorf("check source: %w", err)
	}

	// Atomically claim is_updating, skipping if fresh or already locked.
	var claimed_id string
	err := v.db.QueryRow(ctx, `
		INSERT INTO source (id, is_updating)
		VALUES ('askul', true)
		ON CONFLICT (id) DO UPDATE SET is_updating = true
		WHERE source.is_updating = false AND source.updated < $1
		RETURNING id
	`, cutoff).Scan(&claimed_id)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil // fresh or another scraper is running
	}
	if err != nil {
		return fmt.Errorf("claim scrape lock: %w", err)
	}

	defer func() {
		if is_new_row {
			v.db.Exec(ctx, `DELETE FROM source WHERE id = 'askul'`)
		} else {
			v.db.Exec(ctx, `UPDATE source SET is_updating = false WHERE id = 'askul'`)
		}
	}()

	log.Println("scraper: starting Askul scrape")
	drinks, err := ScrapeAskulDrinks(v.config.Japan.Askul.CategoryURLPrefix)
	if err != nil {
		return fmt.Errorf("ScrapeAskulDrinks: %w", err)
	}

	tx, err := v.db.Begin(ctx)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	for i := range drinks {
		extra_json, jerr := json.Marshal(drinks[i].ExtraJSON)
		if jerr != nil {
			log.Printf("scraper: marshal extra_json for %s: %v", drinks[i].Code, jerr)
			continue
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO drink (source_id, code, extra_json)
			VALUES ('askul', $1, $2)
			ON CONFLICT (source_id, code) DO NOTHING
		`, drinks[i].Code, extra_json); err != nil {
			return fmt.Errorf("insert drink %s: %w", drinks[i].Code, err)
		}
	}

	if _, err := tx.Exec(ctx, `
		UPDATE source SET is_updating = false, updated = NOW() WHERE id = 'askul'
	`); err != nil {
		return fmt.Errorf("release lock: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("commit: %w", err)
	}
	is_new_row = false // committed successfully — defer must not delete the row

	log.Printf("scraper: scraped and inserted %d drinks", len(drinks))
	return nil
}

// StartExporters spawns ExportWorkers goroutines, each with its own detector.
func (v *Vend) StartExporters() {
	log.Printf("StartExporters: starting all exporters")

	for i := range v.config.ExportWorkers {
		go func(id int) {
			// Every thread gets their own detector instance.
			detector, err := LoadYOLODetector(v.config.Japan.Askul.Model, false)
			if err != nil {
				log.Printf("StartExporters[%d]: load detector: %v", id, err)
				return
			}
			defer detector.Close()

			for {
				if err := v.runExporterOnce(detector); err != nil {
					log.Printf("exporter[%d]: %v", id, err)
				}
				time.Sleep(1 * time.Second)
			}
		}(i)
	}
}

func (v *Vend) runExporterOnce(detector *YOLODetector) error {
	ctx := context.Background()

	var drink VendDrink
	err := v.db.QueryRow(ctx, `
		WITH claimed AS (
			SELECT source_id, code FROM drink
			WHERE is_exported = false AND is_exporting = false AND not_a_drink = false
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		UPDATE drink SET is_exporting = true
		FROM claimed
		WHERE drink.source_id = claimed.source_id AND drink.code = claimed.code
		RETURNING drink.source_id, drink.code
	`).Scan(&drink.Source, &drink.Code)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("claim: %w", err)
	}
	defer v.db.Exec(ctx, `UPDATE drink SET is_exporting = false WHERE source_id = $1 AND code = $2`, drink.Source, drink.Code)

	export_start := time.Now()
	export, succeeded, err := ExportAskulDrink(detector, drink)
	log.Printf("exporter: %s/%s took %v", drink.Source, drink.Code, time.Since(export_start))

	if !succeeded {
		return fmt.Errorf("%s/%s: %w", drink.Source, drink.Code, err)
	}

	// If there is still an error this must mean YOLO did not find a drink.
	if err != nil {
		// YOLO could not find a drink. Do not reattempt this drink.
		log.Printf("exporter: found no drinks in %s", drink.Code)
		v.db.Exec(ctx,
			`UPDATE drink SET not_a_drink = true, is_exporting = false WHERE source_id = $1 AND code = $2`, drink.Source, drink.Code)
		return nil
	}

	if _, err := v.db.Exec(ctx, `
		INSERT INTO drink_image (source_id, code, image)
		VALUES ($1, $2, $3)
		ON CONFLICT (source_id, code) DO UPDATE SET image = EXCLUDED.image
	`, drink.Source, drink.Code, export.Image); err != nil {
		return fmt.Errorf("insert image %s/%s: %w", drink.Source, drink.Code, err)
	}

	if _, err := v.db.Exec(ctx, `
		UPDATE drink SET is_exported = true, is_exporting = false WHERE source_id = $1 AND code = $2
	`, drink.Source, drink.Code); err != nil {
		return fmt.Errorf("mark exported %s/%s: %w", drink.Source, drink.Code, err)
	}

	return nil
}

// StartEmbedders connects to Milvus once then spawns EmbedWorkers goroutines sharing the client.
func (v *Vend) StartEmbedders() {
	log.Printf("StartEmbedders: starting all embedders")

	milvus_addr := fmt.Sprintf("milvus-standalone:%d", v.config.MilvusPort)

	base_mc, err := client.NewClient(context.Background(),
		client.Config{Address: milvus_addr})
	if err != nil {
		log.Printf("StartEmbedders: connect milvus: %v", err)
		return
	}

	if err := ensureDrinkCollection(context.Background(), base_mc); err != nil {
		log.Printf("StartEmbedders: ensure collection: %v", err)
		return
	}

	err = base_mc.Close()
	if err != nil {
		log.Printf("StartEmbedders: could not close milvus: %v", err)
		return
	}

	embed_url := fmt.Sprintf("http://embed-service:%d", v.config.EmbedServicePort)

	for i := range v.config.EmbedWorkers {
		go func(id int) {
			// Create new Milvus client for each worker.
			mc, err := client.NewClient(context.Background(),
				client.Config{Address: milvus_addr})
			if err != nil {
				log.Printf("StartEmbedders: connect milvus: %v", err)
				return
			}
			defer mc.Close()

			// Iterate forever with a wait between each.
			for {
				if err := v.runEmbedderOnce(mc, embed_url); err != nil {
					log.Printf("embedder[%d]: %v", id, err)
				}

				time.Sleep(1 * time.Second)
			}
		}(i)
	}
}

func (v *Vend) runEmbedderOnce(mc client.Client, embed_url string) error {
	ctx := context.Background()

	var source_id, code string
	err := v.db.QueryRow(ctx, `
		WITH claimed AS (
			SELECT source_id, code FROM drink
			WHERE is_exported = true AND is_embedded = false AND is_embedding = false
			LIMIT 1
			FOR UPDATE SKIP LOCKED
		)
		UPDATE drink SET is_embedding = true
		FROM claimed
		WHERE drink.source_id = claimed.source_id AND drink.code = claimed.code
		RETURNING drink.source_id, drink.code
	`).Scan(&source_id, &code)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("claim: %w", err)
	}
	defer v.db.Exec(ctx, `UPDATE drink SET is_embedding = false WHERE source_id = $1 AND code = $2`, source_id, code)

	var image []byte
	if err := v.db.QueryRow(ctx,
		`SELECT image FROM drink_image WHERE source_id = $1 AND code = $2`, source_id, code,
	).Scan(&image); err != nil {
		return fmt.Errorf("get image %s/%s: %w", source_id, code, err)
	}

	embed_start := time.Now()
	drink_embed, err := EmbedAskulDrink(embed_url, image)
	log.Printf("embedder: %s/%s took %v", source_id, code, time.Since(embed_start))

	if err != nil {
		return fmt.Errorf("embed %s/%s: %w", source_id, code, err)
	}

	if _, err := mc.Insert(ctx, DRINK_COLLECTION_NAME, "",
		entity.NewColumnVarChar("source_id", []string{source_id}),
		entity.NewColumnVarChar("code", []string{code}),
		entity.NewColumnFloatVector("embedding", DRINK_EMBEDDING_DIMS, [][]float32{drink_embed.Embedding}),
	); err != nil {
		return fmt.Errorf("milvus insert %s/%s: %w", source_id, code, err)
	}

	if _, err := v.db.Exec(ctx, `
		UPDATE drink SET is_embedded = true, is_embedding = false WHERE source_id = $1 AND code = $2
	`, source_id, code); err != nil {
		return fmt.Errorf("mark embedded %s/%s: %w", source_id, code, err)
	}

	log.Printf("embedder: embedded %s/%s", source_id, code)
	return nil
}
