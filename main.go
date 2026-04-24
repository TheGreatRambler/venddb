package main

import (
	"context"
	"embed"
	"flag"
	"io/fs"
	"log"

	"tgrcode.com/vend_db/vend"
)

//go:embed migrations
var MIGRATIONS_EMBED embed.FS

func main() {
	start_server := flag.Bool("start-server", false, "start the HTTP server on :8080")
	flag.Parse()

	v := vend.NewVend()

	if err := v.LoadConfig("config.jsonc"); err != nil {
		log.Fatal(err)
	}

	migrations_fs, err := fs.Sub(MIGRATIONS_EMBED, "migrations")
	if err != nil {
		log.Fatal(err)
	}

	if *start_server {
		log.Println("Starting Vendmigrations_fs DB server on :8080")

		if err := v.StartDB(migrations_fs); err != nil {
			log.Fatal(err)
		}

		defer v.StopServer(context.Background())

		// Run services to get drink data in the background.
		go v.StartServices()

		if err := v.StartServer(":8080"); err != nil {
			log.Fatal(err)
		}
	}
}
