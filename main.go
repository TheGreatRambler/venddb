package main

import (
	"context"
	"encoding/json"
	"flag"
	"log"
	"os"

	"tgrcode.com/vend_db/vend"
)

func main() {
	start_server := flag.Bool("start-server", false, "start the HTTP server on :8080")
	export_askul := flag.Bool("export-askul", false, "export Askul vending machine data")
	flag.Parse()

	v := vend.NewVend()

	if err := v.LoadConfig("config.jsonc"); err != nil {
		log.Fatal(err)
	}

	if *start_server {
		log.Println("Starting Vend DB server on :8080")

		defer v.StopServer(context.Background())
		if err := v.StartServer(":8080"); err != nil {
			log.Fatal(err)
		}
	}

	if *export_askul {
		log.Println("Exporting Askul vending machine data")

		var drinks []vend.AskulDrink
		if data, err := os.ReadFile("askul_drinks.json"); err == nil {
			if err := json.Unmarshal(data, &drinks); err != nil {
				log.Fatal(err)
			}
			log.Printf("Loaded %d drinks from askul_drinks.json", len(drinks))
		} else {
			drinks, err = vend.ScrapeAskulDrinks(v.Config().Japan.Askul.CategoryURLPrefix)
			if err != nil {
				log.Fatal(err)
			}
			data, err := json.MarshalIndent(drinks, "", "  ")
			if err != nil {
				log.Fatal(err)
			}
			if err := os.WriteFile("askul_drinks.json", data, 0644); err != nil {
				log.Fatal(err)
			}
			log.Printf("Scraped and saved %d drinks to askul_drinks.json", len(drinks))
		}

		if err := vend.ExportAskulDrinks(v.Config().Japan.Askul.Model, drinks, v.Config().Japan.Askul.OutputDir); err != nil {
			log.Fatal(err)
		}
	}
}
