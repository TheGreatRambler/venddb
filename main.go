package main

import (
	"log"
	"net/http"

	"github.com/gorilla/mux"
	"tgrcode.com/vend_db/server"
)

func main() {
	r := mux.NewRouter()

	vend_server := server.NewVendServer()
	vend_server.AddRouter(r)

	log.Println("Starting Vend DB server on :8080")
	webserver := &http.Server{
		Addr:    ":8080",
		Handler: r,
	}
	go func() {
		if err := webserver.ListenAndServe(); err != nil {
			log.Fatal(err)
		}
	}()
}
