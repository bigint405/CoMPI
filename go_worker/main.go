package main

import (
	"log"
	"runtime"
)

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)
	workerManager, err := NewWorkerManager("../configs/worker_config.json")
	if err != nil {
		log.Panicln(err)
	}
	if !workerManager.Start() {
		log.Panicf("ERROR: start worker manager failed")
	}
	workerManager.Join()
}
