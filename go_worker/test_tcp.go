package main

import (
	"flag"
	"log"
	"math/rand"
	"net"
	"os"
	"os/signal"
	"sort"
	"sync"
	"syscall"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

var (
	mode   = flag.String("m", "c", "s for server or c for client")
	repeat = flag.Int("r", 100, "number of vectors to send")
	addr   = flag.String("addr", "localhost:9000", "tcp address")
	size   = flag.Int("size", 16*3*224*224*4, "byte size to send (float32 size)")
)

func generateRandomBytes(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}

func recordCDF(data []float64, output string) error {
	sort.Float64s(data)
	pts := make(plotter.XYs, len(data))
	for i, v := range data {
		pts[i].X = v
		pts[i].Y = float64(i+1) / float64(len(data))
	}
	p := plot.New()
	p.Title.Text = "CDF of Time (ms)"
	p.X.Label.Text = "Time (ms)"
	p.Y.Label.Text = "CDF"
	if err := plotutil.AddLinePoints(p, "CDF", pts); err != nil {
		return err
	}
	return p.Save(6*vg.Inch, 4*vg.Inch, output)
}

func runClient() {
	mu := &sync.Mutex{}
	conn, err := net.Dial("tcp", *addr)
	if err != nil {
		log.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()

	times := make([]float64, *repeat)
	payload := generateRandomBytes(*size)

	for i := 0; i < *repeat; i++ {
		start := time.Now()
		mu.Lock()
		err := SendBytesWithID(conn, uint32(i), payload)
		mu.Unlock()
		if err != nil {
			log.Printf("Send error: %v", err)
			continue
		}
		times[i] = float64(time.Since(start).Microseconds()) / 1000.0
	}

	if err := recordCDF(times, "client_cdf.png"); err != nil {
		log.Printf("Plot client CDF failed: %v", err)
	} else {
		log.Println("✅ Client CDF saved to client_cdf.png")
	}
}

func runServer() {
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}
	log.Printf("Listening on %s", *addr)

	done := make(chan struct{})
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		conn, err := ln.Accept()
		if err != nil {
			log.Fatalf("Accept failed: %v", err)
		}
		defer conn.Close()

		times := make([]float64, *repeat)
		for i := 0; i < *repeat; i++ {
			start := time.Now()
			_, _, err := ReceiveBytesWithID(conn)
			if err != nil {
				log.Printf("Receive error: %v", err)
				continue
			}
			times[i] = float64(time.Since(start).Microseconds()) / 1000.0
		}
		if err := recordCDF(times, "server_cdf.png"); err != nil {
			log.Printf("Plot server CDF failed: %v", err)
		} else {
			log.Println("✅ Server CDF saved to server_cdf.png")
		}
		close(done)
	}()

	select {
	case <-done:
		log.Println("Server done.")
	case <-sig:
		log.Println("Terminated by signal")
		_ = ln.Close()
	}
}

// func main() {
// 	flag.Parse()
// 	if *mode == "s" {
// 		runServer()
// 	} else {
// 		runClient()
// 	}
// }
