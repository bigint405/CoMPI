package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"log"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

const target_url = "http://localhost:50050/predict"

var (
	latencies  []float64
	latencyMu  sync.Mutex
	sucNum     int
	failNum    int
	discardNum int
)

// 生成随机 float32 向量
func generateRandomTensor(shape []int64) []float32 {
	size := int64(1)
	for _, dim := range shape {
		size *= dim
	}
	data := make([]float32, size)
	for i := range data {
		data[i] = rand.Float32()*2 - 1
	}
	return data
}

// 使用 CGO 量化后的 multipart 构建
func buildMultipartRequestQuantized(tensorBytes []byte, modelID string) (*bytes.Buffer, string, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	writer.WriteField("model_id", modelID)

	tensorPart, err := writer.CreateFormFile("tensor", "quantized.bin")
	if err != nil {
		return nil, "", err
	}
	tensorPart.Write(tensorBytes)

	writer.Close()
	return body, writer.FormDataContentType(), nil
}

func parseShapes(shapeStrs []string) ([][]int64, error) {
	var shapes [][]int64
	for _, s := range shapeStrs {
		dims := strings.Split(s, ",")
		var shape []int64
		for _, d := range dims {
			val, err := strconv.Atoi(d)
			if err != nil {
				return nil, err
			}
			shape = append(shape, int64(val))
		}
		shapes = append(shapes, shape)
	}
	return shapes, nil
}

// === 核心请求函数 ===
func processRequest(dataBytes []byte, modelID string, timeout float64) {
	body, contentType, err := buildMultipartRequestQuantized(dataBytes, modelID)
	if err != nil {
		log.Printf("model %s 构建 multipart 请求失败: %e", modelID, err)
		return
	}

	req, err := http.NewRequest("POST", target_url, body)
	if err != nil {
		log.Printf("model %s 创建请求失败: %e", modelID, err)
		return
	}
	req.Header.Set("Content-Type", contentType)

	client := &http.Client{}
	if timeout > 0 {
		client.Timeout = time.Duration(timeout * float64(time.Second))
	}
	start := time.Now()
	resp, err := client.Do(req)
	latency := time.Since(start).Seconds()

	if err != nil {
		// 判断是否为超时错误
		if urlErr, ok := err.(*url.Error); ok && urlErr.Timeout() {
			latencyMu.Lock()
			failNum++
			if failNum%100 == 0 || failNum == 1 {
				log.Printf("model %s 请求超时: %d", modelID, failNum)
			}
			latencyMu.Unlock()
		} else {
			log.Printf("model %s 请求失败: %e", modelID, err)
			latencyMu.Lock()
			failNum++
			latencyMu.Unlock()
		}
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		if resp.StatusCode == 429 {
			latencyMu.Lock()
			discardNum++
			if discardNum%100 == 0 || discardNum == 1 {
				log.Printf("model %s 请求被丢弃: %d", modelID, discardNum)
			}
			latencyMu.Unlock()
		} else {
			log.Printf("model %s 服务端错误: %d", modelID, resp.StatusCode)
		}
		return
	}

	latencyMu.Lock()
	sucNum++
	latencies = append(latencies, latency)
	latencyMu.Unlock()
}

func percentile(p float64, data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	index := int(float64(len(data)-1) * p)
	return data[index]
}

func computeCDF(data []float64) *plotter.XYs {
	sort.Float64s(data)
	n := len(data)
	points := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		points[i].X = data[i]
		points[i].Y = float64(i+1) / float64(n)
	}
	return &points
}

func plotCDF(cdfPoints *plotter.XYs, cdf_path string) {
	p := plot.New()
	p.Title.Text = "CDF of Latency"
	p.X.Label.Text = "Latency (s)"
	p.Y.Label.Text = "Cumulative Probability"

	line, err := plotter.NewLine(*cdfPoints)
	if err != nil {
		log.Fatalf("创建折线图失败: %v", err)
	}
	line.Color = color.RGBA{B: 255, A: 255}
	p.Add(line)

	if err := p.Save(6*vg.Inch, 4*vg.Inch, cdf_path); err != nil {
		log.Fatalf("保存 CDF 图失败: %v", err)
	}
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	log.SetFlags(log.LstdFlags | log.Lmicroseconds)

	numReq := flag.Int("n", -1, "总请求数，如果设为0且分布为n则默认为p的5倍，即发送5次；为负数则代表一直发送")
	reqPerSec := flag.Float64("r", 10, "每秒请求数")
	peakLoad := flag.Int("p", 16, "单次一次发送的峰值负载（gamma分布和按次发送时需要）")
	shapeStr := flag.String("s", "1,3,224,224", "形状列表，用逗号分隔，一次发送多个向量其形状用分号分隔")
	modelID := flag.String("m", "M1", "模型 ID")
	timeout := flag.Float64("t", 10, "超时秒数")
	distribution := flag.String("d", "n", "分布，g是gamma，p是泊松，n是按次发送（每次发送峰值负载的数据，等都返回了再发送下一次）")
	cdf_path := flag.String("cdf", "", "cdf图保存路径")
	latencyPath := flag.String("latency", "", "延迟数据保存路径")
	flag.Parse()

	shapes, err := parseShapes(strings.Split(*shapeStr, ";"))
	if err != nil {
		log.Fatalf("解析 shape 失败: %v", err)
	}

	var vectors [][]float32
	for _, shape := range shapes {
		vectors = append(vectors, generateRandomTensor(shape))
	}
	dataBytes, release, err := QuantizeVectorsFBGEMMBytes(vectors, shapes)

	if err != nil {
		log.Println("量化失败:", err)
		latencyMu.Lock()
		failNum++
		latencyMu.Unlock()
		return
	}
	defer release()

	seed := uint64(time.Now().UnixNano())
	src := rand.NewSource(seed)
	poissonDist := rand.New(src)
	gammaDist := distuv.Gamma{
		Alpha: float64(*peakLoad),
		Beta:  *reqPerSec,
		Src:   rand.New(src),
	}

	if *numReq == 0 {
		if *distribution == "n" {
			*numReq = *peakLoad * 5
		}
	}

	var wg sync.WaitGroup
	for *numReq != 0 {
		reqNum := 1
		if *distribution == "g" || *distribution == "n" {
			if *numReq > 0 {
				reqNum = min(*peakLoad, *numReq)
			} else {
				reqNum = *peakLoad
			}
		}

		if *numReq > 0 {
			*numReq -= reqNum
		}

		if *distribution == "n" { //按次发送
			var wgn sync.WaitGroup
			for range reqNum {
				wgn.Add(1)
				wg.Add(1)
				go func() {
					defer wgn.Done()
					defer wg.Done()
					processRequest(dataBytes, *modelID, *timeout)
				}()
			}
			wgn.Wait()
		} else {
			if *distribution == "p" {
				// 泊松过程，指数分布间隔
				interarrivalTime := poissonDist.ExpFloat64() / *reqPerSec
				// log.Printf("sleep %.4f s", interarrivalTime)
				time.Sleep(time.Duration(interarrivalTime * float64(time.Second)))

				// 提交请求事件
				for range reqNum {
					wg.Add(1)
					go func() {
						defer wg.Done()
						processRequest(dataBytes, *modelID, *timeout)
					}()
				}
			} else if *distribution == "g" {
				// Gamma 分布
				interarrivalTime := gammaDist.Rand()
				log.Printf("model %s: sleep %.4f s", *modelID, interarrivalTime)
				time.Sleep(time.Duration(interarrivalTime * float64(time.Second)))
				log.Printf("model %s: 请求数: %d, 剩余请求数: %d", *modelID, reqNum, *numReq)

				// 在1s内提交请求事件
				wg.Add(1)
				go func() {
					defer wg.Done()
					for range reqNum {
						wg.Add(1)
						interarrivalTime := poissonDist.ExpFloat64() / float64(reqNum)
						time.Sleep(time.Duration(interarrivalTime * float64(time.Second)))
						go func() {
							defer wg.Done()
							processRequest(dataBytes, *modelID, *timeout)
						}()
					}
				}()
			} else {
				log.Printf("Invalid distribution: %s", *distribution)
				return
			}
		}

	}
	wg.Wait()

	sort.Float64s(latencies)
	if len(*latencyPath) != 0 {
		log.Printf("写入文件 %s", *latencyPath)
		// 创建文件
		file, err := os.Create(*latencyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "文件创建失败:", err)
			return
		}
		defer file.Close()

		data := map[string]any{
			"latencies":  latencies,
			"failNum":    failNum,
			"discardNum": discardNum,
		}

		// 将数据转换为 JSON 格式并写入文件
		encoder := json.NewEncoder(file)
		err = encoder.Encode(data)
		if err != nil {
			fmt.Fprintln(os.Stderr, "写入 JSON 失败:", err)
		}
	} else {
		log.Printf("请求成功: %d", sucNum)
		log.Printf("请求失败: %d", failNum)
		log.Printf("被丢弃: %d", discardNum)
		log.Printf("P25: %.6f s", percentile(0.25, latencies))
		log.Printf("平均延迟: %.6f s", average(latencies))
		log.Printf("P50: %.6f s", percentile(0.50, latencies))
		log.Printf("P95: %.6f s", percentile(0.95, latencies))
		log.Printf("P99: %.6f s", percentile(0.99, latencies))
	}

	if len(*cdf_path) != 0 {
		cdf := computeCDF(latencies)
		plotCDF(cdf, *cdf_path)
	}
}

func average(data []float64) float64 {
	if len(data) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range data {
		sum += v
	}
	return sum / float64(len(data))
}
