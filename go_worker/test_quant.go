package main

import (
	"math"
	"math/rand"
	"sort"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func generateVectors(num int, dim int) [][]float32 {
	out := make([][]float32, num)
	for i := range out {
		vec := make([]float32, dim)
		for j := range vec {
			vec[j] = rand.Float32()*2 - 1
		}
		out[i] = vec
	}
	return out
}

func computeMSEPSNR(original, recovered [][]float32) (float64, float64) {
	mse := 0.0
	count := 0
	for i := range original {
		for j := range original[i] {
			diff := float64(original[i][j] - recovered[i][j])
			mse += diff * diff
			count++
		}
	}
	mse /= float64(count)
	psnr := 10 * math.Log10(1.0/mse)
	return mse, psnr
}

func plotCDF(values []float64, title, outFile string) error {
	sort.Float64s(values)
	pts := make(plotter.XYs, len(values))
	for i, v := range values {
		pts[i].X = v
		pts[i].Y = float64(i+1) / float64(len(values))
	}

	p := plot.New()
	p.Title.Text = title
	p.X.Label.Text = "Value"
	p.Y.Label.Text = "CDF"
	plotutil.AddLinePoints(p, pts)
	return p.Save(6*vg.Inch, 4*vg.Inch, outFile)
}

// func main() {
// 	rand.Seed(time.Now().UnixNano())

// 	numTests := 100
// 	numVec := 1
// 	dim := 3 * 224 * 224

// 	qTimes := make([]float64, numTests)
// 	dqTimes := make([]float64, numTests)
// 	mses := make([]float64, numTests)
// 	psnrs := make([]float64, numTests)

// 	for i := 0; i < numTests; i++ {
// 		vectors := generateVectors(numVec, dim)

// 		start := time.Now()
// 		dataPtr, _, releaseQuant, err := QuantizeVectorsFBGEMMUnsafe(vectors)
// 		qTimes[i] = float64(time.Since(start).Microseconds()) / 1000.0
// 		if err != nil {
// 			log.Fatalf("Quantize failed at %d: %v", i, err)
// 		}

// 		dims := make([]int32, numVec)
// 		for j := range dims {
// 			dims[j] = int32(dim)
// 		}

// 		start = time.Now()
// 		recovered, releaseDequant, err := DequantizeVectorsFBGEMMUnsafe(dataPtr, dims)
// 		dqTimes[i] = float64(time.Since(start).Microseconds()) / 1000.0
// 		if err != nil {
// 			log.Fatalf("Dequantize failed at %d: %v", i, err)
// 		}

// 		mse, psnr := computeMSEPSNR(vectors, recovered)
// 		mses[i] = mse
// 		psnrs[i] = psnr

// 		// ✅ 数据全部使用完毕后再释放
// 		releaseDequant()
// 		releaseQuant()
// 	}

// 	os.MkdirAll("../test/plots", 0755)
// 	plotCDF(qTimes, "Quantize Time (ms)", "plots/q_times.png")
// 	plotCDF(dqTimes, "Dequantize Time (ms)", "plots/dq_times.png")
// 	plotCDF(mses, "MSE", "plots/mse.png")
// 	plotCDF(psnrs, "PSNR", "plots/psnr.png")

// 	fmt.Println("✅ Done. Plots saved in ./plots")
// }
