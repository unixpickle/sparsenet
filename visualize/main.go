package main

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/nfnt/resize"
	"github.com/unixpickle/sparsenet"
)

const ImageSize = 50

func main() {
	rand.Seed(time.Now().UnixNano())
	if len(os.Args) != 5 {
		fmt.Fprintln(os.Stderr, "Usage: visualize <num_in> <num_weights> <spread> <out.png>")
		os.Exit(1)
	}
	inCount, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid num_in:", os.Args[1])
		os.Exit(1)
	}
	weightCount, err := strconv.Atoi(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid num_weights:", os.Args[2])
		os.Exit(1)
	}
	spread, err := strconv.ParseFloat(os.Args[3], 64)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Invalid spread:", os.Args[3])
		os.Exit(1)
	}
	lastLayer := sparsenet.NewLayerUnbiased(1, inCount, 1)
	layer := sparsenet.NewLayer(lastLayer, 1, weightCount, spread)

	outImage := image.NewRGBA(image.Rect(0, 0, ImageSize, ImageSize))
	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			outImage.Set(x, y, color.Gray{Y: 0xff})
		}
	}
	for _, idx := range layer.Indices[0] {
		coord := lastLayer.Coords[idx]
		outImage.Set(int(coord.X*ImageSize), int(coord.Y*ImageSize), color.RGBA{
			R: 0xff,
			A: 0xff,
		})
	}
	outImage.Set(int(layer.Coords[0].X*ImageSize), int(layer.Coords[0].Y*ImageSize), color.RGBA{
		B: 0xff,
		A: 0xff,
	})

	output := resize.Resize(300, 300, outImage, resize.Bilinear)

	f, err := os.Create(os.Args[4])
	if err != nil {
		fmt.Fprintln(os.Stderr, "Failed to create output:", err)
		os.Exit(1)
	}
	defer f.Close()
	png.Encode(f, output)
}
