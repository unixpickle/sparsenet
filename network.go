package sparsenet

import (
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
)

// Coordinate represents a neuron's location in 3-space.
// Neurons will typically be stored in a 1x1x1 cube.
type Coordinate struct {
	X float64
	Y float64
	Z float64
}

// Distance computes the Euclidean distance between two
// neural coordinates.
func Distance(c1, c2 *Coordinate) float64 {
	return math.Sqrt(math.Pow(c1.X-c2.X, 2) + math.Pow(c1.Y-c2.Y, 2) +
		math.Pow(c1.Z-c2.Z, 2))
}

// A Layer is a sparsely-connected layer which is arranged
// in 3-space.
type Layer struct {
	// Coords stores the coordinates of the output neurons.
	Coords []*Coordinate

	// Indices indices the inputs for each output neuron.
	Indices [][]int

	// Weights stores one weight per entry in Indices,
	// wrapped such that all the weights for an output neuron
	// are all consecutive.
	Weights *autofunc.Variable

	// Biases stores one bias value per output neuron.
	Biases *autofunc.Variable
}

// NewLayerUnbiased creates a Layer with randomly placed
// sparse connections.
// This is for creating the first layer of a network.
func NewLayerUnbiased(inCount, outCount, connCount int) *Layer {
	if connCount > inCount || connCount > outCount {
		panic("cannot mave more connections than neurons")
	}
	res := &Layer{
		Coords:  make([]*Coordinate, outCount),
		Indices: make([][]int, outCount),
		Weights: &autofunc.Variable{Vector: make([]float64, outCount*connCount)},
		Biases:  &autofunc.Variable{Vector: make([]float64, outCount)},
	}
	weightStddev := 1 / math.Sqrt(float64(connCount))
	for i := range res.Weights.Vector {
		res.Weights.Vector[i] = rand.NormFloat64() * weightStddev
	}
	for i := range res.Biases.Vector {
		res.Biases.Vector[i] = rand.NormFloat64()
	}
	for i := range res.Coords {
		res.Coords[i] = &Coordinate{
			X: rand.Float64(),
			Y: rand.Float64(),
			Z: rand.Float64(),
		}
	}
	for i := range res.Indices {
		ch := newUniformChooser(inCount)
		res.Indices[i] = make([]int, connCount)
		for j := 0; j < connCount; j++ {
			res.Indices[i][j] = ch.Choose()
		}
	}
	return res
}

// NewLayer creates a Layer with connections which are
// statistically based on the input neural coordinates.
//
// The spread argument specifies how spread out the
// connections should be.
// A low value (e.g. 1) indices that connections should
// be somewhat localized, while a higher value (e.g. 5)
// indices that the connections should be more random.
func NewLayer(in *Layer, outCount, connCount int, spread float64) *Layer {
	if connCount > len(in.Coords) || connCount > outCount {
		panic("cannot mave more connections than neurons")
	}
	res := &Layer{
		Coords:  make([]*Coordinate, outCount),
		Indices: make([][]int, outCount),
		Weights: &autofunc.Variable{Vector: make([]float64, outCount*connCount)},
		Biases:  &autofunc.Variable{Vector: make([]float64, outCount)},
	}
	weightStddev := 1 / math.Sqrt(float64(connCount))
	for i := range res.Weights.Vector {
		res.Weights.Vector[i] = rand.NormFloat64() * weightStddev
	}
	for i := range res.Biases.Vector {
		res.Biases.Vector[i] = rand.NormFloat64()
	}
	for i := range res.Coords {
		res.Coords[i] = &Coordinate{
			X: rand.Float64(),
			Y: rand.Float64(),
			Z: rand.Float64(),
		}
	}
	for i := range res.Indices {
		ch := newSpatialChooser(in.Coords, res.Coords[i], spread)
		res.Indices[i] = make([]int, connCount)
		for j := 0; j < connCount; j++ {
			res.Indices[i][j] = ch.Choose()
		}
	}
	return res
}
