package sparsenet

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var l Layer
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayer)
}

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

// DeserializeLayer deserializes a Layer.
func DeserializeLayer(d []byte) (*Layer, error) {
	var l Layer
	if err := json.Unmarshal(d, &l); err != nil {
		return nil, err
	}
	return &l, nil
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

// Apply applies the layer to an input vector.
func (l *Layer) Apply(in autofunc.Result) autofunc.Result {
	inVec := in.Output()
	output := make(linalg.Vector, len(l.Indices))
	weights := l.Weights.Vector
	var weightIdx int
	for outIdx, indices := range l.Indices {
		for _, inIdx := range indices {
			weight := weights[weightIdx]
			weightIdx++
			output[outIdx] += weight * inVec[inIdx]
		}
	}
	output.Add(l.Biases.Vector)
	return &layerResult{
		Layer:     l,
		OutputVec: output,
		Input:     in,
	}
}

// ApplyR is like Apply, but for RResults.
func (l *Layer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	inVec := in.Output()
	inVecR := in.ROutput()

	output := make(linalg.Vector, len(l.Indices))
	outputR := make(linalg.Vector, len(l.Indices))

	weightsRVar := autofunc.NewRVariable(l.Weights, rv)
	biasesRVar := autofunc.NewRVariable(l.Biases, rv)

	weights := weightsRVar.Output()
	weightsR := weightsRVar.ROutput()
	var weightIdx int
	for outIdx, indices := range l.Indices {
		for _, inIdx := range indices {
			weight := weights[weightIdx]
			weightR := weightsR[weightIdx]
			weightIdx++
			output[outIdx] += weight * inVec[inIdx]
			outputR[outIdx] += weightR*inVec[inIdx] + weight*inVecR[inIdx]
		}
	}
	output.Add(biasesRVar.Output())
	outputR.Add(biasesRVar.ROutput())
	return &layerRResult{
		Layer:      l,
		RWeights:   weightsRVar,
		RBiases:    biasesRVar,
		OutputVec:  output,
		ROutputVec: outputR,
		Input:      in,
	}
}

// Parameters returns the internal parameters of the layer.
func (l *Layer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{l.Weights, l.Biases}
}

// SerializerType returns the unique ID used to serialize
// a Layer with the serializer package.
func (l *Layer) SerializerType() string {
	return "github.com/unixpickle/sparsenet.Layer"
}

// Serialize serializes the layer.
func (l *Layer) Serialize() ([]byte, error) {
	return json.Marshal(l)
}

type layerResult struct {
	Layer     *Layer
	OutputVec linalg.Vector
	Input     autofunc.Result
}

func (l *layerResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *layerResult) Constant(g autofunc.Gradient) bool {
	return l.Layer.Weights.Constant(g) && l.Layer.Biases.Constant(g) &&
		l.Input.Constant(g)
}

func (l *layerResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	l.Layer.Biases.PropagateGradient(upstream, g)
	if gradVec, ok := g[l.Layer.Weights]; ok {
		inputs := l.Input.Output()
		var weightIdx int
		for outIdx, indices := range l.Layer.Indices {
			outDeriv := upstream[outIdx]
			for _, inIdx := range indices {
				gradVec[weightIdx] += inputs[inIdx] * outDeriv
				weightIdx++
			}
		}
	}
	if !l.Input.Constant(g) {
		downstream := make(linalg.Vector, len(l.Input.Output()))
		var weightIdx int
		weights := l.Layer.Weights.Vector
		for outIdx, indices := range l.Layer.Indices {
			outDeriv := upstream[outIdx]
			for _, inIdx := range indices {
				downstream[inIdx] += weights[weightIdx] * outDeriv
				weightIdx++
			}
		}
		l.Input.PropagateGradient(downstream, g)
	}
}

type layerRResult struct {
	Layer      *Layer
	RWeights   *autofunc.RVariable
	RBiases    *autofunc.RVariable
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      autofunc.RResult
}

func (l *layerRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *layerRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *layerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return l.RWeights.Constant(rg, g) && l.RBiases.Constant(rg, g) &&
		l.Input.Constant(rg, g)
}

func (l *layerRResult) PropagateRGradient(u, uR linalg.Vector, rg autofunc.RGradient,
	g autofunc.Gradient) {
	if g == nil {
		g = autofunc.Gradient{}
	}
	l.RBiases.PropagateRGradient(u, uR, rg, g)
	if gradVec, ok := g[l.Layer.Weights]; ok {
		inputs := l.Input.Output()
		var weightIdx int
		for outIdx, indices := range l.Layer.Indices {
			outDeriv := u[outIdx]
			for _, inIdx := range indices {
				gradVec[weightIdx] += inputs[inIdx] * outDeriv
				weightIdx++
			}
		}
	}
	if rgradVec, ok := rg[l.Layer.Weights]; ok {
		inputs := l.Input.Output()
		inputsR := l.Input.ROutput()
		var weightIdx int
		for outIdx, indices := range l.Layer.Indices {
			outDeriv := u[outIdx]
			outDerivR := uR[outIdx]
			for _, inIdx := range indices {
				rgradVec[weightIdx] += inputs[inIdx]*outDerivR + inputsR[inIdx]*outDeriv
				weightIdx++
			}
		}
	}
	if !l.Input.Constant(rg, g) {
		downstream := make(linalg.Vector, len(l.Input.Output()))
		downstreamR := make(linalg.Vector, len(downstream))
		var weightIdx int
		weights := l.RWeights.Output()
		weightsR := l.RWeights.ROutput()
		for outIdx, indices := range l.Layer.Indices {
			outDeriv := u[outIdx]
			outDerivR := uR[outIdx]
			for _, inIdx := range indices {
				downstream[inIdx] += weights[weightIdx] * outDeriv
				downstreamR[inIdx] += weightsR[weightIdx]*outDeriv +
					weights[weightIdx]*outDerivR
				weightIdx++
			}
		}
		l.Input.PropagateRGradient(downstream, downstreamR, rg, g)
	}
}
