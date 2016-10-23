// Command demo runs an MNIST classification demo.
//
// Must of the code was lifted from
// https://github.com/unixpickle/weakai/blob/c864f75eebf09470d1c0bd1029a956fc04e2b4c0/demos/neuralnet/image_demos/mnist/main.go.
package main

import (
	"log"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/sparsenet"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize = 300
	LabelCount = 10
	StepSize   = 1e-2
	BatchSize  = 20
)

func main() {
	training := mnist.LoadTrainingDataSet()
	crossValidation := mnist.LoadTestingDataSet()

	net := createNet(training)

	trainingSamples := dataSetSamples(training)
	gradienter := &neuralnet.BatchRGradienter{
		Learner:  net.BatchLearner(),
		CostFunc: neuralnet.MeanSquaredCost{},
	}
	rmsGrad := &sgd.RMSProp{Gradienter: gradienter}

	sgd.SGDInteractive(rmsGrad, trainingSamples, StepSize, BatchSize, func() bool {
		log.Println("Printing score...")
		printScore("Cross", net, crossValidation)
		log.Println("Running training round...")
		return true
	})
}

func createNet(d mnist.DataSet) neuralnet.Network {
	layer1 := sparsenet.NewLayerUnbiased(28*28, 1000, 300)
	layer2 := sparsenet.NewLayer(layer1, 2000, 100, 1)
	layer3 := sparsenet.NewLayer(layer2, 3000, 50, 0.8)
	layer4 := sparsenet.NewLayer(layer3, 5000, 30, 0.3)
	layer5 := sparsenet.NewLayer(layer4, 2000, 70, 0.5)
	layer6 := sparsenet.NewLayer(layer5, 500, 300, 1)
	outLayer := &neuralnet.DenseLayer{
		InputCount:  500,
		OutputCount: 10,
	}
	outLayer.Randomize()
	return neuralnet.Network{
		layer1,
		neuralnet.ReLU{},
		layer2,
		neuralnet.ReLU{},
		layer3,
		neuralnet.ReLU{},
		layer4,
		neuralnet.ReLU{},
		layer5,
		neuralnet.ReLU{},
		layer6,
		neuralnet.ReLU{},
		outLayer,
		&neuralnet.LogSoftmaxLayer{},
	}
}

func printScore(prefix string, n neuralnet.Network, d mnist.DataSet) {
	classifier := func(v []float64) int {
		result := n.Apply(&autofunc.Variable{v})
		return outputIdx(result)
	}
	correctCount := d.NumCorrect(classifier)
	histogram := d.CorrectnessHistogram(classifier)
	log.Printf("%s: %d/%d - %s", prefix, correctCount, len(d.Samples), histogram)
}

func outputIdx(r autofunc.Result) int {
	out := r.Output()
	var maxIdx int
	var max float64
	for i, x := range out {
		if i == 0 || x > max {
			max = x
			maxIdx = i
		}
	}
	return maxIdx
}

func dataSetSamples(d mnist.DataSet) sgd.SampleSet {
	labelVecs := d.LabelVectors()
	inputVecs := d.IntensityVectors()
	return neuralnet.VectorSampleSet(vecVec(inputVecs), vecVec(labelVecs))
}

func vecVec(f [][]float64) []linalg.Vector {
	res := make([]linalg.Vector, len(f))
	for i, x := range f {
		res[i] = x
	}
	return res
}
