package sparsenet

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestNetwork(t *testing.T) {
	layer1 := NewLayerUnbiased(3, 4, 2)
	layer2 := NewLayer(layer1, 3, 2, 1.0)
	net := neuralnet.Network{
		layer1,
		neuralnet.HyperbolicTangent{},
		layer2,
		neuralnet.HyperbolicTangent{},
	}

	input := &autofunc.Variable{Vector: []float64{0.5, -0.3, 0.9}}
	params := append([]*autofunc.Variable{input}, net.Parameters()...)
	rvec := autofunc.RVector{}
	for _, v := range params {
		rvec[v] = make(linalg.Vector, len(v.Vector))
		for i := range rvec[v] {
			rvec[v][i] = rand.NormFloat64()
		}
	}

	checker := functest.RFuncChecker{
		F:     net,
		Vars:  params,
		Input: input,
		RV:    rvec,
	}
	checker.FullCheck(t)
}
