package sparsenet

import (
	"math/rand"

	"github.com/unixpickle/autofunc"
)

// A chooser selects numbers in a biased fashion.
type chooser struct {
	Indices     []int
	Weights     []float64
	TotalWeight float64
}

func newUniformChooser(count int) *chooser {
	res := &chooser{
		Indices:     make([]int, count),
		Weights:     make([]float64, count),
		TotalWeight: float64(count),
	}
	for i := 0; i < count; i++ {
		res.Indices[i] = i
		res.Weights[i] = 1
	}
	return res
}

func newSpatialChooser(in []*Coordinate, c *Coordinate, spread float64) *chooser {
	invDist := make([]float64, len(in))
	indices := make([]int, len(in))
	for i, inCoord := range in {
		invDist[i] = 1 / Distance(inCoord, c)
		indices[i] = i
	}
	sm := autofunc.Softmax{Temperature: spread}
	softmaxed := sm.Apply(&autofunc.Variable{Vector: invDist}).Output()
	return &chooser{
		Indices:     indices,
		Weights:     softmaxed,
		TotalWeight: 1,
	}
}

// Choose picks a random index in a biased fashion.
// The returned index is removed from the chooser.
func (c *chooser) Choose() int {
	n := rand.Float64() * c.TotalWeight
	for i, w := range c.Weights {
		n -= w
		if n < 0 || i == len(c.Weights)-1 {
			res := c.Indices[i]
			c.Indices[i] = c.Indices[len(c.Indices)-1]
			c.Weights[i] = c.Weights[len(c.Weights)-1]
			c.Indices = c.Indices[:len(c.Indices)-1]
			c.Weights = c.Weights[:len(c.Weights)-1]
			c.TotalWeight -= w
			return res
		}
	}
	panic("code unreachable")
}
