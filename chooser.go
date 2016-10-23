package sparsenet

import (
	"math/rand"
	"sort"
)

// A chooser selects numbers in a biased fashion.
type chooser struct {
	Indices []int
}

func newUniformChooser(count int) *chooser {
	return &chooser{Indices: rand.Perm(count)}
}

func newSpatialChooser(in []*Coordinate, c *Coordinate, spread float64) *chooser {
	distances := make([]float64, len(in))
	indices := make([]int, len(in))
	for i, inCoord := range in {
		distances[i] = Distance(inCoord, c) + rand.NormFloat64()*spread
		indices[i] = i
	}
	sorter := indexSorter{Indices: indices, Values: distances}
	sort.Sort(&sorter)
	return &chooser{Indices: indices}
}

// Choose picks a random index in a biased fashion.
// The returned index is removed from the chooser.
func (c *chooser) Choose() int {
	res := c.Indices[0]
	c.Indices = c.Indices[1:]
	return res
}

type indexSorter struct {
	Indices []int
	Values  []float64
}

func (i *indexSorter) Len() int {
	return len(i.Indices)
}

func (i *indexSorter) Swap(j, k int) {
	i.Indices[j], i.Indices[k] = i.Indices[k], i.Indices[j]
	i.Values[j], i.Values[k] = i.Values[k], i.Values[j]
}

func (i *indexSorter) Less(j, k int) bool {
	return i.Values[j] < i.Values[k]
}
