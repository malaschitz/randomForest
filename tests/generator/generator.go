// Package generator is creting testing data for machine learning
package generator

import (
	"math/rand"
)

// CreateDataset create test data with parameters N - size of dataset F - number of features Classes - number of classes (results).
//
func CreateDataset(N, Features, Classes int) ([][]float64, []int) {
	data := make([][]float64, 0)
	res := make([]int, 0)
	for i := 0; i < N; i++ {
		d := make([]float64, Features)
		for j := 0; j < Features; j++ {
			d[j] = rand.Float64()
		}
		data = append(data, d)
		v := ClassIt(d, Classes)
		res = append(res, v)
	}
	return data, res
}

// ClassIt calculate class for data.
// It is a little bit chaotic generator - to be hard for machine learning
func ClassIt(a []float64, c int) int {
	s := make([]float64, c)
	for i := 0; i < len(a); i++ {
		k := float64(i)/float64(c) + 1
		s[i%c] += k * a[i]
	}
	cbest := -1
	sbest := 0.0
	for i := 0; i < c; i++ {
		if s[i] > sbest {
			sbest = s[i]
			cbest = i
		}
	}
	return cbest
}
