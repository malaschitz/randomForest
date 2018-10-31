package main

import (
	"fmt"
	"math/rand"

	"github.com/malaschitz/randomForest"
)

func main() {
	rand.Seed(1)
	forest := randomForest.Forest{}
	data, res := createDataset(2000, 12)
	forestData := randomForest.ForestData{X: data, Results: res}
	forest.Data = forestData
	forest.MAttrs = 12
	forest.Train(999)

	//test
	s := 0
	n := 1000
	for i := 0; i < n; i++ {
		datat, rest := createDataset(n, 12)
		voteT, voteF := forest.Vote(datat[i])
		if (voteT > voteF && rest[i]) || (voteT < voteF && !rest[i]) {
			s++
		} else {
			//fmt.Println(i, rest[i], voteT, voteF)
		}
	}
	fmt.Printf("Correct: %5.2f %%\n", float64(s)*100/float64(n))
	forest.PrintFeatureImportance()
}

func createDataset(N, A int) ([][]float64, []bool) {
	data := make([][]float64, 0)
	res := make([]bool, 0)
	for i := 0; i < N; i++ {
		d := make([]float64, A)
		for j := 0; j < A; j++ {
			d[j] = rand.Float64()
		}
		data = append(data, d)
		v := classit(d)
		res = append(res, v)
	}
	return data, res
}

func classit(a []float64) bool {
	s := 0.0
	for i := 0; i < len(a); i++ {
		k := float64((i + 2) / 2)
		if i%2 == 0 {
			s = s + k*a[i]
		} else {
			s = s - k*a[i]
		}
	}
	s = s + 10*(rand.Float64()-.5)
	return s > 0
}
