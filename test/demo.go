package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/malaschitz/randomForest"
)

func main() {
	rand.Seed(1)
	forest := randomForest.Forest{}
	data, res := createDataset(10000, 6)
	forestData := randomForest.ForestData{X: data, Results: res}
	forest.Data = forestData
	t := time.Now()
	forest.Train(999)
	fmt.Println("train", time.Since(t))
	//test
	s := 0
	sw := 0
	n := 10000
	t = time.Now()
	rand.Seed(2)
	datat, rest := createDataset(n, 6)
	for i := 0; i < n; i++ {
		vote := forest.Vote(datat[i])
		if (vote > 0.5 && rest[i]) || (vote < 0.5 && !rest[i]) {
			s++
		}
		//
		wvote := forest.WeightVote(datat[i])
		if (wvote > 0.5 && rest[i]) || (wvote < 0.5 && !rest[i]) {
			sw++
		}
	}
	fmt.Println("try", n, "times", time.Since(t))
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))
	fmt.Printf("Weight Correct: %5.2f %%\n", float64(sw)*100/float64(n))
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

	s = s + 0*(rand.Float64()-.5)
	return s > 0
}
