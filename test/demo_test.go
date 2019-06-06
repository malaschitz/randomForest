package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/malaschitz/randomForest"
)

func main() {
	rand.Seed(1)
	n := 10000
	features := 30
	classes := 10
	trees := 100

	forest := randomForest.Forest{}
	data, res := createDataset(n, features, classes)
	//fmt.Println("data     ", data)
	//fmt.Println("classes  ", res)
	forestData := randomForest.ForestData{X: data, Class: res}
	forest.Data = forestData
	t := time.Now()
	forest.Train(trees)
	//fmt.Println("train", time.Since(t))
	//test
	s := 0
	sw := 0

	t = time.Now()
	rand.Seed(2)
	data, res = createDataset(n, features, classes)
	for i := 0; i < n; i++ {
		vote := forest.Vote(data[i])
		bestV := 0.0
		bestI := -1
		for j, v := range vote {
			if v > bestV {
				bestV = v
				bestI = j
			}
		}
		if bestI == res[i] {
			s++
		} else {
			//fmt.Println("TEST", i, "VOTE", vote, data[i], "=", res[i], "\ts=", s)
		}

		//
		vote = forest.WeightVote(data[i])
		bestV = 0.0
		bestI = -1
		for j, v := range vote {
			if v > bestV {
				bestV = v
				bestI = j
			}
		}
		if bestI == res[i] {
			sw++
		} else {
			//fmt.Println("TEST", i, "VOTE", vote, data[i], "=", res[i], "\ts=", sw)
		}

	}
	fmt.Println("try", n, "times", time.Since(t))
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))
	fmt.Printf("Weight Correct: %5.2f %%\n", float64(sw)*100/float64(n))
	forest.PrintFeatureImportance()

}

func createDataset(N, Features, Classes int) ([][]float64, []int) {
	data := make([][]float64, 0)
	res := make([]int, 0)
	for i := 0; i < N; i++ {
		d := make([]float64, Features)
		for j := 0; j < Features; j++ {
			d[j] = rand.Float64()
		}
		data = append(data, d)
		v := classit(d, Classes)
		res = append(res, v)
	}
	return data, res
}

func classit(a []float64, c int) int {
	s := make([]float64, c)
	for i := 0; i < len(a); i++ {
		k := float64(i/c) + 1
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
	//fmt.Println("class it", a, c, cbest, sbest, s)
	return cbest
}
