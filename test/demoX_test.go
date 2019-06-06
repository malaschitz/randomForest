package test

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/malaschitz/randomForest"
)

func TestExtrRF(t *testing.T) {
	rand.Seed(1)
	n := 10000
	features := 30
	classes := 10
	trees := 100

	forest := randomForest.Forest{}
	data, res := createDataset(n, features, classes)
	forestData := randomForest.ForestData{X: data, Class: res}
	forest.Data = forestData
	forest.TrainX(trees)
	//test
	s := 0
	sw := 0

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
	fmt.Println("try", n, "times")
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))
	fmt.Printf("Weight Correct: %5.2f %%\n", float64(sw)*100/float64(n))
	forest.PrintFeatureImportance()

}
