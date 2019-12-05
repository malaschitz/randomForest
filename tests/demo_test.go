package main

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/malaschitz/randomForest"
	"github.com/malaschitz/randomForest/tests/generator"
)

//TestDemo simple Random Foresr exaple
func TestDemo(t *testing.T) {
	rand.Seed(1)
	n := 100
	features := 20
	classes := 2
	trees := 100

	forest := randomForest.Forest{}
	data, res := generator.CreateDataset(n, features, classes)
	forestData := randomForest.ForestData{X: data, Class: res}
	forest.Data = forestData
	forest.Train(trees)
	//test
	s := 0
	sw := 0

	rand.Seed(2)
	data, res = generator.CreateDataset(n, features, classes)
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
		}

	}
	fmt.Println("try", n, "times")
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))
	fmt.Printf("Weight Correct: %5.2f %%\n", float64(sw)*100/float64(n))
	forest.PrintFeatureImportance()

}
