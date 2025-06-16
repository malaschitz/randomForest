package main

import (
	"fmt"
	"math/rand"
	"testing"

	randomforest "github.com/blue-agency/randomForest"
	"github.com/blue-agency/randomForest/tests/generator"
)

// DeepForest create a bunch of subforests. Results of these subforest are a new inputs (new attributes for original dataset).
// It has reason for using only with special examples !
func TestDeepForest(t *testing.T) {
	rand.Seed(1)
	n := 10000
	features := 20
	classes := 2

	f := randomforest.Forest{}
	data, res := generator.CreateDataset(n, features, classes)
	forestData := randomforest.ForestData{X: data, Class: res}
	f.Data = forestData
	dForest := f.BuildDeepForest()
	dForest.Train(20, 100, 1000)

	//deep test
	s := 0
	rand.Seed(2)
	data, res = generator.CreateDataset(n, features, classes)
	for i := 0; i < n; i++ {
		vote := dForest.Vote(data[i])
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
	}
	fmt.Println("try", n, "times")
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))

	//compare with normal forest
	fmt.Println("Normal Forest ---------------------------------------------------")
	dForest.Forest.Train(dForest.ForestDeep.NTrees)
	s = 0
	rand.Seed(2)
	data, res = generator.CreateDataset(n, features, classes)
	for i := 0; i < n; i++ {
		vote := dForest.Forest.Vote(data[i])
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
	}
	fmt.Println("try", n, "times")
	fmt.Printf("Correct:        %5.2f %%\n", float64(s)*100/float64(n))
}
