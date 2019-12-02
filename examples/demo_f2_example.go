package main

import (
	"fmt"
	"math/rand"

	"github.com/malaschitz/randomForest"
	"github.com/malaschitz/randomForest/example/generator"
)

/*
	Example of data when you need create more features. It is often far better method than DeepForest or NN.
*/
func main() {
	rand.Seed(1)
	n := 1000
	features := 20
	classes := 2
	trees := 10000

	forest := randomForest.Forest{}
	data, res := generator.CreateDataset(n, features, classes)
	for j := 0; j < n; j++ {
		createFeatures2(&data[j])
	}
	forestData := randomForest.ForestData{X: data, Class: res}
	forest.Data = forestData
	forest.Train(trees)
	//test
	s := 0
	sw := 0

	rand.Seed(2)
	data, res = generator.CreateDataset(n, features, classes)
	for i := 0; i < n; i++ {
		createFeatures2(&data[i])
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

func createFeatures2(f *[]float64) {
	flen := len(*f)
	ar := *f
	for i := 0; i < flen; i++ {
		for j := i; j < flen; j++ {
			a := ar[i] * ar[j]
			*f = append(*f, a)
			a = ar[i] + ar[j]
			*f = append(*f, a)
		}
	}
}
