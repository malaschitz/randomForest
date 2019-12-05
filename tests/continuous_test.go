package main

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/malaschitz/randomForest"
)

func TestContinuous(t *testing.T) {
	rand.Seed(1)
	n := 1000
	forest := randomForest.Forest{}
	s := 0
	for i := 0; i < n-1; i++ {
		data := []float64{rand.Float64(), rand.Float64()}
		res := 0
		if data[0]+data[1] > 1 {
			res = 1
		}
		forest.AddDataRow(data, res, 100, 10, 200)

		data = []float64{rand.Float64(), rand.Float64()}
		res = 0
		if data[0]+data[1] > 1 {
			res = 1
		}

		vote := forest.Vote(data)
		bestV := 0.0
		bestI := -1
		for j, v := range vote {
			if v > bestV {
				bestV = v
				bestI = j
			}
		}
		if bestI == res {
			s++
			fmt.Println("OK TEST", i, "VOTE", vote, data, "=", res, "\ts=", s, float64(s)/float64(i+1))
		} else {
			fmt.Println("WRONG TEST", i, "VOTE", vote, data, "=", res, "\ts=", s, float64(s)/float64(i+1))

		}
	}
	forest.PrintFeatureImportance()
	fmt.Println("Result:", float64(s)/float64(n-1))
}
