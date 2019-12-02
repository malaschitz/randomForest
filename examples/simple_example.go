package main

import (
	"fmt"
	"math/rand"

	"github.com/malaschitz/randomForest"
)

func main() {
	xData := [][]float64{}
	yData := []int{}
	for i := 0; i < 1000; i++ {
		x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
		y := int(x[0] + x[1] + x[2] + x[3])
		xData = append(xData, x)
		yData = append(yData, y)
	}
	forest := randomForest.Forest{}
	forestData := randomForest.ForestData{X: xData, Class: yData}
	forest.Data = forestData
	forest.Train(1000)
	//test
	vote := forest.Vote([]float64{0.1, 0.1, 0.1, 0.1})
	fmt.Println("Vote", vote)
	if vote[0] < vote[1] || vote[0] < vote[2] || vote[0] < vote[3] {
		panic("Wrong Machine Learning !")
	}

	vote = forest.Vote([]float64{0.9, 0.9, 0.9, 0.9})
	fmt.Println("Vote", vote)
	if vote[3] < vote[0] || vote[3] < vote[1] || vote[3] < vote[2] {
		panic("Wrong Machine Learning !")
	}

}
