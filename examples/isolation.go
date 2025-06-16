package main

import (
	"fmt"
	"math/rand"

	randomforest "github.com/blue-agency/randomForest"
	"github.com/petar/GoMNIST"
)

func main() {
	rand.Seed(1)
	TREES := 1000
	size := 60000
	xsize := 28 * 28
	labels, err := GoMNIST.ReadLabelFile("examples/train-labels-idx1-ubyte.gz")
	if err != nil {
		panic(err)
	}
	_, _, imgs, err := GoMNIST.ReadImageFile("examples/train-images-idx3-ubyte.gz")
	if err != nil {
		panic(err)
	}
	if len(labels) != size || len(imgs) != size {
		panic("Wrong size")
	}
	//train
	forest := randomforest.Forest{}
	x := make([][]float64, size)
	l := make([]int, size)
	for i := 0; i < size; i++ {
		x[i] = make([]float64, xsize)
		for j := 0; j < xsize; j++ {
			x[i][j] = float64(imgs[i][j])
			l[i] = int(labels[i])
		}
	}
	forest.Data = randomforest.ForestData{X: x, Class: l}
	forest.MaxDepth = 30
	forest.Train(TREES)

	//ISOLATION
	isolations, mean, stddev := forest.IsolationForest()
	for i, d := range isolations {
		if d < (mean - 1.6*stddev) {
			fmt.Println(i, (d-mean)/stddev)
		}
	}
	//
}
