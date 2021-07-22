package main

/*
	Removed from test package, because execution time is too long.
	A it was problem where was package publihed on AwesomeGO.
*/

import (
	"fmt"
	"math/rand"

	randomforest "github.com/malaschitz/randomForest"
	"github.com/petar/GoMNIST"
)

/*
	Train 10 forests. For every label own forest.
	Results are not better as one forest with 10x trees.
*/
func ExampleMNIST() {
	//read data
	rand.Seed(1)
	TREES := 100
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
	//train 10 forests
	forests := [10]randomforest.Forest{}
	for label := 0; label < 10; label++ {
		forest := randomforest.Forest{}
		x := make([][]float64, size)
		l := make([]int, size)
		for i := 0; i < size; i++ {
			x[i] = make([]float64, xsize)
			for j := 0; j < xsize; j++ {
				x[i][j] = float64(imgs[i][j])
			}
			if int(labels[i]) == label {
				l[i] = 1
			}
		}
		forest.Data = randomforest.ForestData{X: x, Class: l}
		forest.Train(TREES)
		forests[label] = forest
	}

	//read test data
	tsize := 10000
	tlabels, err := GoMNIST.ReadLabelFile("examples/t10k-labels-idx1-ubyte.gz")
	if err != nil {
		panic(err)
	}
	_, _, timgs, err := GoMNIST.ReadImageFile("examples/t10k-images-idx3-ubyte.gz")
	if err != nil {
		panic(err)
	}
	if len(tlabels) != tsize || len(timgs) != tsize {
		panic("Wrong size")
	}
	//calculate difference
	x := make([][]float64, tsize)
	for i := 0; i < tsize; i++ {
		x[i] = make([]float64, xsize)
		for j := 0; j < xsize; j++ {
			x[i][j] = float64(timgs[i][j])
		}
	}

	p := 0
	for i := 0; i < tsize; i++ {
		bestLabel := -1
		bestVote := 0.0
		for label := 0; label < 10; label++ {
			vote := forests[label].Vote(x[i])
			if vote[1] > bestVote {
				bestLabel = label
				bestVote = vote[1]
			}
		}
		if int(tlabels[i]) == bestLabel {
			p++
		} else {
			//fmt.Println(i, tlabels[i], bestVote, bestLabel)
			//writeImage(timgs[i], fmt.Sprintf("img%06d_%d_%d", i, tlabels[i], bestLabel))
		}
	}
	fmt.Printf("Trees: %d Results: %5.1f%%\n", TREES, 100.0*float64(p)/float64(tsize))
	//Output: Trees: 10 Results: 96.0%
}

func main() {
	ExampleMNIST()
}
