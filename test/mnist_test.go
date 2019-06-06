package test

import (
	"fmt"
	"testing"
	"time"

	"github.com/malaschitz/randomForest"
	"github.com/petar/GoMNIST"
)

func TestMNIST_binary_tree(t *testing.T) {
	//read data
	TREES := 10
	size := 60000
	xsize := 28 * 28
	labels, err := GoMNIST.ReadLabelFile("train-labels-idx1-ubyte.gz")
	if err != nil {
		panic(err)
	}
	nrow, ncol, imgs, err := GoMNIST.ReadImageFile("train-images-idx3-ubyte.gz")
	if err != nil {
		panic(err)
	}
	if len(labels) != size || len(imgs) != size {
		panic("Wrong size")
	}
	fmt.Println("Data", nrow, ncol, len(imgs), err)
	//train 10 forests
	forests := [10]randomForest.Forest{}
	for label := 0; label < 10; label++ {
		forest := randomForest.Forest{}
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
		forest.Data = randomForest.ForestData{X: x, Class: l}
		t := time.Now()
		forest.Train(TREES)
		fmt.Println("train", label, time.Since(t))
		forests[label] = forest
	}

	//read test data
	tsize := 10000
	tlabels, err := GoMNIST.ReadLabelFile("t10k-labels-idx1-ubyte.gz")
	if err != nil {
		panic(err)
	}
	_, _, timgs, err := GoMNIST.ReadImageFile("t10k-images-idx3-ubyte.gz")
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
			fmt.Println(i, tlabels[i], bestVote, bestLabel)
			//writeImage(timgs[i], fmt.Sprintf("img%06d_%d_%d", i, tlabels[i], bestLabel))
		}
	}
	fmt.Printf("Trees: %d Pomer: %5.2f%%\n", TREES, 100.0*float64(p)/float64(tsize))
}
