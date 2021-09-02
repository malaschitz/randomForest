package main

import (
	"fmt"
	"math/rand"

	randomforest "github.com/malaschitz/randomForest"
	"github.com/malaschitz/randomForest/examples/img"
	"github.com/petar/GoMNIST"
)

/*
	Using boruta for mnist

	With threshold 5% was selected 495 features from 784.
	Result of random forest was the same - 96.2% as in minst.go

	With threshold 10% was selected 493 features from 784 with the same result.

*/
func main() {
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
	x := make([][]float64, size)
	l := make([]int, size)
	for i := 0; i < size; i++ {
		x[i] = make([]float64, xsize)
		for j := 0; j < xsize; j++ {
			x[i][j] = float64(imgs[i][j])
			l[i] = int(labels[i])
		}
	}
	//sample for testing
	//x, l = sample(x, l, 200)
	//
	borutaFeatuters, stats := randomforest.BorutaDefault(x, l)
	//borutaFeatuters := randomforest.Boruta(x, l, 100, 20, 0.05, true, true)
	fmt.Println("Stats", stats)
	image := make([]byte, xsize)
	for _, v := range borutaFeatuters {
		image[v] = 255
	}
	//save redsults as image
	img.WriteImage(image, "boruta")
	//
	// try forest with selected features
	//
	xSmall := make([][]float64, len(x))
	for i := 0; i < len(x); i++ {
		xSmall[i] = make([]float64, len(borutaFeatuters))
		for j := 0; j < len(borutaFeatuters); j++ {
			xSmall[i][j] = x[i][borutaFeatuters[j]]
		}
	}
	forest := randomforest.Forest{}
	forest.Data = randomforest.ForestData{X: xSmall, Class: l}
	forest.Train(100)

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
	x = make([][]float64, tsize)
	for i := 0; i < tsize; i++ {
		x[i] = make([]float64, len(borutaFeatuters))
		for j := 0; j < len(borutaFeatuters); j++ {
			x[i][j] = float64(timgs[i][borutaFeatuters[j]])
		}
	}
	p := 0
	for i := 0; i < tsize; i++ {
		vote := forest.Vote(x[i])
		bestI := -1
		bestV := 0.0
		for j, v := range vote {
			if v > bestV {
				bestV = v
				bestI = j
			}
		}
		if int(tlabels[i]) == bestI {
			p++
		}
	}
	fmt.Printf("Trees: %d Results: %5.1f%%\n", 100, 100.0*float64(p)/float64(tsize))
	// Selected 495 features from 784
	// Trees: 100 Results:  96.2%
}

//create samples from data
func sample(x [][]float64, y []int, count int) (xx [][]float64, yy []int) {
	xx = make([][]float64, count)
	yy = make([]int, count)
	for i := 0; i < count; i++ {
		k := rand.Intn(len(x))
		xx[i] = x[k]
		yy[i] = y[k]
	}
	return
}
