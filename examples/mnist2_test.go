package main

import (
	"fmt"
	"image"
	"image/png"
	"log"
	"math/rand"
	"os"

	randomforest "github.com/malaschitz/randomForest"
	"github.com/petar/GoMNIST"
)

/*
	Train 1 forest for all labels.
*/
func ExampleMNIST2() {
	//read data
	rand.Seed(1)
	TREES := 100
	size := 60000
	xsize := 28 * 28
	labels, err := GoMNIST.ReadLabelFile("train-labels-idx1-ubyte.gz")
	if err != nil {
		panic(err)
	}
	_, _, imgs, err := GoMNIST.ReadImageFile("train-images-idx3-ubyte.gz")
	if err != nil {
		panic(err)
	}
	if len(labels) != size || len(imgs) != size {
		panic("Wrong size")
	}
	//train 1 forest
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
	forest.Train(TREES)

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
	x = make([][]float64, tsize)
	for i := 0; i < tsize; i++ {
		x[i] = make([]float64, xsize)
		for j := 0; j < xsize; j++ {
			x[i][j] = float64(timgs[i][j])
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
		} else {
			//fmt.Println(i, tlabels[i], bestI, bestV)
			//writeImage(timgs[i], fmt.Sprintf("img%06d_%d_%d", i, tlabels[i], bestLabel))
		}
	}
	fmt.Printf("Trees: %d Results: %5.0f%%\n", TREES, 100.0*float64(p)/float64(tsize))
	//Output: Trees: 100 Results:    96%
}

func writeImage(data []byte, name string) {
	img := image.NewGray(image.Rect(0, 0, 28, 28))
	img.Pix = data
	out, err := os.Create("./" + name + ".png")
	err = png.Encode(out, img)
	if err != nil {
		log.Fatal(err)
	}
}
