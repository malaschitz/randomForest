package main

import (
	"fmt"
	"math/rand"
	"sort"

	randomforest "github.com/blue-agency/randomForest"
	"github.com/petar/GoMNIST"
)

func main() {
	rand.Seed(1)
	for l := 0; l < 10; l++ {
		TREES := 100000
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
		//train
		x := make([][]float64, 0)
		mp := map[int]int{}
		for i := 0; i < size; i++ {
			if int(labels[i]) == l {
				xx := make([]float64, xsize)
				for j := 0; j < xsize; j++ {
					xx[j] = float64(imgs[i][j])
				}
				x = append(x, xx)
				mp[len(x)] = i
			}
		}

		fmt.Println(len(x), "x", len(x[0]))
		//
		forest := randomforest.IsolationForest{X: x}
		forest.Train(TREES)
		// results
		a := make([]int, len(x))
		for i := 0; i < len(x); i++ {
			a[i] = i
		}
		sort.Slice(a, func(i, j int) bool {
			ai := forest.Results[a[i]]
			aj := forest.Results[a[j]]
			mi := float64(ai[1]) / (float64(ai[0]) + 0.00001)
			mj := float64(aj[1]) / (float64(aj[0]) + 0.00001)
			return mi < mj
		})
		fmt.Println("-------------", l)
		for i := 0; i < 10; i++ {
			fmt.Printf("%3d %5d %3d %3d %5.2f\n", i, mp[a[i]], forest.Results[a[i]][0], forest.Results[a[i]][1],
				float64(forest.Results[a[i]][1])/(float64(forest.Results[a[i]][0])+0.00001))
		}
	}
}
