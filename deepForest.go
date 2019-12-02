package randomForest

import (
	"fmt"
	"math/rand"
)

type DeepForest struct {
	Forest         *Forest
	ForestDeep     Forest
	Groves         []Forest
	NGroves        int
	NFeatures      int
	NTrees         int
	RandomFeatures [][]int
	ResultFeatures [][]float64
	Results        []float64
}

func (forest *Forest) BuildDeepForest() DeepForest {
	forest.defaults()
	df := DeepForest{
		Forest:    forest,
		NFeatures: forest.MFeatures,
	}
	return df
}

func (dForest *DeepForest) Train(groves int, trees int, deepTrees int) {
	dForest.NGroves = groves
	dForest.NTrees = trees
	//build and train groves
	dForest.Groves = []Forest{}
	dForest.RandomFeatures = make([][]int, 0)
	dForest.ResultFeatures = make([][]float64, dForest.Forest.NSize)
	dForest.Results = make([]float64, 0)
	for i := 0; i < dForest.Forest.NSize; i++ {
		dForest.ResultFeatures[i] = make([]float64, 0)
	}
	{
		for i := 0; i < dForest.NGroves; i++ {
			//create grove
			grove := Forest{}
			x := make([][]float64, 0)
			perm := rand.Perm(dForest.Forest.Features)
			perm = perm[:dForest.NFeatures]
			dForest.RandomFeatures = append(dForest.RandomFeatures, perm)
			for _, datax := range dForest.Forest.Data.X {
				pdatax := make([]float64, dForest.NFeatures)
				for j, p := range perm {
					pdatax[j] = datax[p]
				}
				x = append(x, pdatax)
			}
			grove.Data = ForestData{X: x, Class: dForest.Forest.Data.Class}
			//train grove
			grove.Train(dForest.NTrees)
			dForest.Groves = append(dForest.Groves, grove)
			//store results
			p := 0
			for j, datax := range x {
				vote := grove.Vote(datax)
				dForest.ResultFeatures[j] = append(dForest.ResultFeatures[j], vote...)
				bestI := -1
				bestV := 0.0
				for k, v := range vote {
					if v > bestV {
						bestV = v
						bestI = k
					}
				}
				if grove.Data.Class[j] == bestI {
					p++
				}
			}
			dForest.Results = append(dForest.Results, float64(p)/float64(dForest.Forest.NSize))
			fmt.Println("Grove", i, float64(p)/float64(dForest.Forest.NSize))
		}
	}
	//create deep forest
	{
		x := make([][]float64, dForest.Forest.NSize)
		for i := 0; i < dForest.Forest.NSize; i++ {
			x[i] = make([]float64, dForest.Forest.Features)
			copy(x[i], dForest.Forest.Data.X[i])
			x[i] = append(x[i], dForest.ResultFeatures[i]...)
		}
		deepData := ForestData{X: x, Class: dForest.Forest.Data.Class}
		dForest.ForestDeep = Forest{
			Data: deepData,
		}
	}
	//build and train deep forest
	dForest.ForestDeep.Train(deepTrees)
}

func (dForest *DeepForest) Vote(x []float64) []float64 {
	//groves
	deepx := make([]float64, len(x))
	copy(deepx, x)
	for i, g := range dForest.Groves {
		gX := make([]float64, 0)
		for _, p := range dForest.RandomFeatures[i] {
			gX = append(gX, x[p])
		}
		v := g.Vote(gX)
		deepx = append(deepx, v...)
	}

	//deep tree
	votes := make([]float64, dForest.ForestDeep.Classes)
	for i := 0; i < dForest.ForestDeep.NTrees; i++ {
		v := dForest.ForestDeep.Trees[i].vote(deepx)
		for j := 0; j < dForest.ForestDeep.Classes; j++ {
			votes[j] += v[j]
		}
	}
	for j := 0; j < dForest.ForestDeep.Classes; j++ {
		votes[j] = votes[j] / float64(dForest.ForestDeep.NTrees)
	}
	return votes
}

//Created by Richard Malaschitz malaschitz@gmail.com
//2019-06-10 19:01
//Copyright (c) 2018. All Rights Reserved.
