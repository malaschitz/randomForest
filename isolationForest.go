package randomforest

import (
	"math"
	"math/rand"
)

//Forest je base class for whole forest with database, properties of Forest and trees.
type IsolationForest struct {
	X        [][]float64
	Features int      // number of attributes
	NTrees   int      // number of trees
	NSize    int      // len of data
	Sample   int      //sample size
	Results  [][2]int //results - sum of depths and counts for every data
}

// Train run training process. Parameter is number of calculated trees.
func (forest *IsolationForest) Train(trees int) {
	forest.defaults()
	forest.NTrees = trees
	forest.Results = make([][2]int, len(forest.X))
	for i := 0; i < len(forest.X); i++ {
		forest.Results[i] = [2]int{0, 0}
	}
	forest.buildNewTrees(0, trees)
}

func (forest *IsolationForest) buildNewTrees(firstIndex int, trees int) {
	// constrain parallelism, use buffered channel as counting semaphore
	s := make(chan bool, NumWorkers)
	for i := 0; i < trees; i++ {
		s <- true
		go func(j int) {
			defer func() { <-s }()
			forest.newTree(j)
		}(firstIndex + i)
	}
	// wait for all trees to finish
	for i := 0; i < NumWorkers; i++ {
		s <- true
	}
}

func (forest *IsolationForest) defaults() {
	forest.NSize = len(forest.X)
	forest.Features = len(forest.X[0])
	forest.Sample = 256
	if forest.Sample > forest.NSize {
		forest.Sample = forest.NSize
	}
}

// Calculate a new tree in forest.
func (forest *IsolationForest) newTree(index int) {
	//random samples
	samples := map[int]bool{}
	for {
		samples[rand.Intn(forest.NSize)] = true
		if len(samples) == forest.Sample {
			break
		}
	}
	forest.branch(samples, 0)
}

func (forest *IsolationForest) branch(samples map[int]bool, depth int) {
	if len(samples) > 1 {
		for i := 0; i < forest.Features; i++ {
			feature := rand.Intn(forest.Features)
			min := math.MaxFloat64
			max := -math.MaxFloat64
			for k := range samples {
				if forest.X[k][feature] < min {
					min = forest.X[k][feature]
				}
				if forest.X[k][feature] > max {
					max = forest.X[k][feature]
				}
			}
			if min != max { //found a new split
				splitValue := rand.Float64()*(max-min) + min
				sml0 := map[int]bool{}
				sml1 := map[int]bool{}
				for k := range samples {
					if forest.X[k][feature] < splitValue {
						sml0[k] = true
					} else {
						sml1[k] = true
					}
				}
				forest.branch(sml0, depth+1)
				forest.branch(sml1, depth+1)
				return
			}
		}
	}
	//end of branching
	mux.Lock()
	for k := range samples {
		s := forest.Results[k]
		s[0] = s[0] + 1
		s[1] = s[1] + depth
		forest.Results[k] = s
	}
	mux.Unlock()
}
