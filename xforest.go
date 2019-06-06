package randomForest

import (
	"math"
	"math/rand"
	"sync"
)

// Train Extremely randomized trees
func (forest *Forest) TrainX(trees int) {
	forest.NSize = len(forest.Data.X)
	forest.Features = len(forest.Data.X[0])
	forest.NTrees = trees
	forest.Trees = make([]Tree, forest.NTrees)
	forest.Classes = 0
	for _, c := range forest.Data.Class {
		if c >= forest.Classes {
			forest.Classes = c + 1
		}
	}
	if forest.MFeatures == 0 {
		forest.MFeatures = int(math.Sqrt(float64(forest.Features)))
	}
	if forest.LeafSize == 0 {
		forest.LeafSize = forest.NSize / 20
		if forest.LeafSize <= 0 {
			forest.LeafSize = 1
		} else if forest.LeafSize > 50 {
			forest.LeafSize = 50
		}
	}
	var wg sync.WaitGroup
	wg.Add(trees)
	for i := 0; i < trees; i++ {
		go forest.newXTree(i, &wg)
		//forest.newTree(i, &wg)
		//fmt.Println(i)
	}
	wg.Wait()
	imp := make([]float64, forest.Features)
	for i := 0; i < trees; i++ {
		z := forest.Trees[i].importance(forest)
		for i := 0; i < forest.Features; i++ {
			imp[i] += z[i]
		}
		//forest.Trees[i].Root.print()
	}
	for i := 0; i < forest.Features; i++ {
		imp[i] = imp[i] / float64(trees)
	}
	forest.FeatureImportance = imp
}

// Calculate a new tree in forest.
func (forest *Forest) newXTree(index int, wg *sync.WaitGroup) {
	defer wg.Done()
	//data
	used := make([]bool, forest.NSize)
	x := make([][]float64, forest.NSize)
	results := make([]int, forest.NSize)
	for i := 0; i < forest.NSize; i++ {
		k := rand.Intn(forest.NSize)
		x[i] = forest.Data.X[k]
		results[i] = forest.Data.Class[k]
		used[k] = true
	}
	// build Root
	root := Branch{}
	root.xbuild(forest, x, results, 1)
	tree := Tree{Root: root}
	// validation test tree
	count := 0
	e := 0.0
	for i := 0; i < forest.NSize; i++ {
		if !used[i] {
			count++
			v := root.vote(forest.Data.X[i])
			e += v[forest.Data.Class[i]]
		}
	}
	tree.Validation = e / float64(count)

	// add tree
	mux.Lock()
	forest.Trees[index] = tree
	mux.Unlock()
}

func (branch *Branch) xbuild(forest *Forest, x [][]float64, class []int, depth int) {
	//fmt.Println(repeat(".", depth), depth, len(x))
	classCount := make([]int, forest.Classes)
	for _, r := range class {
		classCount[r]++
	}
	branch.Gini = gini(classCount)
	branch.Size = len(class)
	branch.Depth = depth

	if (len(x) <= forest.LeafSize) || (branch.Gini == 0) {
		branch.IsLeaf = true
		branch.LeafValue = make([]float64, forest.Classes)
		for i, r := range classCount {
			branch.LeafValue[i] = float64(r) / float64(branch.Size)
		}
		return
	}
	//find best extremly random split
	attrsRandom := rand.Perm(forest.Features)[:forest.MFeatures]
	var bestAtrr int
	var bestSplit float64
	var bestGini = 1.0
	for _, a := range attrsRandom {
		//find min and max
		min := x[0][a]
		max := x[0][a]
		for i := 1; i < branch.Size; i++ {
			if x[i][a] > max {
				max = x[i][a]
			}
			if x[i][a] < min {
				min = x[i][a]
			}
		}
		if max == min {
			continue
		}
		split := (max + min) / 2
		s1 := make([]int, forest.Classes)
		s2 := make([]int, forest.Classes)
		c1 := 0
		copy(s2, classCount)
		for i := 0; i < branch.Size; i++ {
			if x[i][a] > split {
				s1[class[i]]++
				s2[class[i]]--
				c1++
			}
		}
		g1 := gini(s1)
		g2 := gini(s2)
		wg := (g1*float64(c1) + g2*float64(branch.Size-c1)) / float64(branch.Size)
		if wg < bestGini {
			bestGini = wg
			bestSplit = split
			bestAtrr = a
		}
	}
	//split it
	branch.GiniGain = branch.Gini - bestGini
	branch.Atribute = bestAtrr
	branch.Value = bestSplit
	x0 := make([][]float64, 0)
	x1 := make([][]float64, 0)
	c0 := make([]int, 0)
	c1 := make([]int, 0)
	for i := 0; i < branch.Size; i++ {
		if x[i][branch.Atribute] > branch.Value {
			x1 = append(x1, x[i])
			c1 = append(c1, class[i])
		} else {
			x0 = append(x0, x[i])
			c0 = append(c0, class[i])
		}
	}
	//create branches
	branch.Branch0 = &Branch{}
	branch.Branch1 = &Branch{}
	branch.Branch0.xbuild(forest, x0, c0, depth+1)
	branch.Branch1.xbuild(forest, x1, c1, depth+1)
}
