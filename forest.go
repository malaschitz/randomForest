package randomforest

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"gonum.org/v1/gonum/stat"
)

var mux = &sync.Mutex{}

//Forest je base class for whole forest with database, properties of Forest and trees.
type Forest struct {
	Data              ForestData // database for calculate trees
	Trees             []Tree     // all generated trees
	Features          int        // number of attributes
	Classes           int        // number of classes
	LeafSize          int        // leaf size
	MFeatures         int        // attributes for choose proper split
	NTrees            int        // number of trees
	NSize             int        // len of data
	FeatureImportance []float64  //stats of FeatureImportance
}

// ForestData contains database
type ForestData struct {
	X     [][]float64 // All data are float64 numbers
	Class []int       // Result should be int numbers 0,1,2,..
}

// Tree is one random tree in forest with Branch and validation number
type Tree struct {
	Root       Branch
	Validation float64
}

// Branch is tree structure of branches
type Branch struct {
	Attribute        int
	Value            float64
	IsLeaf           bool
	LeafValue        []float64
	Gini             float64
	GiniGain         float64
	Size             int
	Branch0, Branch1 *Branch
	Depth            int
}

// Train run training process. Parameter is number of calculated trees.
func (forest *Forest) Train(trees int) {
	forest.defaults()
	forest.NTrees = trees
	forest.Trees = make([]Tree, forest.NTrees)
	var wg sync.WaitGroup
	wg.Add(trees)
	for i := 0; i < trees; i++ {
		go forest.newTree(i, &wg)
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

// AddDataRow add new data
// data: new data row
// class: result
// max: max number of data. Remove first if there is more datas. If max < 1 - unlimited
// newTrees: number of trees after add data row
// maxTress: maximum number of trees
//
// This feature support Continuous Random Forest
func (forest *Forest) AddDataRow(data []float64, class int, max int, newTrees int, maxTrees int) {
	forest.Data.X = append(forest.Data.X, data)
	forest.Data.Class = append(forest.Data.Class, class)
	if max > 0 && len(forest.Data.X) > max {
		forest.Data.X = forest.Data.X[1:]
		forest.Data.Class = forest.Data.Class[1:]
	}
	forest.defaults()
	var wg sync.WaitGroup
	wg.Add(newTrees)
	index := len(forest.Trees)
	for i := 0; i < newTrees; i++ {
		forest.Trees = append(forest.Trees, Tree{})
	}
	for i := 0; i < newTrees; i++ {
		go forest.newTree(index+i, &wg)
	}
	wg.Wait()
	//remove old trees
	if len(forest.Trees) > maxTrees && maxTrees > 0 {
		forest.Trees = forest.Trees[len(forest.Trees)-maxTrees:]
	}
	forest.NTrees = len(forest.Trees)
}

func (forest *Forest) defaults() {
	forest.NSize = len(forest.Data.X)
	forest.Features = len(forest.Data.X[0])
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
}

// Vote is used for calculate class in existed forest
func (forest *Forest) Vote(x []float64) []float64 {
	votes := make([]float64, forest.Classes)
	for i := 0; i < forest.NTrees; i++ {
		v := forest.Trees[i].vote(x)
		for j := 0; j < forest.Classes && j < len(v); j++ {
			votes[j] += v[j]
		}
	}
	for j := 0; j < forest.Classes; j++ {
		votes[j] = votes[j] / float64(forest.NTrees)
	}
	return votes
}

// WeightVote use validation's weight for result
func (forest *Forest) WeightVote(x []float64) []float64 {
	votes := make([]float64, forest.Classes)
	total := 0.0
	for i := 0; i < forest.NTrees; i++ {
		e := 1.0001 - forest.Trees[i].Validation
		w := 0.5 * math.Log(float64(forest.Classes-1)*(1-e)/e)
		if w > 0 {
			v := forest.Trees[i].vote(x)
			for j := 0; j < forest.Classes; j++ {
				votes[j] += v[j] * w
			}
			total += w
		} else {
			//fmt.Println("wv", e, w, total)
		}
	}
	for j := 0; j < forest.Classes; j++ {
		votes[j] = votes[j] / total
	}
	return votes
}

// Calculate a new tree in forest.
func (forest *Forest) newTree(index int, wg *sync.WaitGroup) {
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
	root.build(forest, x, results, 1)
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

// PrintFeatureImportance print list of features
func (forest *Forest) PrintFeatureImportance() {
	imp := make([]float64, forest.Features)
	for i := 0; i < forest.NTrees; i++ {
		z := forest.Trees[i].importance(forest)
		for i := 0; i < forest.Features; i++ {
			imp[i] += z[i]
		}
	}
	for i := 0; i < forest.Features; i++ {
		imp[i] = imp[i] / float64(forest.NTrees)
	}
	forest.FeatureImportance = imp

	fmt.Println("-------- feature importance")
	for i := 0; i < forest.Features; i++ {
		fmt.Println(i, forest.FeatureImportance[i])
	}
	fmt.Println("-------- cross validation")
	xs := make([]float64, 0)
	for _, tree := range forest.Trees {
		xs = append(xs, tree.Validation)
	}
	sort.Float64s(xs)
	mean := stat.Mean(xs, nil)
	median := stat.Quantile(0.5, stat.Empirical, xs, nil)
	variance := stat.Variance(xs, nil)
	stddev := math.Sqrt(variance)

	fmt.Printf("mean=       %v\n", mean)
	fmt.Printf("median=     %v\n", median)
	fmt.Printf("variance=   %v\n", variance)
	fmt.Printf("std-dev=    %v\n", stddev)
	fmt.Printf("worst tree= %v\n", xs[0])
	fmt.Printf("best tree=  %v\n", xs[len(xs)-1])

	fmt.Println("--------")
}

func (branch *Branch) build(forest *Forest, x [][]float64, class []int, depth int) {
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
	//find best split
	attrsRandom := rand.Perm(forest.Features)[:forest.MFeatures]
	var bestAtrr int
	var bestValue float64
	var bestGini = 1.0
	for _, a := range attrsRandom {
		//sort data
		srt := make([]int, branch.Size)
		for i := 0; i < branch.Size; i++ {
			srt[i] = i
		}
		sort.Slice(srt, func(i, j int) bool {
			ii := srt[i]
			jj := srt[j]
			return x[ii][a] < x[jj][a]
		})
		//go throuh data
		v := x[srt[0]][a]
		s1 := make([]int, forest.Classes)
		s2 := make([]int, forest.Classes)
		copy(s2, classCount)
		for i := 0; i < branch.Size; i++ {
			index := srt[i]
			if x[index][a] > v {
				g1 := gini(s1)
				g2 := gini(s2)
				wg := (g1*float64(i) + g2*float64(branch.Size-i)) / float64(branch.Size)
				if wg < bestGini {
					bestGini = wg
					bestValue = v
					bestAtrr = a
				}
				v = x[index][a]
			}
			s1[class[index]]++
			s2[class[index]]--
		}
	}
	//split it
	branch.GiniGain = branch.Gini - bestGini
	branch.Attribute = bestAtrr
	branch.Value = bestValue
	x0 := make([][]float64, 0)
	x1 := make([][]float64, 0)
	c0 := make([]int, 0)
	c1 := make([]int, 0)
	for i := 0; i < branch.Size; i++ {
		if x[i][branch.Attribute] > branch.Value {
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
	branch.Branch0.build(forest, x0, c0, depth+1)
	branch.Branch1.build(forest, x1, c1, depth+1)
}

func (tree *Tree) vote(x []float64) []float64 {
	return tree.Root.vote(x)
}

func (tree *Tree) importance(forest *Forest) []float64 {
	imp := make([]float64, forest.Features)
	tree.Root.importance(imp)
	//normalize
	sum := 0.0
	for i := 0; i < forest.Features; i++ {
		sum += imp[i]
	}
	if sum > 0 {
		for i := 0; i < forest.Features; i++ {
			imp[i] = imp[i] / sum
		}
	}
	return imp
}

func (branch *Branch) importance(imp []float64) {
	if branch.IsLeaf {
		return
	}
	imp[branch.Attribute] += float64(branch.Size) * branch.GiniGain
	branch.Branch0.importance(imp)
	branch.Branch1.importance(imp)
}

func (branch *Branch) vote(x []float64) []float64 {
	if branch.IsLeaf {
		return branch.LeafValue
	}
	if x[branch.Attribute] > branch.Value {
		return branch.Branch1.vote(x)
	}
	return branch.Branch0.vote(x)
}

func (branch *Branch) print() {
	if branch.IsLeaf {
		fmt.Printf("%s ... LEAF %v\tsize: %6d\tgini: %5.4f\n",
			repeat("_", branch.Depth*3), branch.LeafValue, branch.Size, branch.Gini)
	} else {
		fmt.Printf("%s ... size: %6d\tattr: %3d\tgini: %5.4f %5.4f \t\tvalue: %4.3f\n",
			repeat("_", branch.Depth*3), branch.Size, branch.Attribute, branch.Gini, branch.GiniGain, branch.Value)
		branch.Branch0.print()
		branch.Branch1.print()
		fmt.Printf("%s\n", repeat("_", branch.Depth*3))
	}
}

func (branch *Branch) branches() int {
	if branch.IsLeaf {
		return 1
	}
	return branch.Branch0.branches() + branch.Branch1.branches()
}

func repeat(s string, n int) string {
	z := s
	for i := 0; i < n; i++ {
		z = z + s
	}
	return z
}

func gini(data []int) float64 {
	sum := 0
	for _, a := range data {
		sum += a
	}
	sumF := float64(sum)
	g := 1.0
	for _, a := range data {
		g = g - (float64(a)/sumF)*(float64(a)/sumF)
	}
	return g
}
