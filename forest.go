package randomForest

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"

	"gonum.org/v1/gonum/stat"
)

var mux = &sync.Mutex{}

func (forest *Forest) Train(trees int) {
	forest.NSize = len(forest.Data.X)
	forest.NAttrs = len(forest.Data.X[0])
	forest.NTrees = trees
	forest.Trees = make([]Tree, forest.NTrees)
	forest.ClassFunction = gini2
	if forest.MAttrs == 0 {
		forest.MAttrs = int(math.Sqrt(float64(forest.NAttrs)))
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
		go forest.newTree(i, &wg)
	}
	wg.Wait()
	imp := make([]float64, forest.NAttrs)
	for i := 0; i < trees; i++ {
		z := forest.Trees[i].importance(forest)
		for i := 0; i < forest.NAttrs; i++ {
			imp[i] += z[i]
		}
		//forest.Trees[i].Root.print()
	}
	for i := 0; i < forest.NAttrs; i++ {
		imp[i] = imp[i] / float64(trees)
	}
	forest.FeatureImportance = imp
}

func (forest *Forest) Vote(x []float64) float64 {
	votes := 0.0
	for i := 0; i < forest.NTrees; i++ {
		votes += forest.Trees[i].vote(x)
	}
	return votes / float64(forest.NTrees)
}

func (forest *Forest) WeightVote(x []float64) float64 {
	votes := 0.0
	total := 0.0
	for i := 0; i < forest.NTrees; i++ {
		votes += forest.Trees[i].vote(x) * forest.Trees[i].Validation
		total += forest.Trees[i].Validation
	}
	return votes / total
}

// Calculate a new tree in forest.
func (forest *Forest) newTree(index int, wg *sync.WaitGroup) {
	defer wg.Done()
	//data
	used := make([]bool, forest.NSize)
	x := make([][]float64, forest.NSize)
	results := make([]bool, forest.NSize)
	for i := 0; i < forest.NSize; i++ {
		k := rand.Intn(forest.NSize)
		x[i] = forest.Data.X[k]
		results[i] = forest.Data.Results[k]
		used[k] = true
	}
	// build Root
	root := Branch{}
	root.build(forest, x, results, 1)
	tree := Tree{Root: root}
	// validation test tree
	count := 0
	right := 0.0
	for i := 0; i < forest.NSize; i++ {
		if !used[i] {
			count++
			right += root.vote(forest.Data.X[i])
		}
	}
	tree.Validation = right / float64(count)

	// add tree
	mux.Lock()
	forest.Trees[index] = tree
	mux.Unlock()
}

func (forest *Forest) PrintFeatureImportance() {
	fmt.Println("-------- feature importance")
	for i := 0; i < forest.NAttrs; i++ {
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

func (branch *Branch) build(forest *Forest, x [][]float64, results []bool, depth int) {
	//fmt.Println(repeat(".", depth), depth, len(x))
	vTrue := 0
	vFalse := 0
	for _, r := range results {
		if r {
			vTrue++
		} else {
			vFalse++
		}
	}
	branch.Gini = forest.ClassFunction(vTrue, vFalse)
	branch.Size = len(results)
	branch.Depth = depth

	if (len(x) <= forest.LeafSize) || (branch.Gini == 0) {
		branch.IsLeaf = true
		branch.LeafValue = float64(vTrue) / float64(vTrue+vFalse)
		return
	}
	//find best split
	attrsRandom := rand.Perm(forest.NAttrs)[:forest.MAttrs]
	//fmt.Println(repeat(".", depth), "ATRR", attrsRandom)
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
		t := 0
		f := 0
		for i := 0; i < branch.Size; i++ {
			index := srt[i]
			if x[index][a] > v {
				g1 := forest.ClassFunction(t, f)
				g2 := forest.ClassFunction(vTrue-t, vFalse-f)
				wg := (g1*float64(t+f) + g2*float64(branch.Size-t-f)) / float64(branch.Size)
				//fmt.Println(repeat(".", depth), g1, g2)
				if wg < bestGini {
					bestGini = wg
					bestValue = v
					bestAtrr = a
				}
				v = x[index][a]
			}
			if results[index] {
				t++
			} else {
				f++
			}
		}
	}
	//split it
	branch.GiniGain = branch.Gini - bestGini
	branch.Atribute = bestAtrr
	branch.Value = bestValue
	x0 := make([][]float64, 0)
	x1 := make([][]float64, 0)
	r0 := make([]bool, 0)
	r1 := make([]bool, 0)
	for i := 0; i < branch.Size; i++ {
		if x[i][branch.Atribute] > branch.Value {
			x1 = append(x1, x[i])
			r1 = append(r1, results[i])
		} else {
			x0 = append(x0, x[i])
			r0 = append(r0, results[i])
		}
	}
	//create branches
	//fmt.Println(repeat(".", depth), "SPLIT", len(x0), len(x1))
	branch.Branch0 = &Branch{}
	branch.Branch1 = &Branch{}
	branch.Branch0.build(forest, x0, r0, depth+1)
	branch.Branch1.build(forest, x1, r1, depth+1)
}

func (tree *Tree) vote(x []float64) float64 {
	return tree.Root.vote(x)
}

func (tree *Tree) importance(forest *Forest) []float64 {
	imp := make([]float64, forest.NAttrs)
	tree.Root.importance(imp)
	//normalize
	sum := 0.0
	for i := 0; i < forest.NAttrs; i++ {
		sum += imp[i]
	}
	for i := 0; i < forest.NAttrs; i++ {
		imp[i] = imp[i] / sum
	}
	return imp
}

func (branch *Branch) importance(imp []float64) {
	if branch.IsLeaf {
		return
	} else {
		imp[branch.Atribute] += float64(branch.Size) * branch.GiniGain
		branch.Branch0.importance(imp)
		branch.Branch1.importance(imp)
	}
}

func (branch *Branch) vote(x []float64) float64 {
	if branch.IsLeaf {
		return branch.LeafValue
	} else {
		if x[branch.Atribute] > branch.Value {
			return branch.Branch1.vote(x)
		} else {
			return branch.Branch0.vote(x)
		}
	}
}

func (branch *Branch) print() {
	if branch.IsLeaf {
		fmt.Printf("%s\tLEAF %t\tsize: %6d\tgini: %5.4f\n",
			repeat("_", branch.Depth*3), branch.LeafValue, branch.Size, branch.Gini)
	} else {
		fmt.Printf("%s\tsize: %6d\tattr: %3d\tvalue: %4.3f\tgini: %5.4f %5.4f\n",
			repeat("_", branch.Depth*3), branch.Size, branch.Atribute, branch.Value, branch.Gini, branch.GiniGain)
		branch.Branch0.print()
		branch.Branch1.print()
		fmt.Printf("%s\n", repeat("_", branch.Depth*3))
	}
}

func (branch *Branch) branches() int {
	if branch.IsLeaf {
		return 1
	} else {
		return branch.Branch0.branches() + branch.Branch1.branches()
	}
}

func repeat(s string, n int) string {
	z := s
	for i := 0; i < n; i++ {
		z = z + s
	}
	return z
}

func gini2(a, b int) float64 {
	sum := float64(a + b)
	g := 1.0 - ((float64(a)/sum)*(float64(a)/sum) + (float64(b)/sum)*(float64(b)/sum))
	return g
}

func entropy2(a, b int) float64 {
	sum := float64(a + b)
	ap := (float64(a) / sum)
	bp := (float64(b) / sum)
	g := -ap*math.Log2(ap) - bp*math.Log2(bp)
	if math.IsNaN(g) {
		return 0
	}
	return g
}

type Forest struct {
	Data              ForestData
	Trees             []Tree
	LeafSize          int // leaf size
	MAttrs            int // attributes for choose proper split
	NTrees            int // number of trees
	NAttrs            int // number of attributes
	NSize             int // len of data
	ClassFunction     func(a, b int) float64
	FeatureImportance []float64
}

type ForestData struct {
	X       [][]float64
	Results []bool
}

type Tree struct {
	Root       Branch
	Validation float64
}

type Branch struct {
	Atribute         int
	Value            float64
	IsLeaf           bool
	LeafValue        float64
	Gini             float64
	GiniGain         float64
	Size             int
	Branch0, Branch1 *Branch
	Depth            int
}
