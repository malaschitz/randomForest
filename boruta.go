package randomforest

import (
	"fmt"
	"math/big"
	"math/rand"
)

/*
	Boruta is smart algorithm for select important features with Random Forest. It was developed in language R.

	X [][]float64 - data for random forest. At least three features (columns) are required.
	Class []int - classes for random forest (0,1,..)
	trees int - number of trees used by Boruta algorithm. Is not need too big number of trees. (50-200)
	cycles int - number of cycles (20-50) of Boruta algorithm.
	threshold float64 - threshold for select feauters (0.05)
	recursive bool - algorithm repeat process until all features are important
	verbose bool - will print process of boruta algorithm.
*/
func BorutaDefault(x [][]float64, class []int) ([]int, map[int]int) {
	return Boruta(x, class, 100, 20, 0.05, true, true)
}

func Boruta(x [][]float64, class []int, trees int, cycles int, threshold float64, recursive bool, verbose bool) ([]int, map[int]int) {
	//keep mapping of features
	featMap := make(map[int]int, 0)
	for i := 0; i < len(x[0]); i++ {
		featMap[i] = i
	}

	c2 := 0
	for {
		c2++
		features := len(featMap)
		//copy x to working x
		wx := make([][]float64, len(x))
		for i := 0; i < len(x); i++ {
			wx[i] = make([]float64, features)
			for j := 0; j < features; j++ {
				wx[i][j] = x[i][featMap[j]]
			}
		}

		//add shadow columns to wx
		for i := 0; i < len(wx); i++ {
			for j := 0; j < features; j++ {
				wx[i] = append(wx[i], wx[i][j])
			}
		}

		tips := make(map[int]int, 0)
		for cycle := 0; cycle < cycles; cycle++ {
			if verbose {
				fmt.Println("Cycle:", cycle+1, "/", c2)
			}
			//shufle
			for i := 0; i < features; i++ {
				column := features + i
				for j := 0; j < len(wx); j++ {
					k := rand.Intn(len(wx))
					wx[j][column], wx[k][column] = wx[k][column], wx[j][column]
				}
			}
			//forest
			forest := Forest{Data: ForestData{X: wx, Class: class}}
			forest.Train(trees)
			//save tips
			bestShadow := 0.0
			for i := features; i < 2*features; i++ {
				if forest.FeatureImportance[i] > bestShadow {
					bestShadow = forest.FeatureImportance[i]
				}
			}
			c := 0
			for i := 0; i < features; i++ {
				if forest.FeatureImportance[i] > bestShadow {
					tips[i]++
					c++
				}
			}
			if verbose {
				fmt.Println("selected tips:", c, "/", features)
			}
		}
		//select remaining features
		tipThreshold := bionimalThreshold(cycles, threshold)
		newFeatMap := make(map[int]int, 0)
		c := 0
		for i := 0; i < features; i++ {
			if tips[i] >= tipThreshold {
				newFeatMap[c] = featMap[i]
				c++
			}
		}
		if verbose {
			fmt.Println("Threshold count:", tipThreshold)
			fmt.Println("Threshold features", len(newFeatMap), "/", len(featMap))
		}
		if len(newFeatMap) == len(featMap) || len(newFeatMap) < 3 || !recursive {
			result := make([]int, 0)
			for _, v := range newFeatMap {
				result = append(result, v)
			}
			return result, tips
		}
		featMap = newFeatMap
		if verbose {
			result := make([]int, 0)
			for _, v := range newFeatMap {
				result = append(result, v)
			}
			fmt.Println("Selected feautures", result)
		}
	}

}

func bionimalThreshold(n int, threshold float64) int {
	sum := 0.0
	s := make([]float64, n+1)
	bi := big.Int{}
	for i := 0; i <= n; i++ {
		bn := float64(bi.Binomial(int64(n), int64(i)).Int64())
		sum = sum + bn
		s[i] = sum
	}
	for j := 0; j < n; j++ {
		if float64(s[j])/float64(sum) >= threshold {
			return j
		}
	}
	return n
}
