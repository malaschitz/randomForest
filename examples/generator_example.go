package main

import (
	"fmt"
	"math/rand"

	"github.com/malaschitz/randomForest/example/generator"
)

//Example of generated dataset
func main() {
	rand.Seed(1)
	fmt.Println("-------")
	fmt.Println("Dataset with 10 examples, 4 features, 3 classes")
	fmt.Println()
	data, res := generator.CreateDataset(10, 4, 3)
	for i, d := range data {
		fmt.Printf("[%.3f %.3f %.3f %.3f] => %d\n", d[0], d[1], d[2], d[3], res[i])
	}

	fmt.Println()
	fmt.Println("-------")
	fmt.Println("Dataset with 20 examples, 6 features, 2 classes")
	fmt.Println()
	data, res = generator.CreateDataset(20, 6, 2)
	for i, d := range data {
		fmt.Printf("[%.3f %.3f %.3f %.3f %.3f %.3f] => %d\n", d[0], d[1], d[2], d[3], d[4], d[5], res[i])
	}

}
