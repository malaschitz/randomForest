package randomforest

import (
	"encoding/json"
	"math"
	"math/rand"
	"testing"
)

func assertEqualFloat64(f1, f2, precision float64) bool {
	return math.Abs(f1-f2) < precision
}

func dfsBranchVisit(branch *Branch) []int {
	valuesPath := make([]int, 0, 5)
	valuesPath = append(valuesPath, int(branch.Value))
	if branch.Branch0 != nil {
		valuesPath = append(valuesPath, dfsBranchVisit(branch.Branch0)...)
	}
	if branch.Branch1 != nil {
		valuesPath = append(valuesPath, dfsBranchVisit(branch.Branch1)...)
	}
	return valuesPath
}

// TestStructuralConsistency tests whether serialization's flattening
// of Forest trees (a list of nodes along with IDs of their children)
// preserves structure.
func TestStructuralConsistency(t *testing.T) {
	fr := &Forest{Trees: make([]Tree, 1)}
	fr.Trees[0] = Tree{
		Root: Branch{
			Value: 1,
			Branch0: &Branch{
				Value: 2,
				Branch0: &Branch{
					Value: 3,
					Branch0: &Branch{
						Value: 4,
						Branch0: &Branch{
							Value: 5,
						},
					},
				},
				Branch1: &Branch{
					Value: 6,
				},
			},
			Branch1: &Branch{
				Value: 7,
				Branch0: &Branch{
					Value: 8,
					Branch0: &Branch{
						Value: 9,
					},
					Branch1: &Branch{
						Value: 10,
					},
				},
				Branch1: &Branch{
					Value: 11,
				},
			},
		},
	}

	pathExpected := dfsBranchVisit(&fr.Trees[0].Root)

	jsonData, err := json.Marshal(fr)
	if err != nil {
		t.Error("failed to serialize a Forest instance: %w", err)
	}
	var fr2 Forest
	if err := json.Unmarshal(jsonData, &fr2); err != nil {
		t.Error("failed to deserialize a Forest instance: %w", err)
	}
	pathTest := dfsBranchVisit(&fr2.Trees[0].Root)

	if len(pathExpected) != len(pathTest) {
		t.Error("different number of nodes in the original and deserialized Forest trees")
	}
	for i := 0; i < len(pathExpected); i++ {
		if pathExpected[i] != pathTest[i] {
			t.Error("different tree structure between the original and deserialized Forest trees")
		}
	}
}

// TestJSONSerialization ensures that the original Forest and its serialized
// and deserialized form produce exact results.
func TestJSONSerialization(t *testing.T) {
	xData := [][]float64{}
	yData := []int{}
	for i := 0; i < 1000; i++ {
		x := []float64{rand.Float64(), rand.Float64(), rand.Float64(), rand.Float64()}
		y := int(x[0] + x[1] + x[2] + x[3])
		xData = append(xData, x)
		yData = append(yData, y)
	}
	forest := Forest{}
	forestData := ForestData{X: xData, Class: yData}
	forest.Data = forestData
	forest.Train(500)

	testInput := []float64{0.1, 0.1, 0.1, 0.1}

	voteExpected := forest.Vote(testInput)

	data, err := json.Marshal(forest)
	if err != nil {
		t.Error("failed to serialize the forest")
	}
	var forestFromJSON Forest
	if err := json.Unmarshal(data, &forestFromJSON); err != nil {
		t.Error("failed to read a forest from JSON data")
	}
	forestFromJSON.Data = forestData
	voteTest := forestFromJSON.Vote(testInput)

	if len(voteExpected) != len(voteTest) {
		t.Error("vote slice has not the expected size")
	}
	for i := 0; i < len(voteExpected); i++ {
		// note: the precision below must match the JSONNumbersPrecisionDecPlaces package variable
		if !assertEqualFloat64(voteExpected[i], voteTest[i], 0.00001) {
			t.Errorf("vote element %d mismatch (expected %.5f, got %.5f)", i, voteExpected[i], voteTest[i])
		}
	}
}
