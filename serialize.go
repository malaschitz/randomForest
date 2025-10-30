package randomforest

import (
	"encoding/json"
	"fmt"
	"math"
)

var JSONNumbersPrecisionDecPlaces = 5

type jsonFloat float64

func (v jsonFloat) MarshalJSON() ([]byte, error) {
	multiplier := math.Pow(10, float64(JSONNumbersPrecisionDecPlaces))
	rounded := math.Round(float64(v)*multiplier) / multiplier
	format := fmt.Sprintf("%%.%df", JSONNumbersPrecisionDecPlaces)
	return []byte(fmt.Sprintf(format, rounded)), nil
}

func (v jsonFloat) Float64() float64 {
	return float64(v)
}

func floatJSONSlice(slice []float64) []jsonFloat {
	ans := make([]jsonFloat, len(slice))
	for i, v := range slice {
		ans[i] = jsonFloat(v)
	}
	return ans
}

func floatJSONSliceRev(slice []jsonFloat) []float64 {
	ans := make([]float64, len(slice))
	for i, v := range slice {
		ans[i] = v.Float64()
	}
	return ans
}

type jsonTreeNode struct {
	ID        int         `json:"id"`
	Attribute int         `json:"attribute"`
	Branch0   int         `json:"branch0"`
	Branch1   int         `json:"branch1"`
	Value     jsonFloat   `json:"value"`
	LeafValue []jsonFloat `json:"leafValue"`
	Gini      jsonFloat   `json:"gini"`
	GiniGain  jsonFloat   `json:"giniGain"`
	Size      int         `json:"size"`
	Depth     int         `json:"depth"`
}

type jsonTree struct {
	Nodes      []*jsonTreeNode `json:"nodes"`
	Validation jsonFloat       `json:"validation"`
}

type jsonForest struct {
	Trees             []*jsonTree `json:"trees"`
	Features          int         `json:"features"`
	Classes           int         `json:"classes"`
	LeafSize          int         `json:"leafSize"`
	MFeatures         int         `json:"mFeatures"`
	NTrees            int         `json:"nTrees"`
	NSize             int         `json:"nSize"`
	MaxDepth          int         `json:"maxDepth"`
	FeatureImportance []jsonFloat `json:"featureImportance"`
}

func attachIDs(tree *Tree) *jsonTree {
	jsonTree := &jsonTree{
		Nodes:      make([]*jsonTreeNode, 0, 100),
		Validation: jsonFloat(tree.Validation),
	}
	dfsProcNode(&tree.Root, jsonTree, 0)
	return jsonTree
}

// dfsProcNode walks (in DFS manner) through a tree with root in the `node`
// and attaches numeric IDs to nodes.
func dfsProcNode(node *Branch, outTree *jsonTree, availID int) int {
	newNode := &jsonTreeNode{
		ID:        availID,
		Attribute: node.Attribute,
		Value:     jsonFloat(node.Value),
		LeafValue: floatJSONSlice(node.LeafValue),
		Gini:      jsonFloat(node.Gini),
		GiniGain:  jsonFloat(node.GiniGain),
		Size:      node.Size,
		Depth:     node.Depth,
	}
	lastID := availID
	if node.Branch0 != nil {
		newNode.Branch0 = lastID + 1
		lastID = dfsProcNode(node.Branch0, outTree, newNode.Branch0)

	} else {
		newNode.Branch0 = -1
	}
	if node.Branch1 != nil {
		newNode.Branch1 = lastID + 1
		lastID = dfsProcNode(node.Branch1, outTree, newNode.Branch1)

	} else {
		newNode.Branch1 = -1
	}
	outTree.Nodes = append(outTree.Nodes, newNode)
	return lastID
}

func deserializeJsonTreeNode(nodeID int, mapping map[int]*jsonTreeNode) *Branch {
	node := mapping[nodeID]
	br := Branch{
		Attribute: node.Attribute,
		Value:     node.Value.Float64(),
		IsLeaf:    node.Branch0 == -1 && node.Branch1 == -1,
		LeafValue: floatJSONSliceRev(node.LeafValue),
		Gini:      node.Gini.Float64(),
		GiniGain:  node.GiniGain.Float64(),
		Size:      node.Size,
		Depth:     node.Depth,
	}
	if node.Branch0 > -1 {
		br.Branch0 = deserializeJsonTreeNode(node.Branch0, mapping)
	}
	if node.Branch1 > -1 {
		br.Branch1 = deserializeJsonTreeNode(node.Branch1, mapping)
	}
	return &br
}

// -------- Forest's JSON interface methods ----

// UnmarshalJSON implements the json.Unmarshaler method.
func (forest *Forest) UnmarshalJSON(b []byte) error {
	var sForest jsonForest
	if err := json.Unmarshal(b, &sForest); err != nil {
		return fmt.Errorf("failed to load the Forest model from JSON: %w", err)
	}
	forest.Features = sForest.Features
	forest.Classes = sForest.Classes
	forest.LeafSize = sForest.LeafSize
	forest.MFeatures = sForest.MFeatures
	forest.NTrees = sForest.NTrees
	forest.NSize = sForest.NSize
	forest.MaxDepth = sForest.MaxDepth
	forest.FeatureImportance = floatJSONSliceRev(sForest.FeatureImportance)
	forest.Trees = make([]Tree, len(sForest.Trees))

	idMap := make(map[int]*jsonTreeNode)
	for i, tree := range sForest.Trees {
		t := Tree{
			Validation: tree.Validation.Float64(),
		}
		// let's not rely on node order and map IDs properly
		for _, nd := range tree.Nodes {
			idMap[nd.ID] = nd
		}
		t.Root = *deserializeJsonTreeNode(0, idMap)
		forest.Trees[i] = t
	}
	return nil
}

// MarshalJSON implements json.Marshaler interface allowing
// for Forest serialization in a standard way via json.Marshal function.
func (forest Forest) MarshalJSON() ([]byte, error) {
	toSave := jsonForest{
		Trees:             make([]*jsonTree, len(forest.Trees)),
		Features:          forest.Features,
		Classes:           forest.Classes,
		LeafSize:          forest.LeafSize,
		MFeatures:         forest.MFeatures,
		NTrees:            forest.NTrees,
		NSize:             forest.NSize,
		MaxDepth:          forest.MaxDepth,
		FeatureImportance: floatJSONSlice(forest.FeatureImportance),
	}
	for i, tr := range forest.Trees {
		tmp := attachIDs(&tr)
		toSave.Trees[i] = tmp
	}

	return json.Marshal(toSave)
}
