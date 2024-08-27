package nn

import "go-micrograd/autograd"

type Model interface {
	Forward([]float64) []*autograd.Value
	Parameters() []*autograd.Value
}

var _ Model = (*MLP)(nil)
var _ Model = (*KolmogorovArnoldNetwork)(nil)
