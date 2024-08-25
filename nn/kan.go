package nn

import (
	"go-micrograd/autograd"
	"go-micrograd/rng"
)

// KolmogorovArnoldNeuron represents a neuron in a Kolmogorov-Arnold Network layer.
type KolmogorovArnoldNeuron struct {
	w *autograd.Value
	b *autograd.Value
}

func NewKANeuron(random *rng.RNG) *KolmogorovArnoldNeuron {
	w := autograd.NewValue(float64(random.Uniform(-1, 1)), nil)
	b := autograd.NewValue(0, nil)
	return &KolmogorovArnoldNeuron{w: w, b: b}
}

func (n *KolmogorovArnoldNeuron) Forward(x *autograd.Value) *autograd.Value {
	// Use some transformation function, e.g., sin or tanh
	activation := n.w.Mul(x).Add(n.b)
	return activation.Tanh()
}

func (n *KolmogorovArnoldNeuron) Parameters() []*autograd.Value {
	return []*autograd.Value{n.w, n.b}
}

type KolmogorovArnoldLayer struct {
	neurons []*KolmogorovArnoldNeuron
}

func NewKANLayer(nin int, nout int, random *rng.RNG) *KolmogorovArnoldLayer {
	neurons := make([]*KolmogorovArnoldNeuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewKANeuron(random)
	}
	return &KolmogorovArnoldLayer{neurons: neurons}
}

func (l *KolmogorovArnoldLayer) Forward(x []*autograd.Value) []*autograd.Value {
	outputs := make([]*autograd.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Forward(x[i])
	}
	return outputs
}

func (l *KolmogorovArnoldLayer) Parameters() []*autograd.Value {
	var params []*autograd.Value
	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

// KolmogorovArnoldNetwork represents a complete Kolmogorov-Arnold Network.
type KolmogorovArnoldNetwork struct {
	layers []*KolmogorovArnoldLayer
}

func NewKAN(nin int, nouts []int, random *rng.RNG) *KolmogorovArnoldNetwork {
	layers := make([]*KolmogorovArnoldLayer, len(nouts))
	for i := 0; i < len(nouts); i++ {
		layers[i] = NewKANLayer(nin, nouts[i], random)
		nin = nouts[i]
	}
	return &KolmogorovArnoldNetwork{layers: layers}
}

func (kan *KolmogorovArnoldNetwork) Forward(x []*autograd.Value) []*autograd.Value {
	for _, layer := range kan.layers {
		x = layer.Forward(x)
	}
	return x
}

func (kan *KolmogorovArnoldNetwork) Parameters() []*autograd.Value {
	var params []*autograd.Value
	for _, layer := range kan.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
