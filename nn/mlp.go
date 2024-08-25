package nn

import (
	"go-micrograd/autograd"
	"go-micrograd/rng"
	"math"
)

type Module interface {
	Parameters() []*autograd.Value
}

type Neuron struct {
	w      []*autograd.Value
	b      *autograd.Value
	nonlin bool
}

func NewNeuron(nin int, random *rng.RNG, nonlin bool) *Neuron {
	weights := make([]*autograd.Value, nin)
	for i := 0; i < nin; i++ {
		weights[i] = autograd.NewValue(float64(random.Uniform(-1, 1))*math.Sqrt(float64(nin)), nil)
	}
	bias := autograd.NewValue(0, nil)
	return &Neuron{w: weights, b: bias, nonlin: nonlin}
}

func (n *Neuron) Parameters() []*autograd.Value {
	return append(n.w, n.b)
}

func (n *Neuron) Forward(inputs []*autograd.Value) *autograd.Value {
	out := autograd.NewValue(n.b.Data(), nil)
	for i := range inputs {
		out = out.Add(n.w[i].Mul(inputs[i]))
	}
	if n.nonlin {
		return out.Tanh()
	}
	return out
}

type Layer struct {
	neurons []*Neuron
}

func NewLayer(nin, nout int, random *rng.RNG, nonlin bool) *Layer {
	neurons := make([]*Neuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewNeuron(nin, random, nonlin)
	}
	return &Layer{neurons: neurons}
}

func (l *Layer) Forward(inputs []*autograd.Value) []*autograd.Value {
	outputs := make([]*autograd.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Forward(inputs)
	}
	return outputs
}

func (l *Layer) Parameters() []*autograd.Value {
	var params []*autograd.Value
	for _, neuron := range l.neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

type MLP struct {
	layers []*Layer
}

func NewMLP(nin int, nouts []int, random *rng.RNG) *MLP {
	layers := make([]*Layer, len(nouts))
	for i := range nouts {
		layers[i] = NewLayer(nin, nouts[i], random, i != len(nouts)-1)
		nin = nouts[i]
	}
	return &MLP{layers: layers}
}

func (m *MLP) Parameters() []*autograd.Value {
	var params []*autograd.Value
	for _, layer := range m.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (m *MLP) Forward(inputs []*autograd.Value) []*autograd.Value {
	for _, layer := range m.layers {
		inputs = layer.Forward(inputs)
	}
	return inputs
}
