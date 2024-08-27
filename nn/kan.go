package nn

import (
	"go-micrograd/autograd"
	"go-micrograd/rng"
)

type KolmogorovArnoldNeuron struct {
	w []*autograd.Value
	b *autograd.Value
}

func NewKANeuron(nin int, random *rng.RNG) *KolmogorovArnoldNeuron {
	weights := make([]*autograd.Value, nin)
	for i := 0; i < nin; i++ {
		weights[i] = autograd.NewValue(float64(random.Uniform(-1, 1)), nil)
	}
	b := autograd.NewValue(0, nil)
	return &KolmogorovArnoldNeuron{w: weights, b: b}
}

func (n *KolmogorovArnoldNeuron) Forward(x []*autograd.Value) *autograd.Value {
	activation := n.b
	for i, input := range x {
		activation = activation.Add(n.w[i].Mul(input))
	}
	return activation.Tanh()
}

func (n *KolmogorovArnoldNeuron) Parameters() []*autograd.Value {
	return append(n.w, n.b)
}

type KolmogorovArnoldLayer struct {
	neurons []*KolmogorovArnoldNeuron
}

func NewKANLayer(nin int, nout int, random *rng.RNG) *KolmogorovArnoldLayer {
	neurons := make([]*KolmogorovArnoldNeuron, nout)
	for i := 0; i < nout; i++ {
		neurons[i] = NewKANeuron(nin, random)
	}
	return &KolmogorovArnoldLayer{neurons: neurons}
}

func (l *KolmogorovArnoldLayer) Forward(x []*autograd.Value) []*autograd.Value {
	outputs := make([]*autograd.Value, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Forward(x)
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

func (kan *KolmogorovArnoldNetwork) Forward(x []float64) []*autograd.Value {
	values := make([]*autograd.Value, len(x))
	for i, v := range x {
		values[i] = autograd.NewValue(v, nil)
	}
	for i, layer := range kan.layers {
		values = layer.Forward(values)
		if i == len(kan.layers)-1 {
			// Apply sigmoid to the final layer
			for j, v := range values {
				values[j] = v.Sigmoid()
			}
		}
	}
	return values
}

func (kan *KolmogorovArnoldNetwork) Parameters() []*autograd.Value {
	var params []*autograd.Value
	for _, layer := range kan.layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}
