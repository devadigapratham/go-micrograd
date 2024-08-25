package autograd

import (
	"fmt"
	"math"
)

type Value struct {
	data      float64
	grad      float64
	_prev     []*Value
	_op       string
	_backward func()
}

func NewValue(data float64, children []*Value) *Value {
	return &Value{
		data:  data,
		grad:  0,
		_prev: children,
		_op:   "",
		_backward: func() {
		},
	}
}

func (v *Value) Data() float64 {
	return v.data
}

func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.data+other.data, []*Value{v, other})

	out._backward = func() {
		v.grad += out.grad
		other.grad += out.grad
	}

	return out
}

func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data*other.data, []*Value{v, other})

	out._backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}

	return out
}

func (v *Value) Pow(exp float64) *Value {
	out := NewValue(math.Pow(v.data, exp), []*Value{v})

	out._backward = func() {
		v.grad += exp * math.Pow(v.data, exp-1) * out.grad
	}

	return out
}

func (v *Value) Relu() *Value {
	out := NewValue(math.Max(0, v.data), []*Value{v})

	out._backward = func() {
		if out.data > 0 {
			v.grad += out.grad
		}
	}

	return out
}

func (v *Value) Tanh() *Value {
	out := NewValue(math.Tanh(v.data), []*Value{v})

	out._backward = func() {
		v.grad += (1 - out.data*out.data) * out.grad
	}

	return out
}

func (v *Value) Backward() {
	// Topologically sort nodes
	topo := []*Value{}
	visited := map[*Value]bool{}
	var buildTopo func(*Value)
	buildTopo = func(node *Value) {
		if !visited[node] {
			visited[node] = true
			for _, child := range node._prev {
				buildTopo(child)
			}
			topo = append(topo, node)
		}
	}
	buildTopo(v)

	// Set gradient of root to 1
	v.grad = 1.0
	// Apply chain rule in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._backward()
	}
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%.4f, grad=%.4f)", v.data, v.grad)
}
