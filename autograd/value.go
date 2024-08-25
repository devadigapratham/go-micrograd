package autograd

import (
	"fmt"
	"math"
)

// Value represents a node in the computation graph.
type Value struct {
	data      float64
	grad      float64
	_prev     []*Value
	_op       string
	_backward func()
}

// NewValue creates a new Value with the specified data and children.
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

// Data returns the data of the Value.
func (v *Value) Data() float64 {
	return v.data
}

// Grad returns the gradient of the Value.
func (v *Value) Grad() float64 {
	return v.grad
}

// AddToData adds a value to the data of the Value.
func (v *Value) AddToData(delta float64) {
	v.data += delta
}

// SetGrad sets the gradient of the Value.
func (v *Value) SetGrad(grad float64) {
	v.grad = grad
}

// Add performs addition and returns a new Value.
func (v *Value) Add(other *Value) *Value {
	out := NewValue(v.data+other.data, []*Value{v, other})

	out._backward = func() {
		v.grad += out.grad
		other.grad += out.grad
	}

	return out
}

// Mul performs multiplication and returns a new Value.
func (v *Value) Mul(other *Value) *Value {
	out := NewValue(v.data*other.data, []*Value{v, other})

	out._backward = func() {
		v.grad += other.data * out.grad
		other.grad += v.data * out.grad
	}

	return out
}

// Pow performs exponentiation and returns a new Value.
func (v *Value) Pow(exp float64) *Value {
	out := NewValue(math.Pow(v.data, exp), []*Value{v})

	out._backward = func() {
		v.grad += exp * math.Pow(v.data, exp-1) * out.grad
	}

	return out
}

// Relu performs the ReLU activation function and returns a new Value.
func (v *Value) Relu() *Value {
	out := NewValue(math.Max(0, v.data), []*Value{v})

	out._backward = func() {
		if out.data > 0 {
			v.grad += out.grad
		}
	}

	return out
}

// Tanh performs the Tanh activation function and returns a new Value.
func (v *Value) Tanh() *Value {
	out := NewValue(math.Tanh(v.data), []*Value{v})

	out._backward = func() {
		v.grad += (1 - out.data*out.data) * out.grad
	}

	return out
}

// Backward performs backpropagation to compute gradients.
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

// String returns a string representation of the Value.
func (v *Value) String() string {
	return fmt.Sprintf("Value(data=%.4f, grad=%.4f)", v.data, v.grad)
}
