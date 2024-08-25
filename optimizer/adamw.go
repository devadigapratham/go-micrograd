package optimizer

import (
	"go-micrograd/autograd"
	"math"
)

type AdamW struct {
	parameters   []*autograd.Value
	lr           float64
	beta1, beta2 float64
	eps          float64
	weightDecay  float64
	t            int
	m, v         map[*autograd.Value]float64
}

func NewAdamW(parameters []*autograd.Value, lr, weightDecay float64) *AdamW {
	return &AdamW{
		parameters:  parameters,
		lr:          lr,
		beta1:       0.9,
		beta2:       0.999,
		eps:         1e-8,
		weightDecay: weightDecay,
		t:           0,
		m:           make(map[*autograd.Value]float64),
		v:           make(map[*autograd.Value]float64),
	}
}

func (opt *AdamW) Step() {
	opt.t++
	for _, p := range opt.parameters {
		if _, ok := opt.m[p]; !ok {
			opt.m[p], opt.v[p] = 0, 0
		}

		grad := p.Grad()
		m := opt.beta1*opt.m[p] + (1-opt.beta1)*grad
		v := opt.beta2*opt.v[p] + (1-opt.beta2)*grad*grad

		mHat := m / (1 - math.Pow(opt.beta1, float64(opt.t)))
		vHat := v / (1 - math.Pow(opt.beta2, float64(opt.t)))

		p.AddToData(-opt.lr * (mHat/(math.Sqrt(vHat)+opt.eps) + opt.weightDecay*p.Data()))

		opt.m[p], opt.v[p] = m, v
	}
}

func (opt *AdamW) ZeroGrad() {
	for _, p := range opt.parameters {
		p.SetGrad(0)
	}
}
