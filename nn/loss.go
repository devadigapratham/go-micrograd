package nn

import (
	"go-micrograd/autograd"
	"math"
)

type DataPoint struct {
	X, Y  float64
	Label int
}

func LossFunction(model Model, data []DataPoint) *autograd.Value {
	totalLoss := autograd.NewValue(0, nil)
	for _, point := range data {
		input := []float64{point.X, point.Y}
		output := model.Forward(input)[0]
		target := float64(point.Label)

		// Clip output to avoid log(0)
		outputClipped := math.Max(math.Min(output.Data(), 1-1e-7), 1e-7)

		// Binary cross-entropy loss with clipping
		loss := -(target*math.Log(outputClipped) + (1-target)*math.Log(1-outputClipped))
		totalLoss = totalLoss.Add(autograd.NewValue(loss, nil))
	}
	return totalLoss.Div(autograd.NewValue(float64(len(data)), nil))
}
