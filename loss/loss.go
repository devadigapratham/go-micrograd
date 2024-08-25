package loss

import "go-micrograd/autograd"

// LossFunction represents a Mean Squared Error (MSE) loss function.
type LossFunction struct {
}

// NewLossFunction creates a new instance of LossFunction.
func NewLossFunction() *LossFunction {
	return &LossFunction{}
}

// Forward computes the Mean Squared Error (MSE) loss given predictions and targets.
func (lf *LossFunction) Forward(predictions, targets *autograd.Value) *autograd.Value {
	// Ensure predictions and targets are of the same shape
	if predictions.Data() != targets.Data() {
		panic("Predictions and targets must have the same shape")
	}

	// Compute the Mean Squared Error (MSE)
	diff := predictions.Sub(targets) // Compute difference between predictions and targets
	squaredDiff := diff.Pow(2)       // Square the difference
	loss := squaredDiff              // For a single value, we return squaredDiff. For batch, compute mean.
	return loss
}

// Backward computes the gradient of the MSE loss with respect to predictions.
func (lf *LossFunction) Backward(predictions, targets *autograd.Value) {
	// The gradient of MSE with respect to predictions is 2 * (predictions - targets) / number of samples.
	// Since we're dealing with a single value here, we'll handle it directly.
	diff := predictions.Sub(targets) // Compute difference between predictions and targets
	grad := diff.Mul(2)              // Compute gradient
	predictions.SetGrad(grad.Data()) // Set the gradient on predictions
}
