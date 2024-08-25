package main

import (
	"fmt"
	"go-micrograd/dataset"
	"go-micrograd/nn"
	"go-micrograd/optimizer"
	"go-micrograd/rng"
	"time"
)

func main() {
	// Initialize the random number generatorgg
	random := rng.NewRNG(42)

	// Generate dataset
	trainSplit, valSplit, _ := dataset.GenDataYinYang(random, 100)

	// Initialize both models: MLP and KAN
	mlpModel := nn.NewMLP(2, []int{8, 3}, random)
	kanModel := nn.NewKAN(2, []int{8, 3}, random)

	// Initialize optimizers for both models
	mlpOptimizer := optimizer.NewAdamW(mlpModel.Parameters(), 0.1, 1e-4)
	kanOptimizer := optimizer.NewAdamW(kanModel.Parameters(), 0.1, 1e-4)

	// Benchmark variables
	numSteps := 100
	mlpTrainLosses, kanTrainLosses := make([]float64, numSteps), make([]float64, numSteps)
	mlpValLosses, kanValLosses := make([]float64, numSteps/10), make([]float64, numSteps/10)

	// Time measurements
	mlpStartTime := time.Now()
	kanStartTime := time.Now()

	// Training loop for both models
	for step := 0; step < numSteps; step++ {
		// MLP Training
		mlpTrainLoss := nn.LossFunction(mlpModel, trainSplit)
		mlpTrainLoss.Backward()
		mlpOptimizer.Step()
		mlpOptimizer.ZeroGrad()
		mlpTrainLosses[step] = mlpTrainLoss.Data()

		// KAN Training
		kanTrainLoss := nn.LossFunction(kanModel, trainSplit)
		kanTrainLoss.Backward()
		kanOptimizer.Step()
		kanOptimizer.ZeroGrad()
		kanTrainLosses[step] = kanTrainLoss.Data()

		// Every 10 steps, evaluate validation loss
		if step%10 == 0 {
			mlpValLoss := nn.LossFunction(mlpModel, valSplit)
			mlpValLosses[step/10] = mlpValLoss.Data()

			kanValLoss := nn.LossFunction(kanModel, valSplit)
			kanValLosses[step/10] = kanValLoss.Data()

			fmt.Printf("Step %d/%d: MLP Val Loss = %.6f, KAN Val Loss = %.6f\n",
				step+1, numSteps, mlpValLoss.Data(), kanValLoss.Data())
		}
	}

	mlpTotalTime := time.Since(mlpStartTime)
	kanTotalTime := time.Since(kanStartTime)

	// Final benchmark summary
	fmt.Println("\n==== Benchmark Results ====")
	fmt.Printf("MLP Training Time: %v\n", mlpTotalTime)
	fmt.Printf("KAN Training Time: %v\n", kanTotalTime)

	fmt.Println("\nMLP Loss Progression:")
	for step := 0; step < numSteps; step++ {
		fmt.Printf("Step %d: Train Loss = %.6f\n", step+1, mlpTrainLosses[step])
	}
	fmt.Println("\nKAN Loss Progression:")
	for step := 0; step < numSteps; step++ {
		fmt.Printf("Step %d: Train Loss = %.6f\n", step+1, kanTrainLosses[step])
	}

	fmt.Println("\nMLP Validation Loss:")
	for step := 0; step < numSteps/10; step++ {
		fmt.Printf("Step %d: Val Loss = %.6f\n", (step+1)*10, mlpValLosses[step])
	}

	fmt.Println("\nKAN Validation Loss:")
	for step := 0; step < numSteps/10; step++ {
		fmt.Printf("Step %d: Val Loss = %.6f\n", (step+1)*10, kanValLosses[step])
	}
}
