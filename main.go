package main

import (
	"fmt"
	"go-micrograd/dataset"
	"go-micrograd/nn"
	"go-micrograd/optimizer"
	"go-micrograd/rng"
	"time"
)

func convertPoints(points []dataset.Point) []nn.DataPoint {
	result := make([]nn.DataPoint, len(points))
	for i, p := range points {
		result[i] = nn.DataPoint{
			X:     float64(p.X), // Convert float32 to float64
			Y:     float64(p.Y), // Convert float32 to float64
			Label: 0,            // Default label or handle accordingly
		}
	}
	return result
}

func main() {
	random := rng.NewRNG(42)
	trainPoints, valPoints, _ := dataset.GenDataYinYang(random, 100)

	trainSplit := convertPoints(trainPoints)
	valSplit := convertPoints(valPoints)

	mlpModel := nn.NewMLP(2, []int{8, 3}, random)
	kanModel := nn.NewKAN(2, []int{8, 3}, random)

	mlpOptimizer := optimizer.NewAdamW(mlpModel.Parameters(), 0.1, 1e-3)
	kanOptimizer := optimizer.NewAdamW(kanModel.Parameters(), 0.1, 1e-3)

	numSteps := 100
	mlpTrainLosses := make([]float64, numSteps)
	kanTrainLosses := make([]float64, numSteps)
	mlpValLosses := make([]float64, numSteps/10)
	kanValLosses := make([]float64, numSteps/10)

	mlpStartTime := time.Now()
	kanStartTime := time.Now()

	for step := 0; step < numSteps; step++ {
		mlpTrainLoss := nn.LossFunction(mlpModel, trainSplit)
		mlpTrainLoss.Backward()
		mlpOptimizer.Step()
		mlpOptimizer.ZeroGrad()
		mlpTrainLosses[step] = mlpTrainLoss.Data()

		kanTrainLoss := nn.LossFunction(kanModel, trainSplit)
		kanTrainLoss.Backward()
		kanOptimizer.Step()
		kanOptimizer.ZeroGrad()
		kanTrainLosses[step] = kanTrainLoss.Data()

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
