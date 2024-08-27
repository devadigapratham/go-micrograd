# go-micrograd

`go-micrograd` is a Go implementation of an automatic differentiation engine, inspired by micrograd. This project demonstrates fundamental concepts in machine learning, such as gradient-based optimization, by implementing a simple autograd system.

## Features

- **Automatic Differentiation**: Supports basic operations and computes gradients automatically.
- **Gradient Descent**: Enables training simple models by utilizing gradient-based optimization.
- **Dataset Utilities**: Provides dataset structures for training models.
- **KAN and MLP Benchmarking**: Benchmarks between Kolmogorov Arnold Network (KAN) and Multilayer Perceptron (MLP) models are implemented and compared in the project.

## Project Structure

- **`main.go`**: The entry point of the application. It initializes the system and runs the core logic.
- **`go.mod`**: Go module file containing project dependencies and Go version.
- **`/autograd`**: Contains the core logic for automatic differentiation, including the `Value` struct, which tracks data and its gradients.
- **`/dataset`**: Includes utilities for managing datasets like the `yinyang.go`, which likely contains data generation or handling logic for a Yin-Yang dataset.

## Installation

To install and run this project, you need to have Go installed on your machine.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/go-micrograd.git
   cd go-micrograd
   ```

2. Install dependencies:
   ```bash
   go mod tidy
   ```

3. Run the project:
   ```bash
   go run main.go
   ```

## Usage

The project is designed to work with small datasets and demonstrate automatic differentiation. You can adjust the core logic in `main.go` to customize the model training process or to experiment with different datasets.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
