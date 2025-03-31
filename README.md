# Kuramoto-Sivashinsky Equation Neural Network Solver

This project implements a neural network to predict the evolution of the Kuramoto-Sivashinsky (KS) equation. The implementation includes both a numerical solver and a neural network that learns to predict the system's evolution.

## Setup Instructions

1. First, ensure you have Conda installed on your system. If not, you can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

2. Create and activate a new conda environment:
```bash
conda create -n ks_env python=3.8
conda activate ks_env
```

3. Install the required packages using pip:
```bash
pip install -r requirements.txt
```

## Running the Code

The project consists of a single Python script `task1.py` that performs multiple tasks:

1. **Run the Original KS Equation Solver**:
   - The script will first solve the KS equation using a numerical method
   - It will generate and save a 3D plot of the solution as 'ks_original.png'

2. **Train the Neural Network**:
   - The neural network will be trained on the original solution
   - Training progress will be displayed showing the loss every 10 epochs

3. **Test Different Initial Conditions**:
   - The script will test the trained neural network on three different initial conditions:
     - Different frequency: cos(x/8)(1 + sin(x/8))
     - Pure sine: sin(x/16)
     - Pure cosine: cos(x/16)
   - For each condition, it will generate comparison plots saved as 'ks_comparison.png'

To run the entire pipeline, simply execute:
```bash
python task1.py
```

## Output Files

The script generates several visualization files:

1. `ks_original.png`: Shows the original KS equation solution used for training
2. `ks_comparison.png`: Shows comparisons between exact solutions and neural network predictions for different initial conditions

## Expected Runtime

- The numerical solver takes approximately 1-2 minutes to run
- Neural network training takes about 2-3 minutes on a CPU
- Total runtime is approximately 5-7 minutes

## Notes

- The neural network architecture uses fully connected layers with ReLU activation
- The training process uses the Adam optimizer with a learning rate of 0.001
- Some numerical warnings about overflow may appear during execution - these are expected due to the chaotic nature of the KS equation and don't affect the results 