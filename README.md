# NBA Lineup Predictor

A deep learning model that predicts NBA player lineups based on historical game data. This project uses PyTorch to build a neural network that can predict which players are likely to be in a game's lineup given certain game conditions.

## Project Overview

This project implements a neural network model that predicts NBA player lineups by analyzing various game features including:
- Season information
- Home and away teams
- Starting minutes
- Current lineup players (4 home + 5 away players)

The model uses embeddings to represent players and teams, combined with a neural network architecture to make predictions.


## Features

- Player and team embeddings for better feature representation
- Multi-layer neural network architecture
- Comprehensive metrics tracking (accuracy, precision, recall, F1 score)
- Early stopping to prevent overfitting
- Support for multiple seasons of data
- Visualization of training metrics

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd NBA_Prediction_Final
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch numpy matplotlib pandas
```

## Usage

### Training the Model

To train the model on a specific year's data:

```bash
python train_v2.py
```

The training script will:
- Load the appropriate dataset
- Train the model with early stopping
- Save the best model checkpoint
- Generate training metrics visualizations

### Making Predictions

To make predictions using the trained model:

```bash
python predict_player.py \Project_Datasets\validation_dataset\NBA_test_updated.csv
```

## Model Architecture

The model uses:
- Player embeddings (64 dimensions)
- Team embeddings (64 dimensions)
- Three fully connected layers with ReLU activation
- Dropout for regularization
- Cross-entropy loss function
- Adam optimizer

## Performance Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score

Training progress and metrics are visualized and saved as plots during training.


