import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from data_loader import NBALineupDataset
from model import create_model, NBALineupPredictor
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Fixed number of players (0-1085)
FIXED_NUM_PLAYERS = 1086
# Fixed number of teams (0-34)
FIXED_NUM_TEAMS = 35

def get_available_years():
    """Get list of available years from the training datasets directory."""
    data_dir = Path("Project_Datasets/training_datasets_v2")
    years = []
    for file in data_dir.glob("swapped_train_filtered_shuffled_by_game_matchups_*.csv"):
        year = int(file.stem.split("_")[-1])
        years.append(year)
    return sorted(years)

def calculate_metrics(outputs, targets):
    """
    Calculate accuracy, precision, recall, and F1 score.
    
    Args:
        outputs: Model outputs (logits)
        targets: Ground truth labels
    
    Returns:
        Dictionary containing the metrics
    """
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Calculate accuracy
    correct = (predicted == targets).sum()
    total = len(targets)
    accuracy = correct / total
    
    # Calculate precision and recall for each class
    unique_classes = np.unique(targets)
    precisions = []
    recalls = []
    
    for cls in unique_classes:
        # True positives: predicted class is cls and actual class is cls
        tp = ((predicted == cls) & (targets == cls)).sum()
        # False positives: predicted class is cls but actual class is not cls
        fp = ((predicted == cls) & (targets != cls)).sum()
        # False negatives: predicted class is not cls but actual class is cls
        fn = ((predicted != cls) & (targets == cls)).sum()
        
        # Calculate precision and recall for this class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate average precision and recall
    precision = np.mean(precisions)
    recall = np.mean(recalls)
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_metrics(metrics_history, year):
    """
    Plot training and validation metrics.
    
    Args:
        metrics_history: Dictionary containing lists of metrics
        year: The year being trained
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Training Metrics for Year {year}')
    
    # Plot loss
    axes[0, 0].plot(metrics_history['train_loss'], label='Train')
    axes[0, 0].plot(metrics_history['val_loss'], label='Validation')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot accuracy
    axes[0, 1].plot(metrics_history['train_accuracy'], label='Train')
    axes[0, 1].plot(metrics_history['val_accuracy'], label='Validation')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Plot precision and recall
    axes[1, 0].plot(metrics_history['train_precision'], label='Train Precision')
    axes[1, 0].plot(metrics_history['train_recall'], label='Train Recall')
    axes[1, 0].plot(metrics_history['val_precision'], label='Val Precision')
    axes[1, 0].plot(metrics_history['val_recall'], label='Val Recall')
    axes[1, 0].set_title('Precision and Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    
    # Plot F1 score
    axes[1, 1].plot(metrics_history['train_f1'], label='Train')
    axes[1, 1].plot(metrics_history['val_f1'], label='Validation')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'metrics_{year}.png')
    plt.close()

def train_model(train_loader: DataLoader, val_loader: DataLoader, model: NBALineupPredictor, year: int):
    """
    Train the model for a specific year.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: The model to train
        year: The year being trained
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 20
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    # Initialize metrics history
    metrics_history = {
        'train_loss': [], 'val_loss': [],
        'train_accuracy': [], 'val_accuracy': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_outputs = []
        train_targets = []
        
        for features, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_outputs.append(outputs)
            train_targets.append(targets)
        
        # Calculate training metrics
        train_outputs = torch.cat(train_outputs)
        train_targets = torch.cat(train_targets)
        train_metrics = calculate_metrics(train_outputs, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_outputs = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_outputs.append(outputs)
                val_targets.append(targets)
        
        # Calculate validation metrics
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)
        val_metrics = calculate_metrics(val_outputs, val_targets)
        
        # Update metrics history
        metrics_history['train_loss'].append(train_loss / len(train_loader))
        metrics_history['val_loss'].append(val_loss / len(val_loader))
        metrics_history['train_accuracy'].append(train_metrics['accuracy'])
        metrics_history['val_accuracy'].append(val_metrics['accuracy'])
        metrics_history['train_precision'].append(train_metrics['precision'])
        metrics_history['val_precision'].append(val_metrics['precision'])
        metrics_history['train_recall'].append(train_metrics['recall'])
        metrics_history['val_recall'].append(val_metrics['recall'])
        metrics_history['train_f1'].append(train_metrics['f1'])
        metrics_history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Training Loss: {metrics_history['train_loss'][-1]:.4f}")
        print(f"Validation Loss: {metrics_history['val_loss'][-1]:.4f}")
        print(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Training F1: {train_metrics['f1']:.4f}")
        print(f"Validation F1: {val_metrics['f1']:.4f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss and val_metrics['accuracy'] > best_val_acc:
            best_val_loss = val_loss
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            # Save model
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            model_path = models_dir / f"swapped_model_{year}_v2.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': metrics_history['train_loss'][-1],
                'val_loss': metrics_history['val_loss'][-1],
                'train_accuracy': train_metrics['accuracy'],
                'val_accuracy': val_metrics['accuracy'],
                'train_f1': train_metrics['f1'],
                'val_f1': val_metrics['f1'],
            }, str(model_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot metrics at the end of training
    plot_metrics(metrics_history, year)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get available years
    years = get_available_years()
    print(f"Available years: {years}")
    
    # Train model for each year
    for year in years:
        print(f"\nProcessing year {year}")
        
        # Load training data from swapped files
        train_path = Path(f"Project_Datasets/training_datasets_v2/swapped_train_filtered_shuffled_by_game_matchups_{year}.csv")
        dataset = NBALineupDataset(str(train_path))  # Convert Path to string for dataset
        
        # Split into train and validation sets (80/20)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: (
                torch.cat([item[0] for item in x], dim=0),
                torch.cat([item[1] for item in x], dim=0)
            )
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: (
                torch.cat([item[0] for item in x], dim=0),
                torch.cat([item[1] for item in x], dim=0)
            )
        )
        
        print(f"Year {year}:")
        print(f"Number of unique teams: {len(dataset.team_to_idx)}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Create model with fixed number of players and teams
        model = create_model(FIXED_NUM_PLAYERS, FIXED_NUM_TEAMS)
        
        # Train model
        train_model(train_loader, val_loader, model, year)

if __name__ == "__main__":
    main()