from predict import predict_player, get_available_model_years
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def calculate_metrics(predictions, actuals):
    """
    Calculate accuracy, precision, recall, and F1 score for multi-class classification.
    
    Args:
        predictions: List of predicted player indices
        actuals: List of actual player indices
    
    Returns:
        Dictionary containing the metrics
    """
    # Convert to numpy arrays for easier calculation
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    # Get unique players
    unique_players = np.unique(np.concatenate([preds, acts]))
    
    # Initialize metrics for each player
    player_metrics = {}
    
    for player in unique_players:
        # For each player, calculate:
        # True Positives: When we correctly predict this player
        # False Positives: When we predict this player but it's wrong
        # False Negatives: When this player was the actual but we predicted someone else
        
        # True Positives: Correctly predicted this player
        true_positives = np.sum((preds == player) & (acts == player))
        
        # False Positives: Predicted this player but it was wrong
        false_positives = np.sum((preds == player) & (acts != player))
        
        # False Negatives: This player was actual but we predicted someone else
        false_negatives = np.sum((preds != player) & (acts == player))
        
        # Calculate metrics for this player
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        player_metrics[player] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate overall metrics
    total_predictions = len(preds)
    accuracy = np.sum(preds == acts) / total_predictions
    
    # Average precision, recall, and F1 across all players
    avg_precision = np.mean([m['precision'] for m in player_metrics.values()])
    avg_recall = np.mean([m['recall'] for m in player_metrics.values()])
    avg_f1 = np.mean([m['f1'] for m in player_metrics.values()])
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }

def plot_metrics(metrics, save_path=None):
    """
    Plot the metrics in a bar chart.
    
    Args:
        metrics: Dictionary containing the metrics
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics.keys(), metrics.values())
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 0.5)
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def calculate_accuracy(csv_path: str) -> float:
    """
    Calculate prediction accuracy using a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing test data
        
    Returns:
        Accuracy as a decimal
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    correct_predictions = 0
    total_predictions = 0
    failed_predictions = 0
    error_shown = False
    
    # Lists to store predictions and actuals for metrics
    all_predictions = []
    all_actuals = []
    
    print("\n" + "="*50)
    print(f"Processing predictions for: {csv_path}")
    print("="*50 + "\n")
    
    # Process each row with a clean progress bar
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Making predictions", ncols=100, leave=True):
        try:
            # Extract features from the row
            season = row['season']
            home_team_idx = row['home_team']
            away_team_idx = row['away_team']
            starting_min = row['starting_min']
            
            # Get player indices
            home_player_indices = [
                row['home_0'],
                row['home_1'],
                row['home_2'],
                row['home_3'],
                row['home_4']
            ]
            away_player_indices = [
                row['away_0'],
                row['away_1'],
                row['away_2'],
                row['away_3'],
                row['away_4']
            ]
            
            # Make prediction
            predicted_idx, confidence = predict_player(
                season=season,
                home_team_idx=home_team_idx,
                away_team_idx=away_team_idx,
                starting_min=starting_min,
                home_player_indices=home_player_indices,
                away_player_indices=away_player_indices
            )
            
            # Store prediction and actual for metrics
            all_predictions.append(predicted_idx)
            all_actuals.append(row.get('home_4', predicted_idx))
            
            # Compare prediction with actual
            if predicted_idx == row.get('home_4', predicted_idx):
                correct_predictions += 1
            total_predictions += 1
            
        except Exception as e:
            if not error_shown:
                print(f"\nError encountered: {str(e)}")
                error_shown = True
            failed_predictions += 1
            continue
    
    if total_predictions == 0:
        print("\nNo predictions were made successfully.")
        return 0.0
    
    # Calculate all metrics
    metrics = calculate_metrics(all_predictions, all_actuals)
    
    # Plot metrics
    plot_metrics(metrics, save_path='metrics.png')
    
    print("\n" + "="*50)
    print("Prediction Results:")
    print("="*50)
    print(f"Total Predictions: {total_predictions:,}")
    print(f"Correct Predictions: {correct_predictions:,}")
    print(f"Failed Predictions: {failed_predictions:,}")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1 Score: {metrics['f1']:.2f}")
    print("="*50 + "\n")
    print("Metrics plot saved as 'metrics.png'")
    
    return metrics['accuracy']

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # If CSV file is provided, calculate accuracy
        csv_path = sys.argv[1]
        calculate_accuracy(csv_path)
    else:
        # If no arguments provided, show example prediction
        print("\n" + "="*50)
        print("NBA Lineup Predictor")
        print("="*50)
        
        try:
            # Show available models
            available_years = get_available_model_years()
            if available_years:
                print("\nAvailable model years:")
                print("-" * 20)
                for year in available_years:
                    print(f"• {year}")
                print("-" * 20 + "\n")
            
            print("Making example prediction...")
            predicted_player_idx, confidence = predict_player(
                season=2007,
                home_team_idx=14,  # LAL
                away_team_idx=27,  # PHO
                starting_min=0,
                home_player_indices=[51, 617, 651, 882, 915],  # Valid LAL players from 2007
                away_player_indices=[104, 606, 813, 909, 931]  # Valid PHO players from 2007
            )
            
            print("\nPrediction Results:")
            print("-" * 20)
            print(f"Predicted Player Index: {predicted_player_idx}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 20)
            
        except Exception as e:
            print(f"\nError in example prediction: {str(e)}")
        
        print("\nUsage:")
        print("-" * 20)
        print("To calculate accuracy using a CSV file:")
        print("python predict_player.py your_test_file.csv")
        print("-" * 20 + "\n")
