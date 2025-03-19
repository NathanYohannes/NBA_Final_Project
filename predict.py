import torch
import torch.nn as nn
from model import NBALineupPredictor, create_model
from data_loader import NBALineupDataset
import pandas as pd
from typing import Dict, List, Tuple
import os
from pathlib import Path

# Fixed number of players (0-1085)
FIXED_NUM_PLAYERS = 1086
# Fixed number of teams (0-34)
FIXED_NUM_TEAMS = 35

def get_available_model_years() -> List[int]:
    """
    Get list of available model years from the models directory.
    
    Returns:
        List of available years as integers
    """
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    available_years = []
    for file in models_dir.glob("swapped_model_*_v2.pth"):
        try:
            year = int(file.stem.split("_")[-2])  # Get year from filename
            available_years.append(year)
        except (ValueError, IndexError):
            continue
    
    return sorted(available_years)

def get_model_year(season: int) -> str:
    """
    Determine which model to use based on the season.
    Uses the closest available model year.
    
    Args:
        season: The season year (e.g., 2015)
    
    Returns:
        String indicating which model to use (e.g., "2015")
    """
    available_years = get_available_model_years()
    if not available_years:
        raise FileNotFoundError("No model files found in the models directory")
    
    # Find the closest year
    closest_year = min(available_years, key=lambda x: abs(x - season))
    return str(closest_year)

def prepare_single_input(
    season: int,
    home_team_idx: int,
    away_team_idx: int,
    starting_min: int,
    home_player_indices: List[int],  # Now expects 5 home players
    away_player_indices: List[int]
) -> torch.Tensor:
    """
    Prepare a single input for prediction using integer indices.
    
    Args:
        season: The season year (e.g., 2015)
        home_team_idx: Home team index
        away_team_idx: Away team index
        starting_min: Starting minute of the game
        home_player_indices: List of 5 home player indices (home_0 to home_4)
        away_player_indices: List of 5 away player indices
    
    Returns:
        Tensor of shape [1, 13] containing the input features
    """
    # Create feature row
    feature_row = []
    
    # Season and minutes (as integers)
    feature_row.append(season)
    
    # Teams (already indices)
    feature_row.append(home_team_idx)
    feature_row.append(away_team_idx)
    feature_row.append(starting_min)
    
    # Players (already indices)
    # Use first 4 home players (home_0 to home_3) and all 5 away players
    feature_row.extend(home_player_indices[:4])  # Only use first 4 home players
    feature_row.extend(away_player_indices)      # Use all 5 away players
    
    # Convert to tensor and add batch dimension
    features = torch.tensor(feature_row, dtype=torch.float32).unsqueeze(0)
    
    return features

def predict_player(season: int, home_team_idx: int, away_team_idx: int, starting_min: int,
                  home_player_indices: List[int], away_player_indices: List[int]) -> Tuple[int, float]:
    """
    Predict the next player to enter the game.
    
    Args:
        season: The season year
        home_team_idx: Index of the home team
        away_team_idx: Index of the away team
        starting_min: Starting minute of the game
        home_player_indices: List of 5 home player indices
        away_player_indices: List of 5 away player indices
    
    Returns:
        Tuple of (predicted_player_idx, confidence)
    """
    # Get the appropriate model year
    model_year = get_model_year(season)
    
    # Create model with fixed dimensions
    model = create_model(FIXED_NUM_PLAYERS, FIXED_NUM_TEAMS)
    
    # Load the model weights
    model_path = Path("models") / f"swapped_model_{model_year}_v2.pth"
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Prepare input features
    features = torch.tensor([
        season,
        home_team_idx,
        away_team_idx,
        starting_min,
        *home_player_indices,
        *away_player_indices
    ], dtype=torch.float32).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    return predicted_idx.item(), confidence.item()

def main():
    # Example usage with integer indices
    season = 2015
    home_team_idx = 14  # Example: LAL
    away_team_idx = 1   # Example: BOS
    starting_min = 0
    
    # Example player indices (you'll need to replace these with actual indices)
    # Now including all 5 home players
    home_player_indices = [100, 101, 102, 103, 104]  # home_0 to home_4
    away_player_indices = [200, 201, 202, 203, 204]
    
    try:
        predicted_player_idx, confidence = predict_player(
            season, home_team_idx, away_team_idx, starting_min,
            home_player_indices, away_player_indices
        )
        print(f"\nPrediction for season {season}:")
        print(f"Using model: swapped_model_{get_model_year(season)}_v2.pth")
        print(f"Home Team Index: {home_team_idx}")
        print(f"Away Team Index: {away_team_idx}")
        print(f"Input home players (0-3): {home_player_indices[:4]}")
        print(f"Actual home_4: {home_player_indices[4]}")
        print(f"Predicted home_4: {predicted_player_idx}")
        print(f"Confidence: {confidence:.2%}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main() 