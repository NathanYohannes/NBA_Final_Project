import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from pathlib import Path

class NBALineupPredictor(nn.Module):
    def __init__(self, num_players: int, num_teams: int, embedding_dim: int = 64, hidden_dim: int = 128):
        """
        Initialize the NBA Lineup Predictor model.
        
        Args:
            num_players: Number of unique players in the dataset
            num_teams: Number of unique teams in the dataset
            embedding_dim: Dimension of player and team embeddings
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Embeddings
        self.player_embedding = nn.Embedding(num_players, embedding_dim)
        self.team_embedding = nn.Embedding(num_teams, embedding_dim)
        
        # Neural network layers
        # Input: 9 player embeddings (4 home + 5 away) + 2 team embeddings + 2 scalar features
        input_dim = (9 * embedding_dim) + (2 * embedding_dim) + 2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_players)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            features: Tensor of shape [batch_size, 13] containing:
                     [season, home_team, away_team, starting_min,
                      home_0, home_1, home_2, home_3,
                      away_0, away_1, away_2, away_3, away_4]
        
        Returns:
            Tensor of shape [batch_size, num_players] containing player probabilities
        """
        batch_size = features.size(0)
        
        # Extract non-player features
        season = features[:, 0].unsqueeze(1) / 2000.0  # Normalize season
        starting_min = features[:, 3].unsqueeze(1) / 48.0  # Normalize minutes
        
        # Get embeddings for all players and teams
        home_team_emb = self.team_embedding(features[:, 1].long())
        away_team_emb = self.team_embedding(features[:, 2].long())
        
        # Get embeddings for all players (4 home + 5 away)
        home_players_emb = self.player_embedding(features[:, 4:8].long())
        away_players_emb = self.player_embedding(features[:, 8:13].long())
        
        # Flatten and concatenate all embeddings
        x = torch.cat([
            home_team_emb,
            away_team_emb,
            home_players_emb.reshape(batch_size, -1),
            away_players_emb.reshape(batch_size, -1),
            season,
            starting_min
        ], dim=1)
        
        # Forward through neural network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

def create_model(num_players: int, num_teams: int) -> NBALineupPredictor:
    """
    Create and initialize the model.
    
    Args:
        num_players: Number of unique players in the dataset
        num_teams: Number of unique teams in the dataset
    
    Returns:
        Initialized NBALineupPredictor model
    """
    model = NBALineupPredictor(num_players, num_teams)
    return model

def load_model(year: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> Tuple[NBALineupPredictor, dict]:
    """
    Load a trained model for a specific year.
    
    Args:
        year: The year of the model to load
        device: Device to load the model on
    
    Returns:
        Tuple of (model, checkpoint)
    """
    model_path = Path("models") / f"model_{year}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for year {year}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with same dimensions as saved model
    model = create_model(
        num_players=checkpoint['model_state_dict']['player_embedding.weight'].size(0),
        num_teams=checkpoint['model_state_dict']['team_embedding.weight'].size(0)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint

# Example usage:
if __name__ == "__main__":
    # Example dimensions
    num_players = 500  # Replace with actual number from dataset
    num_teams = 30     # Replace with actual number from dataset
    
    # Create model
    model = create_model(num_players, num_teams)
    
    # Create example input
    batch_size = 32
    features = torch.randint(0, num_players, (batch_size, 13))
    
    # Forward pass
    output = model(features)
    
    # Print shapes
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test model loading
    try:
        model, checkpoint = load_model("2015")
        print("\nSuccessfully loaded model for 2015")
        print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    except FileNotFoundError as e:
        print(f"\n{str(e)}")