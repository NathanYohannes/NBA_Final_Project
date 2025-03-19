import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path

class NBALineupDataset(Dataset):
    def __init__(self, csv_path: str):
        """
        Initialize the NBA Lineup Dataset.
        
        Args:
            csv_path: Path to the lineup data CSV
        """
        # Load the lineup data
        self.df = pd.read_csv(csv_path)
        
        # Group data by game
        self.games = self.df.groupby('game')
        
        # Create list of game IDs for indexing
        self.game_ids = list(self.games.groups.keys())
        
        # Create team ID to index mapping
        all_teams = set(self.df['home_team'].unique()) | set(self.df['away_team'].unique())
        self.team_to_idx = {team: idx for idx, team in enumerate(sorted(all_teams))}
        
        # Use fixed number of players (0-1085)
        self.num_players = 1086
        
        print(f"Dataset loaded from {csv_path}")
        print(f"Number of unique teams: {len(self.team_to_idx)}")
        print(f"Number of games: {len(self.game_ids)}")
        print(f"Number of lineup changes: {len(self.df)}")
    
    def __len__(self) -> int:
        return len(self.game_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get game ID
        game_id = self.game_ids[idx]
        
        # Get all lineups for this game
        game_data = self.games.get_group(game_id)
        
        # Create feature matrix with all columns except 'game' and 'home_4'
        feature_columns = ['season', 'home_team', 'away_team', 'starting_min',
                         'home_0', 'home_1', 'home_2', 'home_3',
                         'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
        
        # Convert player IDs to indices for player columns
        features = []
        for _, row in game_data.iterrows():
            feature_row = []
            # Season and minutes
            feature_row.append(row['season'])
            # Teams (convert to indices)
            feature_row.append(self.team_to_idx[row['home_team']])
            feature_row.append(self.team_to_idx[row['away_team']])
            feature_row.append(row['starting_min'])
            # Players (use original indices)
            for col in ['home_0', 'home_1', 'home_2', 'home_3',
                       'away_0', 'away_1', 'away_2', 'away_3', 'away_4']:
                feature_row.append(row[col])
            features.append(feature_row)
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # Create target (home_4 player)
        targets = torch.tensor([pid for pid in game_data['home_4']], dtype=torch.long)
        
        return features, targets

def get_available_years() -> List[str]:
    """Get list of available years from the training datasets directory."""
    training_path = Path("Project_Datasets/training_datasets")
    years = []
    for file in training_path.glob("train_filtered_shuffled_by_game_matchups_*.csv"):
        year = file.stem.split('_')[-1]
        years.append(year)
    return sorted(years)

def create_dataloader(csv_path: str, batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, int, int]:
    """
    Create a DataLoader that batches data by game.
    
    Args:
        csv_path: Path to the CSV file containing the lineup data
        batch_size: Number of games to include in each batch
        shuffle: Whether to shuffle the games
    
    Returns:
        DataLoader object, number of players, number of teams
    """
    dataset = NBALineupDataset(csv_path)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda x: (
            torch.cat([item[0] for item in x], dim=0),  # features
            torch.cat([item[1] for item in x], dim=0)   # targets
        )
    ), dataset.num_players, len(dataset.team_to_idx)

# Example usage:
if __name__ == "__main__":
    # Get available years
    years = get_available_years()
    print(f"Available years: {years}")
    
    # Test data loading for each year
    for year in years:
        print(f"\nTesting data loading for year {year}")
        train_path = f"Project_Datasets/training_datasets/train_filtered_shuffled_by_game_matchups_{year}.csv"
        test_path = f"Project_Datasets/testing_datasets/test_filtered_shuffled_by_game_matchups_{year}.csv"
        
        # Create dataloaders
        train_loader, num_players, num_teams = create_dataloader(train_path, batch_size=2)
        test_loader, _, _ = create_dataloader(test_path, batch_size=2)
        
        # Test the dataloaders
        print(f"Training data:")
        for batch_idx, (features, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            break
        
        print(f"\nTesting data:")
        for batch_idx, (features, targets) in enumerate(test_loader):
            print(f"Batch {batch_idx}:")
            print(f"Features shape: {features.shape}")
            print(f"Targets shape: {targets.shape}")
            break