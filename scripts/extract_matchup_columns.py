import pandas as pd
import os
import time
import json

def extract_matchup_columns():
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (one level up from scripts)
    project_root = os.path.dirname(script_dir)
    
    # Input and output directories
    input_dir = os.path.join(project_root, 'Project_Datasets', 'raw_datasets')
    output_dir = os.path.join(project_root, 'Project_Datasets', 'encoded_data', 'filtered_matchups')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load team encoding
    team_encoding_path = os.path.join(project_root, 'Project_Datasets', 'encoded_data', 'team_encoding.json')
    with open(team_encoding_path, 'r') as f:
        team_encoding = json.load(f)
    team_to_number = team_encoding['team_to_number']
    
    # Load player encoding
    player_encoding_path = os.path.join(project_root, 'Project_Datasets', 'encoded_data', 'player_encoding.json')
    with open(player_encoding_path, 'r') as f:
        player_encoding = json.load(f)
    player_to_number = player_encoding['player_to_number']
    
    # Columns to keep
    columns_to_keep = [
        'game', 'season', 'home_team', 'away_team', 'starting_min', 
        'home_0', 'home_1', 'home_2', 'home_3', 'home_4',
        'away_0', 'away_1', 'away_2', 'away_3', 'away_4'
    ]
    
    # Process each year
    for year in range(2007, 2016):
        print(f"Processing matchups-{year}.csv...")
        start_time = time.time()
        
        # Input file path
        input_file = os.path.join(input_dir, f'matchups-{year}.csv')
        
        # Read only the columns we need
        df = pd.read_csv(input_file, usecols=columns_to_keep)
        
        # Replace team names with their encoded values
        df['home_team'] = df['home_team'].map(team_to_number)
        df['away_team'] = df['away_team'].map(team_to_number)
        
        # Replace player names with their encoded values
        player_columns = ['home_0', 'home_1', 'home_2', 'home_3', 'home_4',
                         'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
        
        for col in player_columns:
            # Replace player names, handling NaN values
            df[col] = df[col].apply(lambda x: player_to_number.get(x, -1) if pd.notna(x) else -1)
        
        # Output file path
        output_file = os.path.join(output_dir, f'encoded_matchups-{year}.csv')
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Calculate file size reduction
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # Size in MB
        output_size = os.path.getsize(output_file) / (1024 * 1024)  # Size in MB
        reduction = (1 - output_size / input_size) * 100  # Percentage reduction
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"Created {output_file}")
        print(f"Original size: {input_size:.2f} MB")
        print(f"Filtered size: {output_size:.2f} MB")
        print(f"Size reduction: {reduction:.2f}%")
        print(f"Processing time: {processing_time:.2f} seconds")
        print("-" * 50)
        
        # Also create a JSON version
        json_data = df.to_dict(orient='records')
        json_file = os.path.join(output_dir, f'encoded_matchups-{year}.json')
        with open(json_file, 'w') as f:
            json.dump(json_data, f)
        print(f"Created JSON file: {json_file}")
        print("-" * 50)

if __name__ == "__main__":
    extract_matchup_columns() 