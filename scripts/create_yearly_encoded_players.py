import pandas as pd
import os
import json

def create_yearly_encoded_players():
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load player encoding
    player_encoding_path = os.path.join(script_dir, 'encoded_data', 'player_encoding.json')
    with open(player_encoding_path, 'r') as f:
        player_encoding = json.load(f)
    
    player_to_number = player_encoding['player_to_number']
    
    # Load team encoding
    team_encoding_path = os.path.join(script_dir, 'encoded_data', 'team_encoding.json')
    with open(team_encoding_path, 'r') as f:
        team_encoding = json.load(f)
    
    team_to_number = team_encoding['team_to_number']
    
    # Create output directory
    output_dir = os.path.join(script_dir, 'encoded_data', 'encoded_yearly_available_players')
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each year
    for year in range(2007, 2016):
        print(f"Processing year {year}...")
        
        # Load the team players file for this year
        input_file = os.path.join(script_dir, 'available_players', f'team_players_{year}.csv')
        df = pd.read_csv(input_file)
        
        # Create a new dataframe for encoded data
        encoded_data = []
        
        # Process each team
        for _, row in df.iterrows():
            team = row['Team']
            team_number = team_to_number[team]
            
            # Split the players string and convert each player to its number
            players_str = row['Players']
            players = [p.strip() for p in players_str.split(',')]
            encoded_players = [player_to_number[player] for player in players]
            
            # Add to encoded data
            encoded_data.append({
                'team': team_number,
                'players': encoded_players
            })
        
        # Save as JSON
        output_file = os.path.join(output_dir, f'encoded_players_{year}.json')
        with open(output_file, 'w') as f:
            json.dump(encoded_data, f, indent=4)
        
        print(f"Created encoded file for {year} at: {output_file}")
        
        # Also create a CSV version for easier viewing
        csv_data = []
        for item in encoded_data:
            csv_data.append({
                'Team': item['team'],
                'Players': ', '.join(map(str, item['players']))
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, f'encoded_players_{year}.csv')
        csv_df.to_csv(csv_file, index=False)
        print(f"Created CSV file for {year} at: {csv_file}")
    
    print("All years processed successfully!")

if __name__ == "__main__":
    create_yearly_encoded_players() 