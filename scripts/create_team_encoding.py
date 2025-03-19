import pandas as pd
import os
import json

def get_all_teams():
    # Get all unique team names across all years
    all_teams = set()
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    players_dir = os.path.join(script_dir, 'available_players')
    
    # Process each year's file
    for year in range(2007, 2016):
        file_path = os.path.join(players_dir, f'team_players_{year}.csv')
        df = pd.read_csv(file_path)
        all_teams.update(df['Team'].unique())
    
    return sorted(list(all_teams))

def create_team_encoding():
    teams = get_all_teams()
    
    # Create mapping dictionary
    team_to_number = {team: idx for idx, team in enumerate(teams)}
    number_to_team = {idx: team for idx, team in enumerate(teams)}
    
    # Create the encoding dictionary with both mappings
    encoding = {
        'team_to_number': team_to_number,
        'number_to_team': number_to_team
    }
    
    # Save to JSON file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, 'available_players', 'team_encoding.json')
    
    with open(output_file, 'w') as f:
        json.dump(encoding, f, indent=4)
    
    print(f"Created team encoding file at: {output_file}")
    print("\nTeam to number mapping:")
    for team, number in team_to_number.items():
        print(f"{team}: {number}")

if __name__ == "__main__":
    create_team_encoding() 