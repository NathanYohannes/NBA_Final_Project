import pandas as pd
import os
import json

def get_all_players():
    # Get all unique player names across all years
    all_players = set()
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    players_dir = os.path.join(script_dir, 'available_players')
    
    # Process each year's file
    for year in range(2007, 2016):
        file_path = os.path.join(players_dir, f'team_players_{year}.csv')
        df = pd.read_csv(file_path)
        
        # Each row has a 'Players' column with comma-separated player names
        for players_str in df['Players']:
            # Split the string into individual player names and add to set
            players = [p.strip() for p in players_str.split(',')]
            all_players.update(players)
    
    return sorted(list(all_players))

def create_player_encoding():
    players = get_all_players()
    
    # Create mapping dictionaries
    player_to_number = {player: idx for idx, player in enumerate(players)}
    number_to_player = {idx: player for idx, player in enumerate(players)}
    
    # Create the encoding dictionary with both mappings
    encoding = {
        'player_to_number': player_to_number,
        'number_to_player': number_to_player,
        'total_players': len(players)
    }
    
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save to available_players directory
    available_dir = os.path.join(script_dir, 'available_players')
    csv_file = os.path.join(available_dir, 'player_encoding.csv')
    
    # Save as CSV for easier viewing
    csv_data = []
    for player, number in player_to_number.items():
        csv_data.append({'Player': player, 'Number': number})
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)
    print(f"Created CSV encoding file at: {csv_file}")
    
    # Save to encoded_data directory
    encoded_dir = os.path.join(script_dir, 'encoded_data')
    os.makedirs(encoded_dir, exist_ok=True)
    json_file = os.path.join(encoded_dir, 'player_encoding.json')
    
    with open(json_file, 'w') as f:
        json.dump(encoding, f, indent=4)
    print(f"Created JSON encoding file at: {json_file}")
    
    # Create a more compact version with just player_to_number
    compact_encoding = {
        'player_to_number': player_to_number,
        'total_players': len(players)
    }
    compact_json_file = os.path.join(encoded_dir, 'player_encoding_compact.json')
    with open(compact_json_file, 'w') as f:
        json.dump(compact_encoding, f, indent=4)
    print(f"Created compact JSON encoding file at: {compact_json_file}")
    
    print(f"\nTotal unique players: {len(players)}")
    print("\nSample of player to number mapping (first 10):")
    for player, number in list(player_to_number.items())[:10]:
        print(f"{player}: {number}")

if __name__ == "__main__":
    create_player_encoding() 