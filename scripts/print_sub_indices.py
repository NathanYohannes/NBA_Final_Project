import csv
from collections import defaultdict
from pathlib import Path

def get_home_players(row):
    # Get all home team players as a set for comparison
    return set(row[5:10])

def print_swapped_players(row, swap_indices):
    """
    Swap players in a given row and return the modified row.
    
    Args:
        row: A list containing the CSV row data
        swap_indices: List of indices (0-4) where swaps occurred
    Returns:
        Modified row with swapped players
    """
    game = row[0]
    minute = row[4]
    
    print(f"\nSwap Details:")
    print(f"Game: {game}")
    print(f"Minute: {minute}")
    
    for idx in swap_indices:
        swapped_player = row[5 + idx]
        print(f"Position {idx}: Player {swapped_player}")
        temp = row[5 + idx]
        row[5+idx] = row[9]
        row[9] = temp
    
    print(f"Full home team lineup: {', '.join(row[5:10])}")
    return row

def process_file(input_file: Path) -> None:
    """
    Process a single CSV file to find and print substitution indices.
    Creates a new file with swapped players.
    """
    # Create output filename by adding 'swapped_' prefix
    output_file = input_file.parent / f"swapped_{input_file.name}"
    
    # Dictionary to store substitutions for each game
    game_subs = defaultdict(list)
    current_game = None
    previous_players = None
    
    # Store all rows including header
    all_rows = []
    
    with open(input_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        all_rows.append(header)  # Keep the header
        
        for row in reader:
            game = row[0]
            minute = row[4]
            current_players = get_home_players(row)
            
            # If this is a new game, reset previous players
            if game != current_game:
                current_game = game
                previous_players = current_players
                all_rows.append(row)  # Keep the first row of each game
                continue
            
            # If players have changed, find the substitution indices
            if current_players != previous_players:
                # Find which players were substituted in
                players_in = current_players - previous_players
                # Find which players were substituted out
                players_out = previous_players - current_players
                
                if players_in and players_out:
                    # Find the indices where substitutions occurred
                    sub_indices = []
                    for player in players_in:
                        for i in range(5, 10):
                            if row[i] == player:
                                sub_indices.append(i-5)  # Convert to 0-based index
                    
                    sub_info = f"Minute {minute}:"
                    sub_info += f"\n  Players out: {', '.join(players_out)}"
                    sub_info += f"\n  Players in: {', '.join(players_in)}"
                    sub_info += f"\n  Substitution indices: {sub_indices}"
                    
                    # Get the modified row with swapped players
                    modified_row = print_swapped_players(row, sub_indices)
                    all_rows.append(modified_row)
                    game_subs[game].append(sub_info)
                else:
                    all_rows.append(row)  # Keep rows without substitutions
                
                previous_players = current_players
            else:
                all_rows.append(row)  # Keep rows without changes
    
    # Write all rows to a new CSV file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)
    
    print(f"\nProcessed {input_file.name}")
    print(f"Output saved to: {output_file.name}")
    print("---")

def main():
    # Process all files in the training datasets directory
    train_dir = Path("Project_Datasets/training_datasets_v2")
    for file_path in train_dir.glob("train_filtered_shuffled_by_game_matchups_*.csv"):
        print(f"\nProcessing {file_path.name}")
        process_file(file_path)

if __name__ == "__main__":
    main() 