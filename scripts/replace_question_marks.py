import pandas as pd
import json

def replace_question_marks():
    # Load the test data and labels
    test_data = pd.read_csv('Project_Datasets/validation_dataset/NBA_test.csv')
    labels = pd.read_csv('Project_Datasets/validation_dataset/NBA_test_labels.csv')
    
    # Load encoding files
    with open('Project_Datasets/encoded_data/team_encoding.json', 'r') as f:
        team_encoding = json.load(f)
    with open('Project_Datasets/encoded_data/player_encoding.json', 'r') as f:
        player_encoding = json.load(f)
    
    # Get all player columns except home_4
    player_columns = ['home_0', 'home_1', 'home_2', 'home_3',
                     'away_0', 'away_1', 'away_2', 'away_3', 'away_4']
    
    # Step 1: Move question marks to home_4 column
    print("Step 1: Moving question marks to home_4 column")
    for col in player_columns:
        mask = test_data[col] == '?'
        if mask.any():
            # Store the original home_4 value
            original_home_4 = test_data.loc[mask, 'home_4'].copy()
            # Move the value from the current column to home_4
            test_data.loc[mask, 'home_4'] = test_data.loc[mask, col]
            # Put the original home_4 value in the current column
            test_data.loc[mask, col] = original_home_4
            print(f"Moved {mask.sum()} question marks from {col} to home_4")
    
    # Step 2: Replace question marks in home_4 with labels
    print("\nStep 2: Replacing question marks in home_4 with labels")
    mask = test_data['home_4'] == '?'
    if mask.any():
        test_data.loc[mask, 'home_4'] = labels.loc[mask, 'removed_value']
        print(f"Replaced {mask.sum()} question marks in home_4 with labels")
    
    # Step 3: Encode team and player names
    print("\nStep 3: Encoding team and player names")
    
    # Encode team names
    test_data['home_team'] = test_data['home_team'].map(team_encoding['team_to_number'])
    test_data['away_team'] = test_data['away_team'].map(team_encoding['team_to_number'])
    print("Encoded team names")
    
    # Encode player names
    for col in player_columns + ['home_4']:
        test_data[col] = test_data[col].map(player_encoding['player_to_number'])
        print(f"Encoded {col}")
    
    # Save the updated data
    output_path = 'Project_Datasets/validation_dataset/NBA_test_updated.csv'
    test_data.to_csv(output_path, index=False)
    print(f"\nUpdated data saved to: {output_path}")
    
    # Print some statistics
    print("\nData Statistics:")
    print(f"Total rows: {len(test_data)}")
    print(f"Number of unique teams: {len(set(test_data['home_team'].unique()) | set(test_data['away_team'].unique()))}")
    print(f"Number of unique players: {len(set().union(*[set(test_data[col].unique()) for col in player_columns + ['home_4']]))}")
    
    # Verify data types
    print("\nData Types:")
    print(test_data.dtypes)

if __name__ == "__main__":
    replace_question_marks() 