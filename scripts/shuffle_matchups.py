import pandas as pd
import numpy as np
import glob
import os

# Get all encoded matchup files except combined_shuffled_matchups.csv
data_dir = 'Project_Datasets/filtered_data'
matchup_files = glob.glob(os.path.join(data_dir, 'encoded_matchups-*.csv'))

# Read and combine all files
dfs = []
for file in matchup_files:
    df = pd.read_csv(file)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Group by game code
game_groups = combined_df.groupby('game')

# Get unique game codes and shuffle them
game_codes = list(game_groups.groups.keys())
np.random.shuffle(game_codes)

# Create new DataFrame with shuffled games
shuffled_dfs = []
for game_code in game_codes:
    shuffled_dfs.append(game_groups.get_group(game_code))

shuffled_df = pd.concat(shuffled_dfs, ignore_index=True)

# Save to new file
output_file = os.path.join(data_dir, 'shuffled_by_game_matchups.csv')
shuffled_df.to_csv(output_file, index=False)
print(f"Saved shuffled matchups to {output_file}") 