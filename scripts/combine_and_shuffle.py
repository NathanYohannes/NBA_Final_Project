import pandas as pd
import os
import glob
from sklearn.utils import shuffle

# Define paths relative to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "Project_Datasets", "filtered_data")
output_dir = data_dir  # Output to the same directory

# Get all encoded_matchups CSV files
csv_files = glob.glob(os.path.join(data_dir, "encoded_matchups-*.csv"))
print(f"Found {len(csv_files)} CSV files to combine")

# Initialize an empty list to store dataframes
dfs = []

# Read each CSV file and append to the list
for file in csv_files:
    print(f"Reading {os.path.basename(file)}...")
    df = pd.read_csv(file)
    dfs.append(df)
    print(f"  - Shape: {df.shape}")

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)
print(f"Combined dataframe shape: {combined_df.shape}")

# Shuffle the data
shuffled_df = shuffle(combined_df, random_state=42)
print(f"Shuffled dataframe shape: {shuffled_df.shape}")

# Save the shuffled data to a new CSV file
output_file = os.path.join(output_dir, "combined_shuffled_matchups.csv")
shuffled_df.to_csv(output_file, index=False)
print(f"Saved shuffled data to {output_file}") 