import pandas as pd
import os
from collections import defaultdict

def print_unique_team_codes():
    # Get the current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    players_dir = os.path.join(script_dir, 'available_players')
    
    # Set to store all unique team codes
    all_teams = set()
    
    # Dictionary to track which years each team appears in
    team_years = defaultdict(list)
    
    # Process each year's file
    for year in range(2007, 2016):
        file_path = os.path.join(players_dir, f'team_players_{year}.csv')
        df = pd.read_csv(file_path)
        
        # Get unique teams for this year
        teams_this_year = df['Team'].unique()
        
        # Add to overall set
        all_teams.update(teams_this_year)
        
        # Track which years this team appears in
        for team in teams_this_year:
            team_years[team].append(year)
    
    # Sort teams alphabetically
    sorted_teams = sorted(all_teams)
    
    print(f"Total unique team codes: {len(sorted_teams)}")
    print("\nAll unique team codes:")
    print("=====================")
    for team in sorted_teams:
        print(team)
    
    print("\nTeam codes with years they appear in:")
    print("================================")
    for team in sorted_teams:
        years_str = ", ".join(map(str, team_years[team]))
        print(f"{team}: {years_str}")
    
    # Check for teams that don't appear in all years
    all_years = set(range(2007, 2016))
    print("\nTeams that don't appear in all years:")
    print("================================")
    for team, years in team_years.items():
        if set(years) != all_years:
            missing_years = all_years - set(years)
            if missing_years:
                print(f"{team}: Missing in years {', '.join(map(str, sorted(missing_years)))}")

if __name__ == "__main__":
    print_unique_team_codes() 