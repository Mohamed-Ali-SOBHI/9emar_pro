import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_all_data():
    """
    Load all CSV files from the Data directory and its subdirectories
    """
    all_data = []
    leagues = ['Bundesliga', 'EPL', 'La_liga', 'Ligue_1', 'Serie_A']
    
    print("Loading data from leagues...")
    for league in tqdm(leagues, desc="Loading leagues"):
        path = os.path.join('Data', league, '*.csv')
        files = glob.glob(path)
        
        if not files:
            print(f"Warning: No CSV files found for {league} in {path}")
            continue
            
        for file in files:
            try:
                df = pd.read_csv(file)
                df['league'] = league
                all_data.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if not all_data:
        print("No data loaded from 'Data' directory. Trying to load from league-named subdirectories in current path...")
        all_data_fallback = []
        for league in tqdm(leagues, desc="Loading leagues (fallback)"):
            path = os.path.join(league, '*.csv')
            files = glob.glob(path)
            if not files:
                print(f"Warning: No CSV files found for {league} in {path} (fallback)")
                continue
            for file in files:
                try:
                    df = pd.read_csv(file)
                    df['league'] = league
                    all_data_fallback.append(df)
                except Exception as e:
                    print(f"Error loading {file} (fallback): {e}")
        if not all_data_fallback:
            raise ValueError("No data loaded. Check your Data directory structure or ensure league subdirectories with CSVs exist.")
        return pd.concat(all_data_fallback, ignore_index=True)

    return pd.concat(all_data, ignore_index=True)

def create_standardized_match_id(row):
    """
    Create a standardized match ID where teams are always in alphabetical order
    """
    try:
        # Ensure team_id and opponent_id are strings for consistent sorting and joining
        team_id_str = str(row['team_id'])
        opponent_id_str = str(row['opponent_id'])
        
        parts = str(row['match_id']).split('_') # Ensure match_id is also treated as string
        season_league = '_'.join(parts[:-3])
        date_part = parts[-1] # Renamed from 'date' to avoid conflict with actual date column
        
        teams = sorted([team_id_str, opponent_id_str])
        return f"{season_league}_{teams[0]}_{teams[1]}_{date_part}"
    except Exception:
        team_id_str = str(row.get('team_id', 'unknown_team'))
        opponent_id_str = str(row.get('opponent_id', 'unknown_opponent'))
        teams = sorted([team_id_str, opponent_id_str])
        return f"{teams[0]}_{teams[1]}_{row.get('date', 'unknown_date_format')}"


def remove_duplicate_matches(df):
    """
    Remove duplicate matches keeping only one record per match.
    Always keeps the home team record to ensure result consistency.
    """
    print("Removing duplicate matches...")
    original_count = len(df)
    
    # Create standardized match ID
    # Ensure 'match_id', 'team_id', 'opponent_id' exist before applying
    if not all(col in df.columns for col in ['match_id', 'team_id', 'opponent_id']):
        print("Warning: 'match_id', 'team_id', or 'opponent_id' not found. Skipping standardized_match_id creation for duplicate removal.")
        if 'match_id' in df.columns:
             df = df.drop_duplicates('match_id', keep='last') # Simplest fallback
        else:
            # If no match_id, duplicate removal becomes very tricky and data-dependent
            print("Warning: Cannot reliably remove duplicates without a 'match_id' or equivalent. Proceeding with current data.")
            return df
    else:
        df['standardized_match_id'] = df.apply(create_standardized_match_id, axis=1)
        # Sort and remove duplicates - Always keep home team
        # Ensure 'is_home' exists, or handle its absence
        if 'is_home' in df.columns:
            # Sort by match_id first, then by is_home (descending to put True first)
            df = df.sort_values(['standardized_match_id', 'is_home'], na_position='last', ascending=[True, False])
            # Keep first (which will be the home team due to descending sort on is_home)
            df = df.drop_duplicates('standardized_match_id', keep='first')
        else:
            df = df.sort_values(['standardized_match_id'], na_position='last') # Sort without 'is_home'
            df = df.drop_duplicates('standardized_match_id', keep='first')
        df = df.drop('standardized_match_id', axis=1, errors='ignore') # errors='ignore' in case it wasn't created
    
    print(f"Removed {original_count - len(df)} duplicate matches")
    return df

def calculate_rolling_stats_optimized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized version of rolling statistics calculation using vectorized operations
    """
    required_columns = ['team_id', 'date', 'team_xG', 'team_deep', 'opponent_xG',
                        'opponent_deep', 'team_ppda_att', 'team_ppda_def', 'opponent_id'] # Added opponent_id
    
    # Validate input
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for rolling stats: {missing_cols}")
    
    df = df.copy()
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['team_id', 'date'])
    
    # Calculate PPDA once
    df['team_ppda'] = df['team_ppda_att'] / df['team_ppda_def'].replace(0, np.nan) # Use np.nan to handle division by zero, then fill
    df['team_ppda'] = df['team_ppda'].fillna(0) # Fill NaNs that resulted from division by zero or original NaNs

    # Define statistics to calculate
    team_stats_sources = { # Renamed to avoid conflict
        'team_xG': 'team_xG',
        'team_deep': 'team_deep', 
        'team_xG_against': 'opponent_xG',
        'team_deep_against': 'opponent_deep',
        'team_ppda': 'team_ppda'
    }
    
    windows = [5, 3, 1]
    
    print("Calculating rolling statistics for teams...")
    # Calculate rolling means for each team (shifted by 1 to exclude current match)
    for window in tqdm(windows, desc="Team rolling windows"):
        grouped = df.groupby('team_id')
        for new_col_suffix, source_col in team_stats_sources.items():
            # Shift before rolling to ensure stats are from *previous* matches
            df[f'{new_col_suffix}_last_{window}'] = grouped[source_col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'{new_col_suffix}_last_{window}'] = df[f'{new_col_suffix}_last_{window}'].fillna(0)


    print("Mapping opponent rolling statistics...")
    # Create a temporary key for merging: match_id and opponent_id
    # This assumes each row has a unique match identifier and the opponent for that match.
    # We need to get the opponent's stats *before* the current match.

    # First, create a DataFrame that has the team's rolling stats for each match
    team_rolling_stats_for_mapping = df[['date', 'team_id'] + [f'{stat_name}_last_{w}' for stat_name in team_stats_sources.keys() for w in windows]].copy()
    team_rolling_stats_for_mapping.rename(columns={'team_id': 'opponent_id_map'}, inplace=True)


    # Merge opponent stats. For each match, find the opponent's stats from their previous matches.
    # This requires careful handling of dates.
    # An alternative, more robust approach if direct mapping is complex:
    # Iterate through each match, find the opponent's previous matches, and calculate their stats.
    # However, for optimization, we try to vectorize.

    # Let's create a multi-index lookup for team stats by date
    df_sorted_for_opponent_lookup = df.sort_values(['team_id', 'date'])
    
    for window in tqdm(windows, desc="Opponent rolling windows"):
        for team_stat_suffix, source_col_for_team in team_stats_sources.items():
            # The corresponding opponent stat we want to create
            opponent_stat_name = f'opponent_{team_stat_suffix.replace("team_", "")}_last_{window}'
            # The source stat from the team's perspective (which becomes opponent's stat)
            # e.g. if team_stat_suffix is 'team_xG', source_col_for_team is 'team_xG'
            # We need the opponent's 'team_xG_last_window'
            
            # Create a map of team's stat for each of their games
            # Key: (team_id, date of game), Value: stat value *before* that game
            stat_to_map_col_name = f'{team_stat_suffix}_last_{window}' # This is already calculated and shifted for the team

            # Create a temporary column in df to store opponent's stat
            df[opponent_stat_name] = np.nan

            # Iterate over each row to find the opponent's stat from *their* history
            # This is less vectorized but more accurate for "stats before this specific match"
            # For a more vectorized approach, one might use pd.merge_asof, but it needs careful key setup.
            
            # Simplified mapping: Use the opponent's calculated rolling stats.
            # We need to map df['opponent_id'] to the 'team_id' in the stats calculation.
            # And ensure the stats are from before the current match date.

            # Create a temporary df with team_id, date, and the specific stat
            temp_stat_df = df[['team_id', 'date', stat_to_map_col_name]].copy()
            temp_stat_df.rename(columns={'team_id': 'opponent_id_lookup', stat_to_map_col_name: 'stat_value'}, inplace=True)
            
            # Merge based on opponent_id and date. This is tricky.
            # A common way is to merge and then filter.
            # For each match in df (left side):
            #   left_on=['opponent_id', 'date']
            # For each stat record in temp_stat_df (right side):
            #   right_on=['opponent_id_lookup', 'date']
            # This would give the opponent's stat for games played on the *same date*.
            # We need stats *before* the current game.

            # Let's use a slightly different mapping strategy:
            # For each opponent_id, get their timeline of stats.
            # Then for each match, lookup the opponent's stat just before that match_date.
            
            # Create a dictionary for faster lookup: opponent_id -> sorted_list_of_tuples (date, stat_value)
            opponent_stats_history = {}
            if stat_to_map_col_name in df.columns:
                for team_id, group in df.groupby('team_id'):
                    opponent_stats_history[team_id] = sorted(
                        [(pd.to_datetime(r['date']), r[stat_to_map_col_name]) for _, r in group.iterrows()],
                        key=lambda x: x[0]
                    )
            
            opponent_stat_values = []
            for _, row in df.iterrows():
                current_opponent_id = row['opponent_id']
                current_match_date = pd.to_datetime(row['date'])
                
                stat_found = 0.0 # Default if no prior stat
                if current_opponent_id in opponent_stats_history:
                    # Find the stat for the opponent just before the current_match_date
                    # Their history is already shifted, so the stat at a date is for *that* game, based on *its* priors
                    # We need the stat that was valid for the opponent *entering* this match.
                    
                    # Find the last record in opponent's history strictly before current_match_date
                    last_valid_stat = 0.0
                    found_any_prior = False
                    for stat_date, stat_val in opponent_stats_history[current_opponent_id]:
                        if stat_date < current_match_date:
                            last_valid_stat = stat_val
                            found_any_prior = True
                        else:
                            break # History is sorted by date
                    if found_any_prior:
                         stat_found = last_valid_stat
                opponent_stat_values.append(stat_found)
            df[opponent_stat_name] = opponent_stat_values
            df[opponent_stat_name] = df[opponent_stat_name].fillna(0)

    return df


def calculate_head_to_head_stats_optimized(df):
    """
    Optimized head-to-head statistics calculation.
    Stats are calculated for the team in 'team_id' against 'opponent_id' based on their past encounters.
    """
    print("Calculating head-to-head statistics (optimized)...")
    
    if 'date' not in df.columns:
        raise ValueError("Missing 'date' column for H2H stats.")
    df['date'] = pd.to_datetime(df['date'])
    
    # Columns to check for existence before using
    # These are potential source columns from the input data for H2H calculations
    source_stat_cols = ['team_goals', 'opponent_goals', 'team_xG', 'opponent_xG', 'team_dominance_score', 'result']
    
    # Initialize all H2H columns that we intend to create
    h2h_cols_to_create = {
        'h2h_matches_played_3': 0,
        'h2h_goals_scored_avg_3': 0.0, 'h2h_goals_conceded_avg_3': 0.0,
        'h2h_xG_avg_3': 0.0, 'h2h_xG_against_avg_3': 0.0,
        'h2h_dominance_score_avg_3': 0.0, # Will only be calculated if 'team_dominance_score' exists
        'h2h_wins_pct_3': 0.0, 'h2h_draws_pct_3': 0.0, 'h2h_losses_pct_3': 0.0, # Only if 'result' exists
        'h2h_goals_scored_last': 0, 'h2h_goals_conceded_last': 0,
        'h2h_xG_last': 0.0, 'h2h_xG_against_last': 0.0,
        'h2h_dominance_score_last': 0.0, # Only if 'team_dominance_score' exists - MODIFIED: User wants this removed
        'h2h_last_result': 'N/A' # Only if 'result' exists
    }
    
    for col, default_val in h2h_cols_to_create.items():
        df[col] = default_val
        if isinstance(default_val, str):
            df[col] = df[col].astype('object') # Ensure string type for 'N/A'
            
    # Create a unique identifier for each match instance (original row) to map results back
    df['original_index'] = df.index

    # Create a DataFrame with records for both perspectives of a match if not already present
    # This function assumes df contains one row per team per match.
    # For H2H, we need to look at past matches between team_A and team_B.

    # Sort dataframe by date to process matches chronologically
    df = df.sort_values('date')

    # Store calculated H2H stats in a temporary list of dicts
    h2h_records = []

    # Iterate through each unique match (defined by team_id, opponent_id, and date for current processing)
    # Grouping by a standardized matchup ID (sorted team names) can help process all historical games for a pair.
    
    # Create a standardized matchup string for grouping
    if not ('team_id' in df.columns and 'opponent_id' in df.columns):
        print("Warning: 'team_id' or 'opponent_id' missing. Cannot calculate H2H stats.")
        return df

    df['matchup_pair'] = df.apply(lambda x: tuple(sorted((str(x['team_id']), str(x['opponent_id'])))), axis=1)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing H2H"):
        current_original_idx = row['original_index']
        current_team_id = str(row['team_id'])
        current_opponent_id = str(row['opponent_id'])
        current_date = row['date']
        current_matchup_pair = row['matchup_pair']

        # Find all historical matches for this specific pair before the current match's date
        # These are matches where (teamA=current_team_id, teamB=current_opponent_id) OR (teamA=current_opponent_id, teamB=current_team_id)
        historical_matches_for_pair = df[
            (df['matchup_pair'] == current_matchup_pair) &
            (df['date'] < current_date)
        ]
        
        # Now, filter these to get matches from the perspective of the current_team_id
        # i.e., where historical_matches_for_pair['team_id'] == current_team_id
        # AND historical_matches_for_pair['opponent_id'] == current_opponent_id
        # This is crucial: H2H stats are for current_team_id *against* current_opponent_id
        
        # We need to consider two scenarios for past matches:
        # 1. current_team_id was 'team_id' and current_opponent_id was 'opponent_id'
        # 2. current_team_id was 'opponent_id' and current_opponent_id was 'team_id' (need to flip stats)

        # Let's simplify: collect all matches involving these two teams, then attribute stats correctly.
        # The provided `df` already has one row per team per match.
        # So, `historical_matches_for_pair` contains all entries related to this pair.
        # We need to select those where `team_id` is `current_team_id` and `opponent_id` is `current_opponent_id`.
        
        # Get previous matches where current_team_id played against current_opponent_id
        # (current_team_id as 'team_id' and current_opponent_id as 'opponent_id')
        prev_matches_perspective = historical_matches_for_pair[
            (historical_matches_for_pair['team_id'].astype(str) == current_team_id) &
            (historical_matches_for_pair['opponent_id'].astype(str) == current_opponent_id)
        ].sort_values('date', ascending=False)

        temp_h2h_stats = {'original_index': current_original_idx} # Start with default values from h2h_cols_to_create

        if not prev_matches_perspective.empty:
            last_3 = prev_matches_perspective.head(3)
            temp_h2h_stats['h2h_matches_played_3'] = len(last_3)

            if 'team_goals' in last_3.columns and 'opponent_goals' in last_3.columns:
                temp_h2h_stats['h2h_goals_scored_avg_3'] = last_3['team_goals'].mean()
                temp_h2h_stats['h2h_goals_conceded_avg_3'] = last_3['opponent_goals'].mean()
            if 'team_xG' in last_3.columns and 'opponent_xG' in last_3.columns:
                temp_h2h_stats['h2h_xG_avg_3'] = last_3['team_xG'].mean()
                temp_h2h_stats['h2h_xG_against_avg_3'] = last_3['opponent_xG'].mean()
            if 'team_dominance_score' in last_3.columns: # This stat is calculated if source exists
                temp_h2h_stats['h2h_dominance_score_avg_3'] = last_3['team_dominance_score'].mean()
            if 'result' in last_3.columns:
                temp_h2h_stats['h2h_wins_pct_3'] = (last_3['result'] == 'w').mean()
                temp_h2h_stats['h2h_draws_pct_3'] = (last_3['result'] == 'd').mean()
                temp_h2h_stats['h2h_losses_pct_3'] = (last_3['result'] == 'l').mean()

            last_match = prev_matches_perspective.iloc[0]
            if 'team_goals' in last_match.index and 'opponent_goals' in last_match.index :
                temp_h2h_stats['h2h_goals_scored_last'] = last_match['team_goals']
                temp_h2h_stats['h2h_goals_conceded_last'] = last_match['opponent_goals']
            if 'team_xG' in last_match.index and 'opponent_xG' in last_match.index:
                temp_h2h_stats['h2h_xG_last'] = last_match['team_xG']
                temp_h2h_stats['h2h_xG_against_last'] = last_match['opponent_xG']
            # 'h2h_dominance_score_last' is intentionally NOT calculated as per user request to remove it.
            if 'result' in last_match.index:
                temp_h2h_stats['h2h_last_result'] = last_match['result']
        
        h2h_records.append(temp_h2h_stats)

    if h2h_records:
        h2h_df = pd.DataFrame(h2h_records)
        h2h_df.set_index('original_index', inplace=True)
        
        # Update the main DataFrame
        for col in h2h_df.columns:
            if col in df.columns: # Ensure column exists in df (it should, as we initialized them)
                 df.update(h2h_df[[col]]) # Use update for safer assignment based on index
            # else: # This case should not happen if initialized correctly
            #    df[col] = h2h_df[col] 
    
    df.drop(columns=['original_index', 'matchup_pair'], inplace=True, errors='ignore')
    return df


def calculate_form_indicators(df):
    """
    Calculate detailed form indicators including streaks and weighted form scores
    """
    print("Calculating form indicators...")
    
    if 'team_id' not in df.columns or 'date' not in df.columns:
        print("Warning: 'team_id' or 'date' missing. Cannot calculate form indicators.")
        return df
        
    df = df.sort_values(['team_id', 'date'])
    
    # Initialize form columns
    form_cols_to_init = {
        'current_streak': 0,
        'streak_type': 'none', 
        'form_score_5': 0.0,
        'form_score_10': 0.0,
        'unbeaten_streak': 0,
        'winless_streak': 0
    }
    
    for col, default_val in form_cols_to_init.items():
        df[col] = default_val
        if isinstance(default_val, str):
            df[col] = df[col].astype('object')

    # Temporary storage for new column data
    new_form_data = {col: [None] * len(df) for col in form_cols_to_init.keys()}
    df_indices = df.index # To map data back correctly

    for team_id_val in tqdm(df['team_id'].unique(), desc="Processing team form"):
        team_mask = (df['team_id'] == team_id_val)
        team_matches_indices = df_indices[team_mask]
        team_matches_df = df.loc[team_matches_indices].sort_values('date') # Use .loc for a copy
        
        if 'result' not in team_matches_df.columns:
            continue # Skip if no 'result' column for this team's matches
            
        # Use arrays for faster assignments within the loop for this team
        # These will hold data for the current team's matches
        current_streak_arr = np.zeros(len(team_matches_df), dtype=int)
        streak_type_arr = np.full(len(team_matches_df), 'none', dtype=object)
        form_score_5_arr = np.zeros(len(team_matches_df), dtype=float)
        form_score_10_arr = np.zeros(len(team_matches_df), dtype=float)
        unbeaten_streak_arr = np.zeros(len(team_matches_df), dtype=int)
        winless_streak_arr = np.zeros(len(team_matches_df), dtype=int)

        # Iterate using integer indices over team_matches_df
        for i in range(len(team_matches_df)):
            current_match_original_idx = team_matches_df.index[i] # Original index in df
            current_date_val = team_matches_df.iloc[i]['date']
            
            prev_matches = team_matches_df[team_matches_df['date'] < current_date_val].sort_values('date', ascending=False)
            
            if prev_matches.empty:
                continue
            
            # Calculate current streak
            streak_val = 0
            s_type = 'none'
            if not prev_matches.empty:
                last_res = prev_matches.iloc[0]['result']
                s_type = last_res
                streak_val = 1
                for k in range(1, len(prev_matches)):
                    if prev_matches.iloc[k]['result'] == last_res:
                        streak_val += 1
                    else:
                        break
            current_streak_arr[i] = streak_val
            streak_type_arr[i] = s_type
            
            # Form scores
            for window in [5, 10]:
                recent_m = prev_matches.head(window)
                if not recent_m.empty:
                    points = recent_m['result'].map({'w': 3, 'd': 1, 'l': 0}).fillna(0)
                    weights = np.array([1.0 / (j + 1) for j in range(len(points))]) # More recent = higher weight (smaller j)
                    weights = weights / weights.sum() if weights.sum() > 0 else weights # Normalize
                    
                    form_s = (points.values * weights).sum() # Use .values for numpy array
                    if window == 5: form_score_5_arr[i] = form_s
                    else: form_score_10_arr[i] = form_s
            
            # Unbeaten and winless streaks
            unbeaten_s = 0
            winless_s = 0
            for _, match_row in prev_matches.iterrows(): # iterrows is slow, but ok for small prev_matches
                if match_row['result'] in ['w', 'd']: unbeaten_s += 1
                else: break
            unbeaten_streak_arr[i] = unbeaten_s

            for _, match_row in prev_matches.iterrows():
                if match_row['result'] in ['d', 'l']: winless_s += 1
                else: break
            winless_streak_arr[i] = winless_s

        # Map results back to the main df structure using original indices
        for i in range(len(team_matches_df)):
            original_idx = team_matches_df.index[i]
            new_form_data['current_streak'][df.index.get_loc(original_idx)] = current_streak_arr[i]
            new_form_data['streak_type'][df.index.get_loc(original_idx)] = streak_type_arr[i]
            new_form_data['form_score_5'][df.index.get_loc(original_idx)] = form_score_5_arr[i]
            new_form_data['form_score_10'][df.index.get_loc(original_idx)] = form_score_10_arr[i]
            new_form_data['unbeaten_streak'][df.index.get_loc(original_idx)] = unbeaten_streak_arr[i]
            new_form_data['winless_streak'][df.index.get_loc(original_idx)] = winless_s

    # Assign collected data to DataFrame
    for col_name, data_list in new_form_data.items():
        # Filter out None placeholders before assigning, or ensure correct length
        # df[col_name] = pd.Series(data_list, index=df.index).fillna(form_cols_to_init[col_name])
        # A more robust way if data_list might have Nones for rows not processed:
        # Create a Series from the list, aligned with df's index
        temp_series = pd.Series(data_list, index=df.index)
        # Fill NaNs that were originally None placeholders with the default value
        df[col_name] = temp_series.fillna(form_cols_to_init[col_name])

    return df


def calculate_season_metrics(df):
    """
    Calculate season performance metrics.
    """
    print("Calculating season performance metrics...")

    if 'year' not in df.columns: # 'year' should be created in remove_post_match_features
        print("Warning: 'year' column missing. Cannot reliably determine season. Skipping season metrics.")
        return df
    if 'team_id' not in df.columns or 'date' not in df.columns or 'league' not in df.columns:
        print("Warning: 'team_id', 'date', or 'league' missing. Cannot calculate season metrics.")
        return df

    df['season'] = df['year'].astype(str) 
    if 'match_id' in df.columns: # If match_id was kept for some reason (it's on removal list)
        try: # Attempt to extract a more specific season if possible
            df['season_extracted'] = df['match_id'].astype(str).str.extract(r'(\d{4})', expand=False)
            df['season'] = df['season_extracted'].fillna(df['season'])
            df.drop(columns=['season_extracted'], inplace=True)
        except Exception:
            pass # Stick with year-based season

    # Initialize season metrics columns
    season_cols_to_init = {
        'season_points': 0, 'season_matches_played': 0, 'season_points_per_game': 0.0,
        'season_wins': 0, 'season_draws': 0, 'season_losses': 0, 'season_win_pct': 0.0,
        'season_form_last_10': 0.0, 'league_position_proxy': 0.0 
    }
    
    for col, default_val in season_cols_to_init.items():
        df[col] = default_val

    # Temporary storage for new column data
    new_season_data = {col: [None] * len(df) for col in season_cols_to_init.keys()}
    df_indices = df.index

    # Group by team, season, and league
    grouped_by_team_season_league = df.groupby(['team_id', 'season', 'league'])

    for (team_id_val, season_val, league_val), group_indices in tqdm(grouped_by_team_season_league.groups.items(), desc="Processing season metrics"):
        group = df.loc[group_indices].sort_values('date') # Use .loc for a copy
        
        # Arrays for current group
        s_points_arr = np.zeros(len(group), dtype=int)
        s_matches_arr = np.zeros(len(group), dtype=int)
        s_ppg_arr = np.zeros(len(group), dtype=float)
        s_wins_arr = np.zeros(len(group), dtype=int)
        s_draws_arr = np.zeros(len(group), dtype=int)
        s_losses_arr = np.zeros(len(group), dtype=int)
        s_win_pct_arr = np.zeros(len(group), dtype=float)
        s_form10_arr = np.zeros(len(group), dtype=float)

        cumulative_points = 0
        cumulative_matches = 0
        cumulative_wins = 0
        cumulative_draws = 0
        cumulative_losses = 0
        
        # Store results of past matches for form_last_10 calculation
        past_results_for_form = []

        for i in range(len(group)):
            current_match_original_idx = group.index[i]
            # The metrics for row 'i' are based on matches *before* it.
            # So, assign the *current cumulative* values to the row, then update cumulatives with this row's result.
            
            s_points_arr[i] = cumulative_points
            s_matches_arr[i] = cumulative_matches
            s_ppg_arr[i] = cumulative_points / cumulative_matches if cumulative_matches > 0 else 0.0
            s_wins_arr[i] = cumulative_wins
            s_draws_arr[i] = cumulative_draws
            s_losses_arr[i] = cumulative_losses
            s_win_pct_arr[i] = cumulative_wins / cumulative_matches if cumulative_matches > 0 else 0.0

            # Season form last 10 (based on matches before current)
            if past_results_for_form:
                last_10_season_results = past_results_for_form[-10:] # Get up to last 10
                if last_10_season_results:
                    points_map = {'w': 3, 'd': 1, 'l': 0}
                    points_last_10 = sum(points_map.get(res, 0) for res in last_10_season_results)
                    possible_points = len(last_10_season_results) * 3
                    s_form10_arr[i] = points_last_10 / possible_points if possible_points > 0 else 0.0
            
            # Now, update cumulatives with the result of match 'i' for the *next* iteration
            if 'result' in group.columns:
                match_result = group.iloc[i]['result']
                points_map = {'w': 3, 'd': 1, 'l': 0}
                cumulative_points += points_map.get(match_result, 0)
                if match_result == 'w': cumulative_wins += 1
                elif match_result == 'd': cumulative_draws += 1
                elif match_result == 'l': cumulative_losses += 1
                past_results_for_form.append(match_result) # Add current match's result for next iteration's form calc

            cumulative_matches += 1

        # Map results back
        for i in range(len(group)):
            original_idx = group.index[i]
            loc_in_main_df = df.index.get_loc(original_idx)
            new_season_data['season_points'][loc_in_main_df] = s_points_arr[i]
            new_season_data['season_matches_played'][loc_in_main_df] = s_matches_arr[i]
            new_season_data['season_points_per_game'][loc_in_main_df] = s_ppg_arr[i]
            new_season_data['season_wins'][loc_in_main_df] = s_wins_arr[i]
            new_season_data['season_draws'][loc_in_main_df] = s_draws_arr[i]
            new_season_data['season_losses'][loc_in_main_df] = s_losses_arr[i]
            new_season_data['season_win_pct'][loc_in_main_df] = s_win_pct_arr[i]
            new_season_data['season_form_last_10'][loc_in_main_df] = s_form10_arr[i]

    # Assign collected data to DataFrame
    for col_name, data_list in new_season_data.items():
        temp_series = pd.Series(data_list, index=df.index)
        df[col_name] = temp_series.fillna(season_cols_to_init[col_name])


    # Calculate league position proxy (relative performance within league-season)
    print("Calculating relative league performance (league_position_proxy)...")
    league_pos_proxy_data = [None] * len(df) # Using df.index for mapping

    for (league_val, season_val), group_indices in tqdm(df.groupby(['league', 'season']).groups.items(), desc="League Position Proxy"):
        league_season_group = df.loc[group_indices]
        if league_season_group.empty or 'season_points_per_game' not in league_season_group.columns:
            continue

        # Using rank(pct=True) on the existing 'season_points_per_game' within the league-season group.
        # This provides a percentile rank for each match's pre-match PPG within its league-season context.
        ranked_ppg = league_season_group['season_points_per_game'].rank(pct=True, method='average').fillna(0.5)
        
        for original_idx, rank_val in ranked_ppg.items():
             loc_in_main_df = df.index.get_loc(original_idx)
             league_pos_proxy_data[loc_in_main_df] = rank_val
             
    df['league_position_proxy'] = pd.Series(league_pos_proxy_data, index=df.index).fillna(0.5)
    
    return df


def add_schedule_and_elo_features(df, base_rating=1500.0, k_factor=20.0, home_advantage=65.0):
    """
    Compute pre-match Elo ratings and rest-day features in a time-aware manner.
    Ratings are tracked separately per league to avoid cross-competition leakage.
    """
    required_columns = {'team_name', 'opponent_name', 'date'}
    if not required_columns.issubset(df.columns):
        print("Skipping Elo/rest features: required columns are missing.")
        return df

    print("Adding Elo strength and rest-day schedule features...")

    working = df[['team_name', 'opponent_name', 'date']].copy()
    working['league'] = df['league'].fillna('GLOBAL') if 'league' in df.columns else 'GLOBAL'
    working['match_date'] = pd.to_datetime(df['date'], errors='coerce')
    working['match_result'] = df['result'] if 'result' in df.columns else pd.Series(np.nan, index=df.index)
    working['orig_index'] = working.index
    working = working.sort_values(['league', 'match_date', 'orig_index']).reset_index(drop=True)

    ratings = defaultdict(lambda: float(base_rating))
    last_played = {}

    team_elo_values = []
    opponent_elo_values = []
    team_rest_values = []
    opponent_rest_values = []
    elo_probability_values = []

    for row in working.itertuples():
        league = row.league if isinstance(row.league, str) and row.league else 'GLOBAL'
        home_team = row.team_name if isinstance(row.team_name, str) else None
        away_team = row.opponent_name if isinstance(row.opponent_name, str) else None
        match_date = row.match_date

        if not home_team or not away_team:
            team_elo_values.append(np.nan)
            opponent_elo_values.append(np.nan)
            team_rest_values.append(np.nan)
            opponent_rest_values.append(np.nan)
            elo_probability_values.append(np.nan)
            continue

        home_key = (league, home_team)
        away_key = (league, away_team)

        home_rating = ratings[home_key]
        away_rating = ratings[away_key]

        team_elo_values.append(home_rating)
        opponent_elo_values.append(away_rating)

        if pd.notna(match_date):
            last_home = last_played.get(home_key)
            last_away = last_played.get(away_key)
            team_rest_values.append((match_date - last_home).days if last_home is not None else np.nan)
            opponent_rest_values.append((match_date - last_away).days if last_away is not None else np.nan)
            expected_home = 1.0 / (1.0 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
            last_played[home_key] = match_date
            last_played[away_key] = match_date
        else:
            team_rest_values.append(np.nan)
            opponent_rest_values.append(np.nan)
            expected_home = 0.5

        elo_probability_values.append(expected_home)

        result_val = row.match_result if isinstance(row.match_result, str) else None
        if result_val in ('w', 'd', 'l') and pd.notna(match_date):
            score_home = {'w': 1.0, 'd': 0.5, 'l': 0.0}[result_val]
            score_away = 1.0 - score_home
            expected_away = 1.0 - expected_home
            ratings[home_key] = home_rating + k_factor * (score_home - expected_home)
            ratings[away_key] = away_rating + k_factor * (score_away - expected_away)

    working['team_elo_rating'] = team_elo_values
    working['opponent_elo_rating'] = opponent_elo_values
    working['team_rest_days'] = team_rest_values
    working['opponent_rest_days'] = opponent_rest_values
    working['elo_win_probability'] = elo_probability_values

    for column in ['team_elo_rating', 'opponent_elo_rating', 'team_rest_days',
                   'opponent_rest_days', 'elo_win_probability']:
        if column not in df.columns:
            df[column] = np.nan
        df.loc[working['orig_index'], column] = working[column].values

    if 'elo_rating_gap' not in df.columns:
        df['elo_rating_gap'] = np.nan
    df['elo_rating_gap'] = df['team_elo_rating'] - df['opponent_elo_rating']

    if 'rest_days_diff' not in df.columns:
        df['rest_days_diff'] = np.nan
    df['rest_days_diff'] = df['team_rest_days'] - df['opponent_rest_days']

    if 'rest_days_ratio' not in df.columns:
        df['rest_days_ratio'] = np.nan
    df['rest_days_ratio'] = df['team_rest_days'] / (df['opponent_rest_days'] + 1.0)

    return df


def remove_post_match_features(df):
    """
    Remove features that would only be available after the match and keep/calculate relevant pre-match features
    """
    if all(col in df.columns for col in ['team_id', 'date', 'team_xG', 'opponent_id']): 
        df = calculate_rolling_stats_optimized(df)
    else:
        print("Skipping rolling stats calculation due to missing core columns.")

    if 'date' in df.columns:
        df['date_col_for_temporal'] = pd.to_datetime(df['date'])
        df['month'] = df['date_col_for_temporal'].dt.month
        df['year'] = df['date_col_for_temporal'].dt.year
        df.drop(columns=['date_col_for_temporal'], inplace=True) 
    else:
        if 'year' not in df.columns: df['year'] = 0
        if 'month' not in df.columns: df['month'] = 0

    base_columns = []
    for col in ['team_id', 'opponent_id', 'date']:
        if col in df.columns:
            base_columns.append(col)
    
    base_columns.extend(['month', 'year']) 
    
    if 'league' in df.columns: 
        base_columns.append('league')
    
    if 'team_name' in df.columns:
        base_columns.append('team_name')
    if 'opponent_name' in df.columns:
        base_columns.append('opponent_name')
    
    base_columns = list(dict.fromkeys(base_columns))
    
    rolling_columns = []
    potential_rolling_stats_cols = [f'{prefix}_{stat}_last_{window}'
                                     for window in [5, 3, 1]
                                     for prefix in ['team', 'opponent']
                                     for stat in ['xG', 'deep', 'xG_against', 'deep_against', 'ppda']]
    for col in potential_rolling_stats_cols:
        if col in df.columns:
            rolling_columns.append(col)
            
    h2h_columns_potential = [
        'h2h_matches_played_3', 'h2h_goals_scored_avg_3', 'h2h_goals_conceded_avg_3',
        'h2h_xG_avg_3', 'h2h_xG_against_avg_3', 'h2h_dominance_score_avg_3',
        'h2h_wins_pct_3', 'h2h_draws_pct_3', 'h2h_losses_pct_3',
        'h2h_goals_scored_last', 'h2h_goals_conceded_last', 'h2h_xG_last',
        'h2h_xG_against_last', 'h2h_last_result' 
    ]
    h2h_columns = [col for col in h2h_columns_potential if col in df.columns]

    form_columns_potential = [
        'current_streak', 'streak_type', 'form_score_5', 'form_score_10',
        'unbeaten_streak', 'winless_streak'
    ]
    form_columns = [col for col in form_columns_potential if col in df.columns]
    
    season_columns_potential = [
        'season_points', 'season_matches_played', 'season_points_per_game',
        'season_wins', 'season_draws', 'season_losses', 'season_win_pct',
        'season_form_last_10', 'league_position_proxy'
    ]
    season_columns = [col for col in season_columns_potential if col in df.columns]

    engineered_columns_potential = [
        'team_form_xG', 'opponent_form_xG', 'team_defensive_form', 'opponent_defensive_form',
        'h2h_dominance', 'xG_strength_diff', 'defensive_strength_diff',
        'form_diff_5', 'form_momentum', 'ppg_vs_league_avg',
        'opponent_form_score_5', 'opponent_form_score_10', 'opponent_current_streak',
        'opponent_season_points_per_game',
        'team_elo_rating', 'opponent_elo_rating', 'elo_rating_gap', 'elo_win_probability',
        'team_rest_days', 'opponent_rest_days', 'rest_days_diff', 'rest_days_ratio'
    ]
    engineered_columns = [col for col in engineered_columns_potential if col in df.columns]

    target_column = ['result'] if 'result' in df.columns else []
    
    columns_to_keep = base_columns + rolling_columns + h2h_columns + form_columns + season_columns + engineered_columns + target_column
    columns_to_keep = list(dict.fromkeys(columns_to_keep))

    available_columns = [col for col in columns_to_keep if col in df.columns]
    
    print(f"Keeping {len(available_columns)} features out of {len(df.columns)} current columns after processing")
    
    user_requested_drops = [
        'match_id', 'day_of_week', 'is_weekend', 
        'h2h_dominance_score_last', 'goals_trend_5', 'goals_conceded_trend_5',
        'season_goal_difference', 'season_goals_scored', 'season_goals_conceded', 
        'opponent_season_goal_difference', 'opponent_league_position_proxy'
    ]
    
    final_columns_to_select = [col for col in available_columns if col not in user_requested_drops]
    final_columns_to_select = [col for col in final_columns_to_select if col in df.columns]

    print(f"Final selection has {len(final_columns_to_select)} features.")
    return df[final_columns_to_select]


def add_feature_engineering(df):
    """
    Add comprehensive engineered features including form indicators and season metrics
    """
    print("Adding engineered features (calling form and season calcs)...")
    
    # Calculate form indicators (if not already done or if columns are missing)
    # Check if key form columns exist, if not, calculate them.
    if not all(col in df.columns for col in ['form_score_5', 'current_streak']):
        df = calculate_form_indicators(df)
    
    # Calculate season performance metrics (if not already done or if columns are missing)
    if not all(col in df.columns for col in ['season_points_per_game', 'league_position_proxy']):
        df = calculate_season_metrics(df)
    
    print("Calculating derived engineered features...")
    # Team form indicators (difference between recent and longer-term performance)
    if 'team_xG_last_3' in df.columns and 'team_xG_last_5' in df.columns:
        df['team_form_xG'] = df['team_xG_last_3'] - df['team_xG_last_5']
    if 'opponent_xG_last_3' in df.columns and 'opponent_xG_last_5' in df.columns: # Assuming these are calculated in rolling_stats
        df['opponent_form_xG'] = df['opponent_xG_last_3'] - df['opponent_xG_last_5']
    
    # Defensive solidity indicators
    if 'team_xG_against_last_3' in df.columns and 'team_xG_against_last_5' in df.columns:
        df['team_defensive_form'] = df['team_xG_against_last_5'] - df['team_xG_against_last_3'] # Higher is worse (more xG conceded recently)
    if 'opponent_xG_against_last_3' in df.columns and 'opponent_xG_against_last_5' in df.columns:
        df['opponent_defensive_form'] = df['opponent_xG_against_last_5'] - df['opponent_xG_against_last_3']
    
    # Head-to-head dominance
    if 'h2h_wins_pct_3' in df.columns and 'h2h_losses_pct_3' in df.columns:
        df['h2h_dominance'] = df['h2h_wins_pct_3'] - df['h2h_losses_pct_3']
    
    # Strength difference indicators
    if 'team_xG_last_5' in df.columns and 'opponent_xG_last_5' in df.columns:
        df['xG_strength_diff'] = df['team_xG_last_5'] - df['opponent_xG_last_5']
    if 'team_xG_against_last_5' in df.columns and 'opponent_xG_against_last_5' in df.columns: # Corrected logic for defensive strength
        df['defensive_strength_diff'] = df['team_xG_against_last_5'] - df['opponent_xG_against_last_5'] # Team's xGA vs Opponent's xGA. Lower is better for team.
    
    # Advanced form comparisons
    if 'form_score_5' in df.columns and 'league' in df.columns and 'season' in df.columns:
        # Ensure groupby keys exist and are not all NaN
        if df[['league', 'season']].notna().all(axis=1).any():
             df['form_diff_5'] = df['form_score_5'] - df.groupby(['league', 'season'])['form_score_5'].transform('mean')
        else:
            df['form_diff_5'] = 0.0
    if 'form_score_5' in df.columns and 'form_score_10' in df.columns:
        df['form_momentum'] = df['form_score_5'] - df['form_score_10']
    
    # Season performance differences
    if 'season_points_per_game' in df.columns and 'league' in df.columns and 'season' in df.columns:
        if df[['league', 'season']].notna().all(axis=1).any():
            df['ppg_vs_league_avg'] = df['season_points_per_game'] - df.groupby(['league', 'season'])['season_points_per_game'].transform('mean')
        else:
            df['ppg_vs_league_avg'] = 0.0

    # Create opponent form features
    print("Adding opponent specific form/season features...")
    opponent_source_features_map = {
        'opponent_form_score_5': 'form_score_5',
        'opponent_form_score_10': 'form_score_10',
        'opponent_current_streak': 'current_streak',
        'opponent_season_points_per_game': 'season_points_per_game',
    }
    
    # Check if source columns for opponent mapping exist
    source_cols_for_opp_map = [scol for scol in opponent_source_features_map.values() if scol in df.columns]
    
    if source_cols_for_opp_map and 'opponent_id' in df.columns and 'team_id' in df.columns and 'date' in df.columns:
        # For each opponent_col and its source_col:
        for opponent_col, source_col_name in opponent_source_features_map.items():
            if source_col_name in df.columns:
                # Create a map: team_id -> list of (date, stat_value)
                team_stat_history_map = {}
                for team_id_val, group in df.groupby('team_id'):
                    team_stat_history_map[team_id_val] = sorted(
                        [(pd.to_datetime(r['date']), r[source_col_name]) for _, r in group.iterrows() if pd.notna(r[source_col_name])],
                        key=lambda x:x[0]
                    )

                mapped_opponent_stats = []
                for _, row in df.iterrows():
                    opp_id = row['opponent_id']
                    match_d = pd.to_datetime(row['date'])
                    stat_val_found = 0.0 # Default
                    if opp_id in team_stat_history_map:
                        # Find the latest stat for opp_id strictly before match_d
                        last_known_stat = 0.0
                        found_prior = False
                        for stat_date, s_val in team_stat_history_map[opp_id]:
                            if stat_date < match_d:
                                last_known_stat = s_val
                                found_prior = True
                            else:
                                break # Sorted by date
                        if found_prior:
                            stat_val_found = last_known_stat
                    mapped_opponent_stats.append(stat_val_found)
                df[opponent_col] = mapped_opponent_stats
                df[opponent_col] = df[opponent_col].fillna(0) # Fill any NaNs from mapping
            else:
                df[opponent_col] = 0.0 # If source column doesn't exist
    else: # If essential columns for mapping are missing
        for opponent_col in opponent_source_features_map.keys():
            df[opponent_col] = 0.0

    return df


def preprocess_data_memory_only(df_input=None):
    """
    Version optimisée du preprocessing qui ne sauvegarde pas de fichier
    Prend un DataFrame en entrée ou charge les données si df_input=None
    Retourne uniquement le DataFrame processé
    """
    try:
        # Load data if not provided
        if df_input is None:
            print("Loading data...")
            df = load_all_data()
            print(f"Loaded {len(df)} total records")
        else:
            df = df_input.copy()
            print(f"Processing provided DataFrame with {len(df)} records")
        
        # Remove duplicates
        df = remove_duplicate_matches(df)

        # Add schedule & Elo features before any aggregation
        df = add_schedule_and_elo_features(df)
        
        # Calculate head-to-head statistics
        if all(col in df.columns for col in ['date', 'team_id', 'opponent_id']):
            print("Calculating head-to-head statistics...")
            df = calculate_head_to_head_stats_optimized(df)
        else:
            print("Skipping H2H stats due to missing required columns.")

        # Add engineered features
        df = add_feature_engineering(df)
        
        # Remove post-match features and select final feature set
        print("Processing features (selecting final set, removing post-match)...")
        df = remove_post_match_features(df)
        
        # Final data quality checks
        print("Performing data quality checks...")
        print(f"Final dataset shape: {df.shape}")
        
        if not df.empty:
            missing_summary = df.isnull().sum()
            if missing_summary.sum() == 0:
                print("No missing values in the final dataset.")
            else:
                print(f"Missing values detected in {(missing_summary > 0).sum()} columns")
        else:
            print("Warning: DataFrame is empty after preprocessing.")
            
        print("In-memory preprocessing completed successfully!")
        return df
        
    except Exception as e:
        print(f"Error during in-memory preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def preprocess_data(add_features=True, remove_duplicates_flag=True):
    """
    Main preprocessing function with additional options
    
    Args:
        add_features (bool): Whether to add engineered features
        remove_duplicates_flag (bool): Whether to remove duplicate matches
    """
    try:
        # Load all data
        print("Loading data...")
        df = load_all_data()
        print(f"Loaded {len(df)} total records")
        
        # Remove duplicates if requested
        if remove_duplicates_flag:
            df = remove_duplicate_matches(df)
        
        # Add schedule & Elo features prior to aggregations
        df = add_schedule_and_elo_features(df)
        
        # Calculate head-to-head statistics (needs date, team_id, opponent_id)
        if all(col in df.columns for col in ['date', 'team_id', 'opponent_id']):
            print("Calculating head-to-head statistics...")
            df = calculate_head_to_head_stats_optimized(df)
        else:
            print("Skipping H2H stats due to missing 'date', 'team_id', or 'opponent_id'.")

        # Add engineered features if requested
        if add_features:
            df = add_feature_engineering(df)
        
        # Remove post-match features and select final feature set
        print("Processing features (selecting final set, removing post-match)...")
        df = remove_post_match_features(df)
        
        # Final data quality checks
        print("Performing data quality checks...")
        print(f"Final dataset shape: {df.shape}")
        
        if not df.empty:
            print(f"Missing values per column (only showing columns with missing values):")
            missing_summary = df.isnull().sum()
            print(missing_summary[missing_summary > 0])
            if missing_summary.sum() == 0:
                print("No missing values in the final dataset.")
        else:
            print("Warning: DataFrame is empty after preprocessing.")
            
        # Save preprocessed data (optional for backward compatibility)
        output_file = "preprocessed_data.csv"
        print(f"Saving preprocessed data to {output_file}...")
        df.to_csv(output_file, index=False)
        print("Preprocessing completed successfully!")
        
        return df
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def analyze_data_quality(df):
    """
    Analyze the quality of the preprocessed data
    """
    print("\n" + "="*50)
    print("DATA QUALITY ANALYSIS")
    print("="*50)

    if df.empty:
        print("DataFrame is empty. No quality analysis to perform.")
        return

    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_cols = missing_pct[missing_pct > 0].sort_values(ascending=False)
    
    if len(missing_cols) > 0:
        print(f"\nColumns with missing values (%):")
        for col, pct_val in missing_cols.items():
            print(f"  {col}: {pct_val:.2f}%")
    else:
        print("\nNo missing values found!")
    
    # Feature distribution
    if 'result' in df.columns:
        print(f"\nTarget variable distribution ('result'):")
        print(df['result'].value_counts(normalize=True))
    
    # Form indicators summary
    if 'form_score_5' in df.columns:
        print(f"\nForm score (5 games) summary:")
        print(f"  Mean: {df['form_score_5'].mean():.3f}, Median: {df['form_score_5'].median():.3f}, StdDev: {df['form_score_5'].std():.3f}")
    if 'season_points_per_game' in df.columns:
         print(f"\nSeason points per game summary:")
         print(f"  Mean: {df['season_points_per_game'].mean():.3f}, Median: {df['season_points_per_game'].median():.3f}, StdDev: {df['season_points_per_game'].std():.3f}")
        
    if 'current_streak' in df.columns and 'streak_type' in df.columns:
        print(f"\nCurrent streaks distribution:")
        try:
            streak_summary = df.groupby('streak_type')['current_streak'].agg(['count', 'mean', 'max'])
            print(streak_summary)
        except Exception as e:
            print(f"  Could not generate streak summary: {e}")
    
    print(f"\nLeague distribution:")
    if 'league' in df.columns:
        print(df['league'].value_counts())
    else:
        print("  'league' column not found.")

if __name__ == "__main__":
    # Run preprocessing
    processed_df = preprocess_data(add_features=True, remove_duplicates_flag=True) 
    
    # Analyze data quality if df is not empty
    if processed_df is not None and not processed_df.empty:
        analyze_data_quality(processed_df)
    else:
        print("Preprocessing returned an empty or None DataFrame. Skipping analysis.")
