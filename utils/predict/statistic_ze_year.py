from time import sleep
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import os
import sys


def get_soup(url):
    """
    Get the soup from the url
    :param url: url to get the soup from
    :return: soup
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


def get_data_from_soup(soup, seasons, leagues, baseUrl):
    """
    Get the data from the soup
    :param soup: soup to get the data from
    :return: data
    """
    data = {}

    for league in tqdm(range(len(leagues))):
        for season in range(len(seasons)):
            url = baseUrl + '/' + leagues[league] + '/' + seasons[season]  # get url for each league and season
            soup = get_soup(url)  # get the soup from the url
            script = soup.find_all('script')[2].string  # get the script tags
            stringsInJson = script.split("('")[1].split("')")  # get the string in the script tags
            jsonData = stringsInJson[0].encode('utf-8').decode('unicode_escape')
            data[leagues[league] + ' ' + seasons[season]] = json.loads(jsonData)  # get the json data from the string

    return data


def extract_stats_from_data(data):
    """
    Extract the stats from the data including opponent stats for each match
    :param data: data to extract the stats from
    :return: stats dictionary with home and away team stats for each match
    """
    stats = {}
    
    for season_key in data.keys():
        season_data = data[season_key]
        
        # Collect all matches first to be able to identify opponents
        match_pairs = {}
        for team_id, team_data in season_data.items():
            for match in team_data['history']:
                date = match['date']
                is_home = match['h_a'] == 'h'
                # Use date as temporary key to group matches
                if date not in match_pairs:
                    match_pairs[date] = []
                match_pairs[date].append({
                    'team_id': team_id,
                    'match': match
                })

        # Process matches and create records with proper IDs
        for team_id, team_data in season_data.items():
            team_name = team_data['title']
            team_matches = []
            
            for match in team_data['history']:
                date = match['date']
                is_home = match['h_a'] == 'h'
                
                # Find opponent ID
                opponent_id = None
                if date in match_pairs:
                    for pair in match_pairs[date]:
                        if pair['team_id'] != team_id:
                            # Verify this is really the opponent by checking stats
                            pair_match = pair['match']
                            if (match['h_a'] != pair_match['h_a'] and  # One home, one away
                                abs(float(match['xG']) - float(pair_match['xGA'])) < 0.01 and  # xG matches
                                abs(float(match['xGA']) - float(pair_match['xG'])) < 0.01 and  # xGA matches
                                match['scored'] == pair_match['missed'] and  # Goals match
                                match['missed'] == pair_match['scored']):  # Goals match
                                opponent_id = pair['team_id']
                                break

                # Create a consistent match ID regardless of which team we're processing
                match_teams = sorted([team_id, opponent_id if opponent_id else 'unknown'])
                match_id = f"{season_key}_{match_teams[0]}_{match_teams[1]}_{match['date']}"
                
                # Create a new match record with both team stats
                combined_match = {                    
                    'match_id': match_id,
                    'date': match['date'],
                    'is_home': match['h_a'] == 'h',
                    'team_id': team_id,
                    'team_name': team_name,
                    'result': match['result'],
                    'opponent_id': 'Unknown',  # We'll need to update this with actual opponent data
                    'opponent_name': 'Unknown',  # We'll need to update this with actual opponent data
                    
                    # Own team stats
                    'team_xG': match['xG'],
                    'team_shots': match['scored'],
                    'team_goals': match['scored'],
                    'team_ppda_att': match['ppda']['att'],
                    'team_ppda_def': match['ppda']['def'],
                    'team_deep': match['deep'],
                    'team_xpts': match['xpts'],
                    'team_result': match['result'],
                    'team_npxG': match['npxG'],
                    
                    # Opponent stats 
                    'opponent_xG': match['xGA'],
                    'opponent_shots': match['missed'],
                    'opponent_goals': match['missed'],
                    'opponent_ppda_att': match['ppda_allowed']['att'],
                    'opponent_ppda_def': match['ppda_allowed']['def'],
                    'opponent_deep': match['deep_allowed'],
                    'opponent_npxG': match['npxGA'],
                }
                
                team_matches.append(combined_match)
            
            season = season_key.split(' ')[1]  # Extract season from the key
            stats[season_key.split(' ')[0] + ' ' + season + ' ' + team_name] = team_matches
    
    return stats


def save_data(stats):
    """
    Save the stats to a csv file
    :param stats: stats to save
    """
    for team in stats.keys():
        dir = team.split(' ')[0]
        # split the team name from the index 2 to the end of the string
        teamName = team.split(' ')[2:]
        season = team.split(' ')[1]

        # Define the base directory - save in Data/ instead of Match of ze day/
        base_dir = './Data/' + dir

        # Check if the directory exists. If not, create it.
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(stats[team])
        output_file = f"{base_dir}/{season} {' '.join(teamName)}.csv"
        df.to_csv(output_file, index=False)


def find_matching_matches(data):
    """
    Post-process the data to find matching matches and update opponent information
    :param data: Dictionary containing all match data
    :return: Updated match data with opponent information
    """
    matched_games = {}
    
    # First, create a dictionary of all matches by date for each league/season
    for season_key, season_data in data.items():
        matches_by_date = {}
        
        for team_id, team_data in season_data.items():
            team_name = team_data['title']
            for match in team_data['history']:
                match_date = match['date']
                if match_date not in matches_by_date:
                    matches_by_date[match_date] = []
                
                matches_by_date[match_date].append({
                    'team_id': team_id,
                    'team_name': team_name,
                    'match': match
                })
        
        # For each date, find matching pairs of matches
        for date, matches in matches_by_date.items():
            for i in range(len(matches)):
                for j in range(i + 1, len(matches)):
                    match1 = matches[i]['match']
                    match2 = matches[j]['match']
                    
                    # Compare multiple parameters to ensure it's the same match
                    is_same_match = (
                        match1['h_a'] != match2['h_a']  # One home, one away
                        and abs(float(match1['xG']) - float(match2['xGA'])) < 0.01  # xG matches
                        and abs(float(match1['xGA']) - float(match2['xG'])) < 0.01  # xGA matches
                        and match1['scored'] == match2['missed']  # Goals scored/conceded match
                        and match1['missed'] == match2['scored']  # Goals scored/conceded match
                        and match1['deep'] == match2['deep_allowed']  # Deep passes match
                        and match1['deep_allowed'] == match2['deep']  # Deep passes allowed match
                    )
                    
                    if is_same_match:
                        match_key = f"{season_key}_{date}_{matches[i]['team_id']}_{matches[j]['team_id']}"
                        
                        # Store the match information
                        if match1['h_a'] == 'h':
                            home_team, away_team = matches[i], matches[j]
                        else:
                            home_team, away_team = matches[j], matches[i]
                            
                        matched_games[match_key] = {
                            'date': date,
                            'home_team_id': home_team['team_id'],
                            'home_team_name': home_team['team_name'],
                            'away_team_id': away_team['team_id'],
                            'away_team_name': away_team['team_name']
                        }
    
    return matched_games


def update_stats_with_opponents(stats, data):
    """
    Update the stats dictionary with opponent information
    """
    # First get all matched games
    matched_games = find_matching_matches(data)
    
    # Update each team's matches with opponent information
    for team_key in stats.keys():
        for match in stats[team_key]:
            date = match['date']
            team_id = match['team_id']
            
            # Search for this match in matched_games
            for match_key, game_info in matched_games.items():
                if game_info['date'] == date:
                    if game_info['home_team_id'] == team_id:
                        match['opponent_id'] = game_info['away_team_id']
                        match['opponent_name'] = game_info['away_team_name']
                        break
                    elif game_info['away_team_id'] == team_id:
                        match['opponent_id'] = game_info['home_team_id']
                        match['opponent_name'] = game_info['home_team_name']
                        break
    
    return stats


if __name__ == '__main__':
    baseUrl = 'https://understat.com/league'
    leagues = ['La_liga', 'Bundesliga', 'EPL', 'Serie_A', 'Ligue_1']
    seasons = ['2025']

    soup = get_soup(baseUrl)
    data = get_data_from_soup(soup, seasons, leagues, baseUrl)
    stats = extract_stats_from_data(data)
    stats = update_stats_with_opponents(stats, data)  # Update with opponent information
    save_data(stats)
