#!/usr/bin/env python3
"""
Enhanced feature engineering pour améliorer les performances du modèle.
Ajoute des features avancées basées sur l'analyse des données football.
Merged from advanced_features.py and enhanced_feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import os
import glob
warnings.filterwarnings('ignore')

def load_season_stats(df):
    """Calcule les statistiques de saison pour chaque équipe"""
    print("[FEATURE] Loading season-long statistics...")
    
    # Dictionnaire pour stocker les stats de saison
    season_stats = {}
    
    # Parcourir les ligues et années uniques
    leagues = df['league'].unique()
    years = df['year'].unique()
    
    for league in leagues:
        for year in years:
            # Construire le chemin vers les données de la ligue
            league_path = f"Data/{league}"
            if not os.path.exists(league_path):
                continue
                
            # Charger tous les fichiers CSV de l'année
            pattern = f"{league_path}/{year} *.csv"
            files = glob.glob(pattern)
            
            for file_path in files:
                try:
                    team_data = pd.read_csv(file_path)
                    if len(team_data) == 0:
                        continue
                        
                    # Extraire le nom de l'équipe du nom de fichier
                    team_name = os.path.basename(file_path).replace(f"{year} ", "").replace(".csv", "")
                    
                    # Calculer les stats de saison
                    team_stats = {
                        'season_avg_xG': team_data['team_xG'].mean(),
                        'season_avg_goals': team_data['team_goals'].mean(),
                        'season_avg_shots': team_data['team_shots'].mean(),
                        'season_avg_xG_against': team_data['opponent_xG'].mean(),
                        'season_avg_goals_against': team_data['opponent_goals'].mean(),
                        'season_avg_deep': team_data['team_deep'].mean(),
                        'season_avg_ppda_att': team_data['team_ppda_att'].mean(),
                        'season_avg_ppda_def': team_data['team_ppda_def'].mean(),
                        'season_goals_per_xG': team_data['team_goals'].sum() / (team_data['team_xG'].sum() + 0.1),
                        'season_xG_variance': team_data['team_xG'].var(),
                        'season_goals_variance': team_data['team_goals'].var(),
                        'season_total_matches': len(team_data),
                        'season_home_matches': len(team_data[team_data['is_home'] == True]),
                        'season_away_matches': len(team_data[team_data['is_home'] == False]),
                        'season_win_rate': len(team_data[team_data['result'] == 'w']) / len(team_data),
                        'season_draw_rate': len(team_data[team_data['result'] == 'd']) / len(team_data),
                        'season_loss_rate': len(team_data[team_data['result'] == 'l']) / len(team_data),
                        'season_clean_sheets': len(team_data[team_data['opponent_goals'] == 0]) / len(team_data),
                        'season_avg_xpts': team_data['team_xpts'].mean() if 'team_xpts' in team_data.columns else 0,
                    }
                    
                    # Ajouter les stats de forme à domicile/extérieur
                    home_data = team_data[team_data['is_home'] == True]
                    away_data = team_data[team_data['is_home'] == False]
                    
                    if len(home_data) > 0:
                        team_stats.update({
                            'season_home_avg_xG': home_data['team_xG'].mean(),
                            'season_home_avg_goals': home_data['team_goals'].mean(),
                            'season_home_win_rate': len(home_data[home_data['result'] == 'w']) / len(home_data),
                        })
                    else:
                        team_stats.update({
                            'season_home_avg_xG': 0,
                            'season_home_avg_goals': 0,
                            'season_home_win_rate': 0,
                        })
                    
                    if len(away_data) > 0:
                        team_stats.update({
                            'season_away_avg_xG': away_data['team_xG'].mean(),
                            'season_away_avg_goals': away_data['team_goals'].mean(),
                            'season_away_win_rate': len(away_data[away_data['result'] == 'w']) / len(away_data),
                        })
                    else:
                        team_stats.update({
                            'season_away_avg_xG': 0,
                            'season_away_avg_goals': 0,
                            'season_away_win_rate': 0,
                        })
                    
                    # Clé unique pour l'équipe
                    key = f"{league}_{year}_{team_name}"
                    season_stats[key] = team_stats
                    
                except Exception as e:
                    print(f"[WARNING] Error loading {file_path}: {e}")
                    continue
    
    print(f"[OK] Loaded season stats for {len(season_stats)} team-seasons")
    return season_stats

def add_season_metrics_features(df):
    """Ajoute des features basées sur les métriques de saison complète"""
    print("[FEATURE] Adding season-long metrics features...")
    
    # Charger les stats de saison
    season_stats = load_season_stats(df)
    
    # Initialiser les nouvelles colonnes
    season_features = [
        'season_avg_xG', 'season_avg_goals', 'season_avg_shots', 'season_avg_xG_against',
        'season_avg_goals_against', 'season_avg_deep', 'season_avg_ppda_att', 'season_avg_ppda_def',
        'season_goals_per_xG', 'season_xG_variance', 'season_goals_variance', 'season_total_matches',
        'season_home_matches', 'season_away_matches', 'season_win_rate', 'season_draw_rate',
        'season_loss_rate', 'season_clean_sheets', 'season_avg_xpts', 'season_home_avg_xG',
        'season_home_avg_goals', 'season_home_win_rate', 'season_away_avg_xG', 'season_away_avg_goals',
        'season_away_win_rate'
    ]
    
    # Ajouter les colonnes pour team et opponent
    for feature in season_features:
        df[f'team_{feature}'] = 0.0
        df[f'opponent_{feature}'] = 0.0
    
    # Remplir les données
    for idx, row in df.iterrows():
        # Clés pour team et opponent
        team_key = f"{row['league']}_{row['year']}_{row['team_name']}"
        opponent_key = f"{row['league']}_{row['year']}_{row['opponent_name']}"
        
        # Remplir les stats team
        if team_key in season_stats:
            for feature in season_features:
                if feature in season_stats[team_key]:
                    df.at[idx, f'team_{feature}'] = season_stats[team_key][feature]
        
        # Remplir les stats opponent
        if opponent_key in season_stats:
            for feature in season_features:
                if feature in season_stats[opponent_key]:
                    df.at[idx, f'opponent_{feature}'] = season_stats[opponent_key][feature]
    
    # Ajouter des features dérivées
    df['season_xG_advantage'] = df['team_season_avg_xG'] - df['opponent_season_avg_xG']
    df['season_goals_advantage'] = df['team_season_avg_goals'] - df['opponent_season_avg_goals']
    df['season_defensive_advantage'] = df['opponent_season_avg_goals_against'] - df['team_season_avg_goals_against']
    df['season_efficiency_advantage'] = df['team_season_goals_per_xG'] - df['opponent_season_goals_per_xG']
    df['season_experience_advantage'] = df['team_season_total_matches'] - df['opponent_season_total_matches']
    df['season_form_advantage'] = df['team_season_win_rate'] - df['opponent_season_win_rate']
    df['season_variance_stability'] = 1.0 / (df['team_season_xG_variance'] + df['opponent_season_xG_variance'] + 0.1)
    
    # Features de contexte domicile/extérieur 
    df['team_home_away_xG_diff'] = df['team_season_home_avg_xG'] - df['team_season_away_avg_xG']
    df['opponent_home_away_xG_diff'] = df['opponent_season_home_avg_xG'] - df['opponent_season_away_avg_xG']
    df['team_home_away_win_diff'] = df['team_season_home_win_rate'] - df['team_season_away_win_rate']
    df['opponent_home_away_win_diff'] = df['opponent_season_home_win_rate'] - df['opponent_season_away_win_rate']
    
    return df

def add_complementary_features(df):
    """Ajoute des features complémentaires basées sur l'analyse des features sélectionnées"""
    print("[FEATURE] Adding complementary features based on model analysis...")
    
    # 1. Features xG avancées (le modèle adore les xG)
    if all(col in df.columns for col in ['team_xG_last_5', 'team_xG_last_3', 'team_xG_last_1']):
        # xG momentum et accélération (patterns temporels)
        df['xG_momentum_short'] = df['team_xG_last_1'] / (df['team_xG_last_3'] + 0.1)
        df['xG_momentum_long'] = df['team_xG_last_3'] / (df['team_xG_last_5'] + 0.1)
        df['xG_acceleration'] = df['xG_momentum_short'] - df['xG_momentum_long']
        
        # xG volatility (consistance)
        df['xG_volatility'] = abs(df['team_xG_last_1'] - df['team_xG_last_3']) + abs(df['team_xG_last_3'] - df['team_xG_last_5'])
        df['opponent_xG_volatility'] = abs(df['opponent_xG_last_1'] - df['opponent_xG_last_3']) + abs(df['opponent_xG_last_3'] - df['opponent_xG_last_5'])
        df['xG_volatility_advantage'] = df['opponent_xG_volatility'] - df['xG_volatility']
    
    # 2. Features Deep avancées (aussi très sélectionnées)
    if all(col in df.columns for col in ['team_deep_last_5', 'team_deep_last_3', 'team_deep_last_1']):
        df['deep_momentum'] = df['team_deep_last_1'] / (df['team_deep_last_5'] + 0.1)
        df['opponent_deep_momentum'] = df['opponent_deep_last_1'] / (df['opponent_deep_last_5'] + 0.1)
        df['deep_momentum_clash'] = df['deep_momentum'] / (df['opponent_deep_momentum'] + 0.1)
        
        # Deep vs xG efficiency
        df['deep_per_xG'] = df['team_deep_last_5'] / (df['team_xG_last_5'] + 0.1)
        df['opponent_deep_per_xG'] = df['opponent_deep_last_5'] / (df['opponent_xG_last_5'] + 0.1)
        df['deep_efficiency_advantage'] = df['deep_per_xG'] - df['opponent_deep_per_xG']
    
    # 3. Market sophistication (très important pour le modèle)
    if all(col in df.columns for col in ['B365H', 'B365A', 'B365D', 'true_prob_home', 'true_prob_away']):
        # Market confidence levels
        df['market_confidence_level'] = np.where(df['odds_spread'] < 1.0, 'high',
                                                np.where(df['odds_spread'] < 2.0, 'medium', 'low'))
        df['market_confidence_numeric'] = np.where(df['market_confidence_level'] == 'high', 3,
                                                  np.where(df['market_confidence_level'] == 'medium', 2, 1))
        
        # Expected value calculations
        df['home_expected_value'] = (df['true_prob_home'] * df['B365H']) - 1
        df['away_expected_value'] = (df['true_prob_away'] * df['B365A']) - 1
        df['draw_expected_value'] = (df['true_prob_draw'] * df['B365D']) - 1
        df['best_expected_value'] = np.maximum.reduce([df['home_expected_value'], df['away_expected_value'], df['draw_expected_value']])
        
        # Odds momentum (changement de cotes implicite)
        df['odds_momentum'] = df['B365H'] / (df['B365A'] + 0.1)
        
    # 4. Strength sophistication (le modèle utilise beaucoup strength_gap)
    if all(col in df.columns for col in ['home_team_strength', 'away_team_strength']):
        # Strength categories
        df['strength_category'] = np.where(df['strength_gap'] > 0.2, 'home_dominant',
                                          np.where(df['strength_gap'] < -0.2, 'away_dominant', 'balanced'))
        df['strength_category_numeric'] = np.where(df['strength_category'] == 'home_dominant', 2,
                                                  np.where(df['strength_category'] == 'balanced', 1, 0))
        
        # Strength squared (effet non-linéaire)
        df['strength_gap_squared'] = df['strength_gap'] ** 2
        df['strength_gap_cubed'] = df['strength_gap'] ** 3
    
    # 5. Form sophistication (forme vs cotes très importante)
    if all(col in df.columns for col in ['form_x_odds', 'relative_form_5']):
        # Form categories
        df['form_trend_category'] = np.where(df['relative_form_5'] > 0.3, 'hot',
                                            np.where(df['relative_form_5'] < -0.3, 'cold', 'neutral'))
        df['form_trend_numeric'] = np.where(df['form_trend_category'] == 'hot', 2,
                                           np.where(df['form_trend_category'] == 'neutral', 1, 0))
        
        # Form vs Market mismatch
        if 'market_inefficiency_home' in df.columns:
            df['form_market_mismatch'] = abs(df['relative_form_5'] - df['market_inefficiency_home'])
    
    # 6. Style clash sophistication (style_clash est sélectionné)
    if all(col in df.columns for col in ['attack_defense_ratio', 'opponent_attack_defense_ratio']):
        # Style mismatch categories  
        df['style_mismatch'] = np.where(
            (df['attack_defense_ratio'] > 2) & (df['opponent_attack_defense_ratio'] > 2), 'both_attacking',
            np.where(
                (df['attack_defense_ratio'] < 0.5) & (df['opponent_attack_defense_ratio'] < 0.5), 'both_defensive',
                'mixed'
            )
        )
        df['style_mismatch_numeric'] = np.where(df['style_mismatch'] == 'both_attacking', 2,
                                               np.where(df['style_mismatch'] == 'mixed', 1, 0))
        
        # Attack vs Defense dominance
        df['attack_dominance'] = np.maximum(df['attack_defense_ratio'], df['opponent_attack_defense_ratio'])
        df['defense_dominance'] = np.minimum(df['attack_defense_ratio'], df['opponent_attack_defense_ratio'])
    
    # 7. Saison features plus sophistiqués (pour compléter season_form_advantage)
    if 'team_season_avg_xG' in df.columns and 'opponent_season_avg_xG' in df.columns:
        # Season consistency features
        df['season_xG_consistency'] = 1.0 / (df['team_season_xG_variance'] + df['opponent_season_xG_variance'] + 0.1)
        df['season_vs_recent_xG'] = df['team_xG_last_5'] / (df['team_season_avg_xG'] + 0.1)
        df['opponent_season_vs_recent_xG'] = df['opponent_xG_last_5'] / (df['opponent_season_avg_xG'] + 0.1)
        df['season_recent_divergence'] = abs(df['season_vs_recent_xG'] - df['opponent_season_vs_recent_xG'])
        
        # Season home/away advantage
        if all(col in df.columns for col in ['team_season_home_avg_xG', 'team_season_away_avg_xG']):
            df['team_home_advantage_strength'] = df['team_season_home_avg_xG'] / (df['team_season_away_avg_xG'] + 0.1)
            df['opponent_home_advantage_strength'] = df['opponent_season_home_avg_xG'] / (df['opponent_season_away_avg_xG'] + 0.1)
    
    # 8. H2H sophistication (h2h_dominance est sélectionné)
    if 'h2h_dominance' in df.columns:
        # H2H categories
        df['h2h_category'] = np.where(df['h2h_dominance'] > 0.6, 'dominant',
                                     np.where(df['h2h_dominance'] < 0.4, 'dominated', 'balanced'))
        df['h2h_category_numeric'] = np.where(df['h2h_category'] == 'dominant', 2,
                                             np.where(df['h2h_category'] == 'balanced', 1, 0))
    
    # 9. Features composites (combinaisons des features importantes)
    if all(col in df.columns for col in ['xG_advantage_5', 'strength_gap', 'market_inefficiency_home']):
        df['composite_advantage'] = (df['xG_advantage_5'] * 0.4 + 
                                    df['strength_gap'] * 0.3 + 
                                    df['market_inefficiency_home'] * 0.3)
        
        df['triple_alignment'] = np.where(
            (df['xG_advantage_5'] > 0) & (df['strength_gap'] > 0) & (df['market_inefficiency_home'] > 0), 1,
            np.where(
                (df['xG_advantage_5'] < 0) & (df['strength_gap'] < 0) & (df['market_inefficiency_home'] < 0), -1,
                0
            )
        )
    
    return df

def add_advanced_team_features(df):
    """Ajoute des features avancées d'équipe"""
    print("[FEATURE] Adding advanced team features...")
    
    # 1. Ratios de performance
    df['xG_efficiency_5'] = df['team_xG_last_5'] / (df['team_xG_against_last_5'] + 0.1)
    df['xG_efficiency_3'] = df['team_xG_last_3'] / (df['team_xG_against_last_3'] + 0.1)
    df['xG_efficiency_1'] = df['team_xG_last_1'] / (df['team_xG_against_last_1'] + 0.1)
    
    # 2. Momentum features
    df['recent_xG_trend'] = df['team_xG_last_1'] - df['team_xG_last_5']
    df['defensive_trend'] = df['team_xG_against_last_1'] - df['team_xG_against_last_5']
    df['ppda_pressure_trend'] = df['team_ppda_last_1'] - df['team_ppda_last_5']
    
    # 3. Consistency features
    periods = ['last_5', 'last_3', 'last_1']
    for feature in ['team_xG', 'team_deep', 'team_ppda']:
        values = [df[f'{feature}_{period}'] for period in periods]
        df[f'{feature}_consistency'] = 1.0 / (np.std(values, axis=0) + 0.1)
    
    return df

def add_opponent_comparative_features(df):
    """Ajoute des features comparatives avec l'adversaire"""
    print("[FEATURE] Adding opponent comparative features...")
    
    # 1. Différences directes
    df['xG_advantage_5'] = df['team_xG_last_5'] - df['opponent_xG_last_5']
    df['xG_advantage_3'] = df['team_xG_last_3'] - df['opponent_xG_last_3'] 
    df['xG_advantage_1'] = df['team_xG_last_1'] - df['opponent_xG_last_1']
    
    df['defensive_advantage_5'] = df['opponent_xG_against_last_5'] - df['team_xG_against_last_5']
    df['defensive_advantage_3'] = df['opponent_xG_against_last_3'] - df['team_xG_against_last_3']
    df['defensive_advantage_1'] = df['opponent_xG_against_last_1'] - df['team_xG_against_last_1']
    
    df['deep_advantage_5'] = df['team_deep_last_5'] - df['opponent_deep_last_5']
    df['deep_advantage_3'] = df['team_deep_last_3'] - df['opponent_deep_last_3']
    df['deep_advantage_1'] = df['team_deep_last_1'] - df['opponent_deep_last_1']
    
    df['ppda_advantage_5'] = df['team_ppda_last_5'] - df['opponent_ppda_last_5']
    df['ppda_advantage_3'] = df['team_ppda_last_3'] - df['opponent_ppda_last_3']
    df['ppda_advantage_1'] = df['team_ppda_last_1'] - df['opponent_ppda_last_1']
    
    # 2. Ratios
    df['xG_ratio_5'] = df['team_xG_last_5'] / (df['opponent_xG_last_5'] + 0.1)
    df['xG_ratio_3'] = df['team_xG_last_3'] / (df['opponent_xG_last_3'] + 0.1)
    df['xG_ratio_1'] = df['team_xG_last_1'] / (df['opponent_xG_last_1'] + 0.1)
    
    return df

def add_betting_intelligence_features(df):
    """Ajoute des features d'intelligence basées sur les cotes"""
    print("[FEATURE] Adding betting intelligence features...")
    
    # 1. Probabilités implicites
    df['prob_home'] = 1 / df['B365H']
    df['prob_draw'] = 1 / df['B365D'] 
    df['prob_away'] = 1 / df['B365A']
    
    # 2. Marge du bookmaker
    df['bookmaker_margin'] = df['prob_home'] + df['prob_draw'] + df['prob_away'] - 1
    
    # 3. Vraies probabilités (sans marge)
    total_prob = df['prob_home'] + df['prob_draw'] + df['prob_away']
    df['true_prob_home'] = df['prob_home'] / total_prob
    df['true_prob_draw'] = df['prob_draw'] / total_prob
    df['true_prob_away'] = df['prob_away'] / total_prob
    
    # 4. Features de favori
    df['is_home_favorite'] = (df['B365H'] <= df['B365A']) & (df['B365H'] <= df['B365D'])
    df['is_away_favorite'] = (df['B365A'] <= df['B365H']) & (df['B365A'] <= df['B365D'])
    df['is_draw_favorite'] = (df['B365D'] <= df['B365H']) & (df['B365D'] <= df['B365A'])
    
    # 5. Écarts de cotes
    df['home_away_odds_gap'] = abs(df['B365H'] - df['B365A'])
    df['favorite_odds'] = np.minimum(np.minimum(df['B365H'], df['B365A']), df['B365D'])
    df['underdog_odds'] = np.maximum(np.maximum(df['B365H'], df['B365A']), df['B365D'])
    df['odds_spread'] = df['underdog_odds'] - df['favorite_odds']
    
    # 6. Incertitude du match
    entropies = -(df['true_prob_home'] * np.log(df['true_prob_home'] + 1e-10) +
                  df['true_prob_draw'] * np.log(df['true_prob_draw'] + 1e-10) +
                  df['true_prob_away'] * np.log(df['true_prob_away'] + 1e-10))
    df['match_uncertainty'] = entropies

    # 7. Spreads avancés basés sur les probabilités et cotes
    epsilon = 1e-6
    prob_matrix = df[['true_prob_home', 'true_prob_draw', 'true_prob_away']].to_numpy(dtype=float)
    prob_matrix = np.where(np.isnan(prob_matrix), 0.0, prob_matrix)
    sorted_probs = np.sort(prob_matrix, axis=1)

    df['implied_prob_spread'] = sorted_probs[:, -1] - sorted_probs[:, 0]
    df['favorite_implied_edge'] = sorted_probs[:, -1] - sorted_probs[:, -2]
    df['prob_home_vs_away_gap'] = df['true_prob_home'] - df['true_prob_away']
    df['prob_draw_vs_favorite_gap'] = df['true_prob_draw'] - sorted_probs[:, -1]

    df['odds_ratio_home_away'] = df['B365H'] / (df['B365A'] + epsilon)
    df['log_odds_spread'] = np.log(df['B365H'] + epsilon) - np.log(df['B365A'] + epsilon)
    df['normalized_odds_spread'] = df['odds_spread'] / (df['favorite_odds'] + epsilon)

    df['logit_true_prob_home'] = np.log((df['true_prob_home'] + epsilon) / (1 - df['true_prob_home'] + epsilon))
    df['logit_true_prob_away'] = np.log((df['true_prob_away'] + epsilon) / (1 - df['true_prob_away'] + epsilon))
    df['logit_true_prob_draw'] = np.log((df['true_prob_draw'] + epsilon) / (1 - df['true_prob_draw'] + epsilon))
    df['logit_prob_gap_home_away'] = df['logit_true_prob_home'] - df['logit_true_prob_away']
    
    return df

def add_advanced_market_features(df):
    """Ajouter des features avancées basées sur les probabilités market (TOP importance)"""
    print("[FEATURE] Adding advanced market intelligence features...")
    
    # 1. Divergence entre true_prob et prob (market inefficiency)
    if all(col in df.columns for col in ['true_prob_home', 'prob_home']):
        df['market_inefficiency_home'] = df['true_prob_home'] - df['prob_home']
        df['market_inefficiency_away'] = df['true_prob_away'] - df['prob_away']
        df['market_inefficiency_draw'] = df['true_prob_draw'] - df['prob_draw']
        
        # Magnitude totale d'inefficacité du marché
        df['total_market_inefficiency'] = (
            abs(df['market_inefficiency_home']) + 
            abs(df['market_inefficiency_away']) + 
            abs(df['market_inefficiency_draw'])
        )
    
    # 2. Confidence du marché (écart entre probabilités)
    if all(col in df.columns for col in ['true_prob_home', 'true_prob_away', 'true_prob_draw']):
        df['market_confidence'] = np.maximum.reduce([
            df['true_prob_home'], df['true_prob_away'], df['true_prob_draw']
        ]) - np.minimum.reduce([
            df['true_prob_home'], df['true_prob_away'], df['true_prob_draw']
        ])
    
    # 3. Ratio de probabilités extrêmes
    if 'true_prob_away' in df.columns and 'true_prob_home' in df.columns:
        df['prob_ratio_away_home'] = df['true_prob_away'] / (df['true_prob_home'] + 0.001)
        df['prob_dominance'] = np.where(
            df['true_prob_home'] > df['true_prob_away'],
            df['true_prob_home'] / (df['true_prob_away'] + 0.001),
            df['true_prob_away'] / (df['true_prob_home'] + 0.001)
        )
    
    return df

def add_temporal_features(df):
    """Ajoute des features temporelles"""
    print("[FEATURE] Adding temporal features...")
    
    # 1. Features de saison
    df['season_progress'] = (df['month'] - 8) / 10  # Août = début, Mai = fin
    df['season_progress'] = df['season_progress'].clip(0, 1)
    
    # 2. Features de mois
    df['is_early_season'] = (df['month'] <= 10).astype(int)
    df['is_mid_season'] = ((df['month'] >= 11) & (df['month'] <= 2)).astype(int) 
    df['is_late_season'] = (df['month'] >= 3).astype(int)
    
    # 3. Interaction année-mois
    df['year_month'] = df['year'] + df['month'] / 12.0
    
    return df

def add_advanced_strength_features(df):
    """Ajouter des features avancées de force d'équipe (2ème plus importante catégorie)"""
    print("[FEATURE] Adding advanced strength features...")
    
    # 1. Interaction strength x home advantage
    if all(col in df.columns for col in ['home_team_strength', 'home_field_advantage']):
        df['boosted_home_strength'] = df['home_team_strength'] * (1 + df['home_field_advantage'])
        df['strength_gap'] = df['home_team_strength'] - df['away_team_strength']
        df['strength_gap_with_home'] = df['boosted_home_strength'] - df['away_team_strength']
    
    # 2. Strength momentum (strength x form)
    if all(col in df.columns for col in ['home_team_strength', 'form_score_5']):
        df['home_strength_momentum'] = df['home_team_strength'] * df['form_score_5']
        df['away_strength_momentum'] = df['away_team_strength'] * df['opponent_form_score_5']
        df['momentum_gap'] = df['home_strength_momentum'] - df['away_strength_momentum']
    
    # 3. Weighted strength par consistency
    if 'form_score_10' in df.columns and 'form_score_5' in df.columns:
        df['form_consistency'] = abs(df['form_score_10'] - df['form_score_5'])
        df['consistent_strength'] = df['home_team_strength'] * (1 - df['form_consistency'])
    
    return df

def add_elo_schedule_features(df):
    """Ajoute des features combinant Elo et gestion du repos"""
    print("[FEATURE] Adding Elo & schedule synergy features...")
    
    if 'elo_rating_gap' in df.columns:
        df['elo_gap_abs'] = df['elo_rating_gap'].abs()
        df['elo_gap_squared'] = df['elo_rating_gap'] ** 2
    
    if 'elo_win_probability' in df.columns:
        df['elo_prob_confidence'] = abs(df['elo_win_probability'] - 0.5)
    
    if all(col in df.columns for col in ['elo_win_probability', 'true_prob_home']):
        df['elo_vs_market_gap'] = df['elo_win_probability'] - df['true_prob_home']
    
    if all(col in df.columns for col in ['team_rest_days', 'opponent_rest_days']):
        df['fatigue_flag_team'] = (df['team_rest_days'].fillna(0) < 4).astype(int)
        df['fatigue_flag_opponent'] = (df['opponent_rest_days'].fillna(0) < 4).astype(int)
        df['fatigue_mismatch'] = df['fatigue_flag_opponent'] - df['fatigue_flag_team']
    
    if 'rest_days_diff' in df.columns:
        df['rest_days_clipped'] = df['rest_days_diff'].clip(-7, 7)
    
    if all(col in df.columns for col in ['rest_days_diff', 'strength_gap']):
        df['rest_strength_synergy'] = df['rest_days_diff'].fillna(0) * df['strength_gap']
    
    if all(col in df.columns for col in ['elo_rating_gap', 'rest_days_diff']):
        df['elo_rest_synergy'] = df['elo_rating_gap'] * df['rest_days_diff'].fillna(0)
    
    if all(col in df.columns for col in ['elo_rating_gap', 'form_momentum']):
        df['elo_form_synergy'] = df['elo_rating_gap'] * df['form_momentum']
    
    return df

def add_form_momentum_features(df):
    """Ajoute des features avancées de forme"""
    print("[FEATURE] Adding form momentum features...")
    
    # 1. Momentum de forme
    if 'form_score_5' in df.columns and 'form_score_10' in df.columns:
        df['form_acceleration'] = df['form_score_5'] - df['form_score_10']
        if 'form_consistency' not in df.columns:  # Avoid duplicate if already created
            df['form_consistency'] = abs(df['form_score_5'] - df['form_score_10'])
        
        # Forme relative
        df['relative_form_5'] = df['form_score_5'] - df['opponent_form_score_5']
        df['relative_form_10'] = df['form_score_10'] - df['opponent_form_score_10']
        
        # Form trend analysis (from advanced_features.py)
        df['form_trend'] = df['form_score_5'] - df['form_score_10']
        df['form_stability'] = 1 - abs(df['form_trend'])
        df['opponent_form_trend'] = df['opponent_form_score_5'] - df['opponent_form_score_10']
        df['form_clash'] = df['form_trend'] - df['opponent_form_trend']
    
    # 2. Streaks features
    if 'unbeaten_streak' in df.columns and 'winless_streak' in df.columns:
        df['streak_balance'] = df['unbeaten_streak'] - df['winless_streak']
        df['has_positive_streak'] = (df['unbeaten_streak'] > 0).astype(int)
        df['has_negative_streak'] = (df['winless_streak'] > 0).astype(int)
        
        # Advanced streak analysis
        df['streak_momentum'] = np.log1p(df['unbeaten_streak'])
        if 'form_score_5' in df.columns:
            df['streak_quality'] = df['streak_momentum'] * df['form_score_5']
    
    # 3. Form x odds interaction (important feature)
    if all(col in df.columns for col in ['form_score_5', 'B365H', 'B365A']):
        # Nouvelle variante: form consistency x odds
        if 'form_stability' in df.columns:
            df['stable_form_x_odds'] = df['form_stability'] * df['form_x_odds']
        
        # Over/undervalued par le marché
        expected_odds = 1 / (df['form_score_5'] + 0.1)
        df['odds_vs_form_home'] = df['B365H'] / expected_odds
        
        opponent_expected_odds = 1 / (df['opponent_form_score_5'] + 0.1)
        df['odds_vs_form_away'] = df['B365A'] / opponent_expected_odds
    
    return df

def add_home_away_features(df):
    """Ajoute des features basées sur jouer à domicile/extérieur"""
    print("[FEATURE] Adding home/away context features...")
    
    # On peut inférer qui joue à domicile par les cotes et probabilités
    # Généralement, l'équipe avec les meilleures cotes à domicile joue à domicile
    df['likely_home_team'] = (df['true_prob_home'] > df['true_prob_away']).astype(int)
    
    # Avantage du terrain
    df['home_field_advantage'] = df['true_prob_home'] - df['true_prob_away']
    
    # Force de l'équipe à domicile vs à l'extérieur
    df['home_team_strength'] = df['true_prob_home'] + df['true_prob_draw'] * 0.5
    df['away_team_strength'] = df['true_prob_away'] + df['true_prob_draw'] * 0.5
    
    return df

def add_advanced_xg_features(df):
    """Ajouter des features xG avancées basées sur xG vs market"""
    print("[FEATURE] Adding advanced xG features...")
    
    # 1. xG momentum (combinaisons court/moyen/long terme)
    xg_periods = ['1', '3', '5']
    for period in xg_periods:
        if f'xG_vs_market_{period}' in df.columns:
            # xG momentum = recent performance vs expected
            if f'team_xG_last_{period}' in df.columns:
                df[f'xG_momentum_{period}'] = df[f'team_xG_last_{period}'] * df[f'xG_vs_market_{period}']
            
            # xG advantage amplifié par market performance
            if f'xG_advantage_{period}' in df.columns:
                df[f'xG_market_advantage_{period}'] = df[f'xG_advantage_{period}'] * (1 + df[f'xG_vs_market_{period}'])
    
    # 2. Defensive vs offensive patterns
    if all(col in df.columns for col in ['team_xG_last_5', 'team_xG_against_last_5']):
        df['attack_defense_ratio'] = df['team_xG_last_5'] / (df['team_xG_against_last_5'] + 0.1)
        
        if all(col in df.columns for col in ['opponent_xG_last_5', 'opponent_xG_against_last_5']):
            df['opponent_attack_defense_ratio'] = df['opponent_xG_last_5'] / (df['opponent_xG_against_last_5'] + 0.1)
            df['style_clash'] = df['attack_defense_ratio'] / (df['opponent_attack_defense_ratio'] + 0.1)
    
    # 3. xG trend analysis
    if all(col in df.columns for col in ['team_xG_last_1', 'team_xG_last_3', 'team_xG_last_5']):
        df['xG_trend_short'] = df['team_xG_last_1'] - df['team_xG_last_3']
        df['xG_trend_long'] = df['team_xG_last_3'] - df['team_xG_last_5']
        df['xG_acceleration'] = df['xG_trend_short'] - df['xG_trend_long']
    
    return df

def add_interaction_features(df):
    """Ajoute des features d'interaction complexes"""
    print("[FEATURE] Adding interaction features...")
    
    # 1. Interactions forme x qualité
    if 'form_score_5' in df.columns:
        df['form_x_odds'] = df['form_score_5'] * df['true_prob_home']
        df['opponent_form_x_odds'] = df['opponent_form_score_5'] * df['true_prob_away']
    
    # 2. Interactions xG x cotes
    df['xG_vs_market_5'] = df['team_xG_last_5'] * df['true_prob_home']
    df['xG_vs_market_3'] = df['team_xG_last_3'] * df['true_prob_home']
    df['xG_vs_market_1'] = df['team_xG_last_1'] * df['true_prob_home']
    
    # 3. Interactions défensives
    df['defense_vs_market'] = (1 / (df['team_xG_against_last_5'] + 0.1)) * df['true_prob_home']
    
    # 4. H2H x momentum
    if 'h2h_dominance' in df.columns and 'form_momentum' in df.columns:
        df['h2h_x_momentum'] = df['h2h_dominance'] * df['form_momentum']
    
    # 5. Advanced interactions from advanced_features.py
    # Market confidence x team strength
    if all(col in df.columns for col in ['market_confidence', 'strength_gap']):
        df['confident_strength'] = df['market_confidence'] * abs(df['strength_gap'])
    
    # Market inefficiency x form
    if all(col in df.columns for col in ['total_market_inefficiency', 'form_clash']):
        df['market_form_opportunity'] = df['total_market_inefficiency'] * abs(df['form_clash'])
    
    # xG momentum x probability divergence
    if all(col in df.columns for col in ['xG_momentum_5', 'market_inefficiency_home']):
        df['xG_market_signal'] = df['xG_momentum_5'] * df['market_inefficiency_home']
    
    # Compound advantage score
    advantage_features = [col for col in df.columns if 'advantage' in col or 'momentum' in col]
    if len(advantage_features) >= 3:
        df['compound_advantage'] = df[advantage_features].mean(axis=1)
    
    return df

def add_all_enhanced_features(df_input):
    """
    Version optimisée qui prend un DataFrame en entrée et retourne le DataFrame enrichi
    sans sauvegarde de fichier
    """
    print("="*80)
    print("ENHANCED FEATURE ENGINEERING - MEMORY MODE")
    print("="*80)
    
    # Work with copy
    df = df_input.copy()
    print(f"[INPUT] Processing DataFrame with {len(df)} rows, {df.shape[1]} columns")
    
    # Store original column count
    original_features = df.shape[1]
    
    # Apply all feature engineering steps (order matters for dependencies)
    df = add_season_metrics_features(df)  # Add season statistics first
    df = add_advanced_team_features(df)
    df = add_opponent_comparative_features(df)
    df = add_betting_intelligence_features(df)
    df = add_advanced_market_features(df)
    df = add_temporal_features(df)
    df = add_home_away_features(df)  # Must come before advanced_strength_features
    df = add_advanced_strength_features(df)
    df = add_elo_schedule_features(df)
    df = add_interaction_features(df)  # Must come before form_momentum_features (creates form_x_odds)
    df = add_form_momentum_features(df)
    df = add_advanced_xg_features(df)
    df = add_complementary_features(df)  # Add targeted features based on model analysis
    
    # Clean infinities and NaN
    print("[CLEANING] Handling infinities and NaN values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Summary
    new_features = df.shape[1] - original_features
    print(f"[SUMMARY] Added {new_features} new features")
    print(f"[SUMMARY] Total features: {df.shape[1]}")
    print(f"[OK] Enhanced features added in memory")
    print("="*80)
    
    return df

def enhance_dataset(input_file="preprocessed_data_with_odds.csv", output_file="preprocessed_data_enhanced.csv"):
    """Pipeline complet d'amélioration des features"""
    print("="*80)
    print("ENHANCED FEATURE ENGINEERING PIPELINE")
    print("="*80)
    
    # Load data
    print(f"[LOADING] Reading {input_file}...")
    df = pd.read_csv(input_file)
    print(f"[OK] {len(df)} rows, {df.shape[1]} columns loaded")
    
    # Store original column count
    original_features = df.shape[1]
    
    # Apply all feature engineering steps (order matters for dependencies)
    df = add_season_metrics_features(df)  # Add season statistics first
    df = add_advanced_team_features(df)
    df = add_opponent_comparative_features(df)
    df = add_betting_intelligence_features(df)
    df = add_advanced_market_features(df)
    df = add_temporal_features(df)
    df = add_home_away_features(df)  # Must come before advanced_strength_features
    df = add_advanced_strength_features(df)
    df = add_elo_schedule_features(df)
    df = add_interaction_features(df)  # Must come before form_momentum_features (creates form_x_odds)
    df = add_form_momentum_features(df)
    df = add_advanced_xg_features(df)
    df = add_complementary_features(df)  # Add targeted features based on model analysis
    
    # Clean infinities and NaN
    print("[CLEANING] Handling infinities and NaN values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Summary
    new_features = df.shape[1] - original_features
    print(f"[SUMMARY] Added {new_features} new features")
    print(f"[SUMMARY] Total features: {df.shape[1]}")
    
    # Save enhanced dataset
    print(f"[SAVING] Writing to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"[OK] Enhanced dataset saved: {len(df)} rows, {df.shape[1]} columns")
    print("="*80)
    
    return df

# Alias for backward compatibility with advanced_features.py
add_all_advanced_features = add_all_enhanced_features

if __name__ == "__main__":
    enhanced_df = enhance_dataset()
    print("Enhanced feature engineering completed!")
