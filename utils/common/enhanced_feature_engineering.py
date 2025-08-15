#!/usr/bin/env python3
"""
Enhanced feature engineering pour améliorer les performances du modèle.
Ajoute des features avancées basées sur l'analyse des données football.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

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

def add_form_momentum_features(df):
    """Ajoute des features avancées de forme"""
    print("[FEATURE] Adding form momentum features...")
    
    # 1. Momentum de forme
    if 'form_score_5' in df.columns and 'form_score_10' in df.columns:
        df['form_acceleration'] = df['form_score_5'] - df['form_score_10']
        df['form_consistency'] = abs(df['form_score_5'] - df['form_score_10'])
        
        # Forme relative
        df['relative_form_5'] = df['form_score_5'] - df['opponent_form_score_5']
        df['relative_form_10'] = df['form_score_10'] - df['opponent_form_score_10']
    
    # 2. Streaks features
    if 'unbeaten_streak' in df.columns and 'winless_streak' in df.columns:
        df['streak_balance'] = df['unbeaten_streak'] - df['winless_streak']
        df['has_positive_streak'] = (df['unbeaten_streak'] > 0).astype(int)
        df['has_negative_streak'] = (df['winless_streak'] > 0).astype(int)
    
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
    
    # Apply all feature engineering steps
    df = add_advanced_team_features(df)
    df = add_opponent_comparative_features(df)
    df = add_betting_intelligence_features(df)
    df = add_temporal_features(df)
    df = add_form_momentum_features(df)
    df = add_home_away_features(df)
    df = add_interaction_features(df)
    
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
    
    # Apply all feature engineering steps
    df = add_advanced_team_features(df)
    df = add_opponent_comparative_features(df)
    df = add_betting_intelligence_features(df)
    df = add_temporal_features(df)
    df = add_form_momentum_features(df)
    df = add_home_away_features(df)
    df = add_interaction_features(df)
    
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

if __name__ == "__main__":
    enhanced_df = enhance_dataset()
    print("Enhanced feature engineering completed!")