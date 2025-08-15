#!/usr/bin/env python3
"""
Advanced Feature Engineering for Improved Accuracy
Basé sur l'analyse d'importance des features du modèle RandomForest
"""

import pandas as pd
import numpy as np

def add_advanced_market_features(df):
    """Ajouter des features avancées basées sur les probabilités market (TOP importance)"""
    
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

def add_advanced_strength_features(df):
    """Ajouter des features avancées de force d'équipe (2ème plus importante catégorie)"""
    
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

def add_advanced_xg_features(df):
    """Ajouter des features xG avancées basées sur xG vs market"""
    
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

def add_advanced_form_features(df):
    """Ajouter des features de forme avancées"""
    
    # 1. Form momentum et volatilité
    if all(col in df.columns for col in ['form_score_5', 'form_score_10']):
        df['form_trend'] = df['form_score_5'] - df['form_score_10']
        df['form_stability'] = 1 - abs(df['form_trend'])
        df['opponent_form_trend'] = df['opponent_form_score_5'] - df['opponent_form_score_10']
        
        # Form clash analysis
        df['form_clash'] = df['form_trend'] - df['opponent_form_trend']
    
    # 2. Form x odds interaction (importante feature existante)
    if all(col in df.columns for col in ['form_score_5', 'B365H', 'B365A']):
        # Nouvelle variante: form consistency x odds
        if 'form_stability' in df.columns:
            df['stable_form_x_odds'] = df['form_stability'] * df['form_x_odds']
        
        # Over/undervalued par le marché
        expected_odds = 1 / (df['form_score_5'] + 0.1)  # Plus de forme = cotes plus faibles
        df['odds_vs_form_home'] = df['B365H'] / expected_odds
        
        opponent_expected_odds = 1 / (df['opponent_form_score_5'] + 0.1)
        df['odds_vs_form_away'] = df['B365A'] / opponent_expected_odds
    
    # 3. Streak analysis avancé
    if 'unbeaten_streak' in df.columns:
        df['streak_momentum'] = np.log1p(df['unbeaten_streak'])  # Log scale pour les longues séries
        if 'form_score_5' in df.columns:
            df['streak_quality'] = df['streak_momentum'] * df['form_score_5']
    
    return df

def add_interaction_features(df):
    """Ajouter des features d'interaction entre les catégories importantes"""
    
    # 1. Market confidence x team strength
    if all(col in df.columns for col in ['market_confidence', 'strength_gap']):
        df['confident_strength'] = df['market_confidence'] * abs(df['strength_gap'])
    
    # 2. Market inefficiency x form
    if all(col in df.columns for col in ['total_market_inefficiency', 'form_clash']):
        df['market_form_opportunity'] = df['total_market_inefficiency'] * abs(df['form_clash'])
    
    # 3. xG momentum x probability divergence
    if all(col in df.columns for col in ['xG_momentum_5', 'market_inefficiency_home']):
        df['xG_market_signal'] = df['xG_momentum_5'] * df['market_inefficiency_home']
    
    # 4. Compound advantage score
    advantage_features = [col for col in df.columns if 'advantage' in col or 'momentum' in col]
    if len(advantage_features) >= 3:
        df['compound_advantage'] = df[advantage_features].mean(axis=1)
    
    return df

def add_all_advanced_features(df):
    """Pipeline complet des nouvelles features avancées"""
    
    print("[ADVANCED] Adding market intelligence features...")
    df = add_advanced_market_features(df)
    
    print("[ADVANCED] Adding strength interaction features...")
    df = add_advanced_strength_features(df)
    
    print("[ADVANCED] Adding xG momentum features...")  
    df = add_advanced_xg_features(df)
    
    print("[ADVANCED] Adding form pattern features...")
    df = add_advanced_form_features(df)
    
    print("[ADVANCED] Adding interaction features...")
    df = add_interaction_features(df)
    
    # Nettoyer les infinités et NaN
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    return df

if __name__ == "__main__":
    # Test avec un échantillon
    print("Testing advanced features...")
    
    # Créer des données de test
    test_data = {
        'true_prob_home': [0.6], 'prob_home': [0.5], 'true_prob_away': [0.2], 'prob_away': [0.3],
        'true_prob_draw': [0.2], 'prob_draw': [0.2], 'home_team_strength': [0.8], 'away_team_strength': [0.4],
        'home_field_advantage': [0.15], 'form_score_5': [0.7], 'opponent_form_score_5': [0.3],
        'form_score_10': [0.65], 'opponent_form_score_10': [0.35], 'team_xG_last_5': [2.1],
        'team_xG_against_last_5': [0.8], 'opponent_xG_last_5': [1.1], 'opponent_xG_against_last_5': [1.5],
        'xG_vs_market_5': [0.2], 'B365H': [1.5], 'B365A': [3.0], 'form_x_odds': [1.05], 'unbeaten_streak': [5]
    }
    
    df_test = pd.DataFrame(test_data)
    print(f"Dataset original: {df_test.shape[1]} features")
    
    df_enhanced = add_all_advanced_features(df_test)
    print(f"Dataset enhancé: {df_enhanced.shape[1]} features")
    print(f"Nouvelles features: {df_enhanced.shape[1] - df_test.shape[1]}")
    
    # Afficher quelques nouvelles features
    new_features = [col for col in df_enhanced.columns if col not in df_test.columns]
    print(f"\\nExemples de nouvelles features:")
    for feature in new_features[:10]:
        print(f"  {feature}: {df_enhanced[feature].iloc[0]:.3f}")