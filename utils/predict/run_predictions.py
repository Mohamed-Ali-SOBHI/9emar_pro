#!/usr/bin/env python3
"""
Module de prédiction pour les matchs de football
Utilise le modèle trained pour faire des prédictions sur les matchs fournis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_model_exists():
    """Vérifie si le modèle trained existe"""
    model_dir = Path("models/optimal_model")
    required_files = [
        "trained_model_realistic.pkl",
        "scaler_realistic.pkl", 
        "features_realistic.json"
    ]
    
    if not model_dir.exists():
        return False
    
    for file in required_files:
        if not (model_dir / file).exists():
            return False
    
    return True

def run_predictions(df_for_prediction):
    """
    Fait les prédictions pour les matchs fournis
    
    Args:
        df_for_prediction (DataFrame): Données des matchs avec toutes les features nécessaires
        
    Returns:
        tuple: (success: bool, results: list)
    """
    try:
        # Vérifier si le modèle existe
        if not check_model_exists():
            print("[ERROR] Modèle trained non trouvé!")
            print("[INFO] Exécutez d'abord: python utils/train/realistic_training.py")
            return False, []
        
        # Charger le modèle et faire les prédictions
        model_dir = Path("models/optimal_model")
        
        # Charger le modèle, scaler et features
        model = joblib.load(model_dir / "trained_model_realistic.pkl")
        scaler = joblib.load(model_dir / "scaler_realistic.pkl")
        
        with open(model_dir / "features_realistic.json", 'r') as f:
            model_features = json.load(f)
        
        print(f"[INFO] Modèle chargé avec {len(model_features)} features")
        
        # Vérifier que toutes les features requises sont présentes
        missing_features = [f for f in model_features if f not in df_for_prediction.columns]
        if missing_features:
            print(f"[ERROR] Features requises manquantes: {missing_features}")
            print(f"[INFO] Impossible de faire des prédictions sans toutes les features nécessaires")
            return False, []
        
        # Sélectionner seulement les features du modèle
        X = df_for_prediction[model_features].fillna(0)
        
        # Scaler les données
        X_scaled = scaler.transform(X)
        
        print(f"[DEBUG] Shape des données pour prédiction: {X_scaled.shape}")
        print(f"[DEBUG] Types des données: {X.dtypes.value_counts()}")
        
        # Faire les prédictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Préparer les résultats
        results = []
        for i, (_, match) in enumerate(df_for_prediction.iterrows()):
            pred_label = predictions[i]
            pred_probs = probabilities[i]
            
            # Correspondance des classes
            class_names = ['Loss', 'Draw', 'Win']  # Basé sur l'ordre habituel du modèle
            
            result = {
                'league': match['league'],
                'home_team': match['team_name'],
                'away_team': match['opponent_name'],
                'prediction': pred_label,
                'prediction_label': class_names[pred_label] if pred_label < len(class_names) else f"Class_{pred_label}",
                'prob_loss': pred_probs[0] if len(pred_probs) > 0 else 0,
                'prob_draw': pred_probs[1] if len(pred_probs) > 1 else 0,
                'prob_win': pred_probs[2] if len(pred_probs) > 2 else 0,
                'confidence': max(pred_probs),
                'odds_home': match.get('B365H'),
                'odds_draw': match.get('B365D'),
                'odds_away': match.get('B365A')
            }
            results.append(result)
        
        return True, results
        
    except Exception as e:
        print(f"[ERROR] Predictions failed: {e}")
        import traceback
        traceback.print_exc()
        return False, []

def display_and_save_predictions(results):
    """
    Affiche et sauvegarde les résultats de prédiction
    
    Args:
        results (list): Liste des résultats de prédiction
        
    Returns:
        str: Chemin du fichier sauvegardé
    """
    if not results:
        print("[INFO] Aucun résultat à afficher")
        return None
    
    # Afficher les résultats
    print("\n" + "="*80)
    print("PREDICTIONS DU JOUR".center(80))
    print("="*80)
    
    for result in results:
        print(f"\n🏆 {result['league']}: {result['home_team']} vs {result['away_team']}")
        print(f"   Prédiction: {result['prediction_label']} ({result['confidence']:.1%} confiance)")
        print(f"   Probabilités: Win {result['prob_win']:.1%} | Draw {result['prob_draw']:.1%} | Loss {result['prob_loss']:.1%}")
        if result['odds_home']:
            print(f"   Cotes: Home {result['odds_home']:.2f} | Draw {result['odds_draw']:.2f} | Away {result['odds_away']:.2f}")
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    output_dir = Path("Match of ze day")
    output_dir.mkdir(exist_ok=True)
    
    today_str = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f"predictions_{today_str}.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n[SUCCESS] Prédictions sauvegardées: {output_file}")
    return str(output_file)

def predict_matches(df_for_prediction):
    """
    Interface principale pour faire des prédictions
    
    Args:
        df_for_prediction (DataFrame): Données des matchs à prédire
        
    Returns:
        bool: Succès de l'opération
    """
    print(f"\n{'-'*50}")
    print(f"PREDICTIONS AVEC LE MODELE")
    print(f"{'-'*50}")
    
    if df_for_prediction is None or df_for_prediction.empty:
        print("[ERROR] Aucune donnée fournie pour les prédictions")
        return False
    
    # Faire les prédictions
    success, results = run_predictions(df_for_prediction)
    
    if not success or not results:
        print("[ERROR] Échec des prédictions")
        return False
    
    # Afficher et sauvegarder
    output_file = display_and_save_predictions(results)
    
    return output_file is not None

if __name__ == "__main__":
    print("Ce module est destiné à être importé, pas exécuté directement")
    print("Utilisez prediction_complete_pipeline.py pour lancer le pipeline complet")