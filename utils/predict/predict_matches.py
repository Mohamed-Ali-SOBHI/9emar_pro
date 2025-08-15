#!/usr/bin/env python3
"""
Module de prédiction simple sans les problèmes de types
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

def predict_matches(df_for_prediction, df_original=None):
    """
    Fait les prédictions pour les matchs fournis
    Version simple qui gère les problèmes de types
    
    Args:
        df_for_prediction: DataFrame avec données nettoyées pour le modèle
        df_original: DataFrame original avec les noms d'équipes (optionnel)
    """
    try:
        # Vérifier si le modèle existe
        model_dir = Path("models/optimal_model")
        required_files = [
            "trained_model_realistic.pkl",
            "scaler_realistic.pkl", 
            "features_realistic.json"
        ]
        
        for file in required_files:
            if not (model_dir / file).exists():
                print(f"[ERROR] Fichier manquant: {file}")
                return False
        
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
            return False
        
        # Sélectionner seulement les features du modèle et s'assurer qu'elles sont numériques
        X = df_for_prediction[model_features].copy()
        
        # Convertir toutes les colonnes en float
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)
        
        print(f"[DEBUG] Shape des données: {X.shape}")
        print(f"[DEBUG] Toutes les données sont numériques: {X.dtypes.value_counts()}")
        
        # Scaler les données
        X_scaled = scaler.transform(X)
        
        # Faire les prédictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Préparer les résultats
        results = []
        # Utiliser les vraies classes du modèle dans l'ordre alphabétique
        model_classes = model.classes_  # ['d', 'l', 'w']
        class_names_mapping = {'d': 'Draw', 'l': 'Loss', 'w': 'Win'}
        
        for i, (_, match) in enumerate(df_for_prediction.iterrows()):
            # Récupérer la prédiction du modèle (string)
            model_prediction = predictions[i]  # 'd', 'l', ou 'w'
            pred_probs = probabilities[i] if len(probabilities) > i else [0.33, 0.33, 0.34]
            
            # Trouver la classe avec la plus haute probabilité
            max_prob_index = int(np.argmax(pred_probs))
            max_prob_class = model_classes[max_prob_index]  # 'd', 'l', ou 'w'
            
            # Utiliser la classe avec la plus haute probabilité pour la prédiction
            prediction_class = max_prob_class
            
            # Récupérer les vrais noms depuis les données originales
            if df_original is not None and i < len(df_original):
                original_match = df_original.iloc[i]
                home_team = str(original_match.get('team_name', 'Home Team'))
                away_team = str(original_match.get('opponent_name', 'Away Team'))  
                league = str(original_match.get('league', 'League'))
            else:
                # Fallback: essayer depuis match ou utiliser des noms génériques
                home_team = str(match.get('team_name', 'Home Team'))
                away_team = str(match.get('opponent_name', 'Away Team'))
                league = str(match.get('league', 'League'))
                
                # Si les noms ont été convertis en numérique, utiliser des noms génériques
                if home_team in ['0.0', 'nan', 'Unknown']:
                    home_team = 'Home Team'
                if away_team in ['0.0', 'nan', 'Unknown']:
                    away_team = 'Away Team' 
                if league in ['0.0', 'nan', 'Unknown']:
                    league = 'League'
            
            if prediction_class == 'l':  # Loss (défaite domicile)
                predicted_winner = away_team
                prediction_text = f"{away_team} (Away Win)"
            elif prediction_class == 'd':  # Draw
                predicted_winner = "Draw"
                prediction_text = "Draw"
            else:  # 'w' = Win (victoire domicile)
                predicted_winner = home_team
                prediction_text = f"{home_team} (Home Win)"
            
            result = {
                'league': league,
                'home_team': home_team,
                'away_team': away_team,
                'prediction': predicted_winner,
                'prediction_label': class_names_mapping[prediction_class],  # Utiliser la vraie prédiction
                'prediction_text': prediction_text,
                # Probabilités dans l'ordre alphabétique: ['d', 'l', 'w'] = [Draw, Loss, Win]
                'prob_loss': float(pred_probs[1]) if len(pred_probs) > 1 else 0.33,  # index 1 = 'l'
                'prob_draw': float(pred_probs[0]) if len(pred_probs) > 0 else 0.33,  # index 0 = 'd'
                'prob_win': float(pred_probs[2]) if len(pred_probs) > 2 else 0.34,   # index 2 = 'w'
                'confidence': float(max(pred_probs)) if len(pred_probs) > 0 else 0.34,
                'odds_home': float(match.get('B365H', 0)),
                'odds_draw': float(match.get('B365D', 0)),
                'odds_away': float(match.get('B365A', 0))
            }
            results.append(result)
        
        # Afficher les résultats
        print(f"\n{'='*80}")
        print(f"{'PREDICTIONS FOOTBALL':^80}")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\nMatch: {result['home_team']} vs {result['away_team']}")
            print(f"Ligue: {result['league']}")
            print(f"Prédiction: {result['prediction_text']}")
            print(f"Confiance: {result['confidence']:.1%}")
            print(f"Probabilités:")
            print(f"  - Défaite domicile: {result['prob_loss']:.1%}")
            print(f"  - Match nul: {result['prob_draw']:.1%}")
            print(f"  - Victoire domicile: {result['prob_win']:.1%}")
            print(f"Cotes: H:{result['odds_home']:.2f} D:{result['odds_draw']:.2f} A:{result['odds_away']:.2f}")
        
        print(f"\n{'='*80}")
        print(f"[SUCCESS] {len(results)} prédiction(s) réalisée(s) avec succès")
        
        # Sauvegarder les prédictions automatiquement
        save_predictions_to_csv(results)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Erreur lors des prédictions: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_predictions_to_csv(results):
    """Sauvegarder les prédictions dans un fichier CSV"""
    try:
        # Créer le répertoire des résultats s'il n'existe pas
        results_dir = Path("predictions_results")
        results_dir.mkdir(exist_ok=True)
        
        # Nom du fichier avec timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = results_dir / f"predictions_{timestamp}.csv"
        
        # Préparer les données pour CSV
        csv_data = []
        for result in results:
            csv_row = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'league': result['league'],
                'home_team': result['home_team'],
                'away_team': result['away_team'],
                'prediction': result['prediction'],  # Nom de l'équipe gagnante ou "Draw"
                'prediction_detail': result['prediction_text'],  # Détail complet
                'confidence': f"{result['confidence']:.3f}",
                'prob_loss': f"{result['prob_loss']:.3f}",
                'prob_draw': f"{result['prob_draw']:.3f}",
                'prob_win': f"{result['prob_win']:.3f}",
                'odds_home': f"{result['odds_home']:.2f}",
                'odds_draw': f"{result['odds_draw']:.2f}",
                'odds_away': f"{result['odds_away']:.2f}"
            }
            csv_data.append(csv_row)
        
        # Créer le DataFrame et sauvegarder
        df_results = pd.DataFrame(csv_data)
        df_results.to_csv(csv_filename, index=False, encoding='utf-8')
        
        print(f"[SUCCESS] Prédictions sauvegardées: {csv_filename}")
        print(f"[INFO] {len(csv_data)} prédiction(s) enregistrée(s)")
        
        return str(csv_filename)
        
    except Exception as e:
        print(f"[ERROR] Impossible de sauvegarder les prédictions: {e}")
        return None