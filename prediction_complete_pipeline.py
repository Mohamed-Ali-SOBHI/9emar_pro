#!/usr/bin/env python3
"""
Pipeline Complet de Prédiction - Football Match Prediction
1. Récupère les matchs du jour avec les cotes (odds api)
2. Si matchs trouvés, lance la récupération des statistiques (statistic_ze_year)
3. Lance le preprocessing des données (predict_preprocessing)  
4. Merge les données avec les cotes
5. Lance le modèle pour faire les prédictions
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
import traceback
import json
warnings.filterwarnings('ignore')

# Import des modules utils
sys.path.append('utils/predict')
sys.path.append('utils/train')

def print_section(title):
    """Affiche une section formatee"""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}")

def print_step(step_num, step_name):
    """Affiche une etape formatee"""
    print(f"\n{'-'*50}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'-'*50}")


def step1_get_todays_matches():
    """STEP 1: Récupérer les matchs du jour avec les cotes via odds API"""
    print_step(1, "RECUPERATION DES MATCHS DU JOUR (ODDS API)")
    
    try:
        # Lancer odds api
        print("[RUNNING] Executing odds API script...")
        result = subprocess.run([
            sys.executable, "utils/predict/odds api.py"
        ], cwd=".", capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"[ERROR] Odds API failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None, False
        
        print("[OK] Odds API executed successfully")
        print(result.stdout)  # Afficher les résultats
        
        # Chercher le fichier CSV généré
        match_dir = Path("Match of ze day")
        if not match_dir.exists():
            print("[ERROR] 'Match of ze day' directory not found")
            return None, False
        
        # Chercher le fichier le plus récent avec european_leagues_matches
        csv_files = list(match_dir.glob("european_leagues_matches_*.csv"))
        if not csv_files:
            print("[INFO] Aucun match trouvé pour aujourd'hui")
            return None, False
        
        # Prendre le fichier le plus récent
        latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Charger et vérifier le contenu
        df_matches = pd.read_csv(latest_file)
        if df_matches.empty:
            print("[INFO] Fichier CSV vide - aucun match aujourd'hui")
            return None, False
        
        print(f"[SUCCESS] {len(df_matches)} match(s) trouvé(s) pour aujourd'hui")
        print(f"[FILE] {latest_file}")
        
        # Afficher les matchs trouvés
        for _, match in df_matches.iterrows():
            print(f"  - {match['league']}: {match['home_team']} vs {match['away_team']}")
            if pd.notna(match.get('B365H')):
                print(f"    Cotes: H:{match['B365H']:.2f} D:{match['B365D']:.2f} A:{match['B365A']:.2f}")
        
        return df_matches, True
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Odds API timed out")
        return None, False
    except Exception as e:
        print(f"[ERROR] Failed to run odds API: {e}")
        traceback.print_exc()
        return None, False

def step2_get_current_statistics():
    """STEP 2: Récupérer les statistiques actuelles avec statistic_ze_year"""
    print_step(2, "RECUPERATION DES STATISTIQUES ACTUELLES")
    
    try:
        # Vérifier tous les fichiers (pas seulement 2025)
        data_dir = Path("Data")
        leagues = ['Bundesliga', 'EPL', 'La_liga', 'Ligue_1', 'Serie_A']
        total_files = 0
        
        for league in leagues:
            league_path = data_dir / league
            if league_path.exists():
                csv_files = list(league_path.glob("*.csv"))
                total_files += len(csv_files)
        
        print(f"[INFO] {total_files} fichiers de données trouvés")
        print("[RUNNING] Executing statistic_ze_year script (force run)...")
        
        # Lancer statistic_ze_year
        result = subprocess.run([
            sys.executable, "utils/predict/statistic_ze_year.py"
        ], cwd=".", capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            print(f"[ERROR] statistic_ze_year failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print("[OK] Statistics scraping completed")
        print(result.stdout)
        
        # Vérifier les nouvelles données
        new_total = 0
        for league in leagues:
            league_path = data_dir / league
            if league_path.exists():
                csv_files = list(league_path.glob("*.csv"))
                new_total += len(csv_files)
        
        print(f"[SUCCESS] {new_total} fichiers de statistiques disponibles")
        return True
        
    except subprocess.TimeoutExpired:
        print("[ERROR] Statistics scraping timed out")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run statistics scraper: {e}")
        traceback.print_exc()
        return False

def step3_preprocess_data():
    """STEP 3: Preprocessing des données avec predict_preprocessing"""
    print_step(3, "PREPROCESSING DES DONNEES")
    
    try:
        # Import des fonctions de preprocessing
        from predict_preprocessing import preprocess_data_memory_only
        
        print("[RUNNING] Processing data in memory...")
        
        # Preprocessing en mémoire
        df_processed = preprocess_data_memory_only()
        
        if df_processed is None or df_processed.empty:
            print("[ERROR] Preprocessing returned empty dataset")
            return None
        
        print(f"[SUCCESS] Preprocessing completed: {len(df_processed)} matches, {df_processed.shape[1]} features")
        
        return df_processed
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        traceback.print_exc()
        return None

def step4_merge_with_odds(df_processed, df_matches_today):
    """STEP 4: Merger les données preprocessed avec les cotes du jour"""
    print_step(4, "MERGE DONNEES AVEC COTES DU JOUR")
    
    try:
        print(f"[INFO] Données preprocessed: {len(df_processed)} lignes")
        print(f"[INFO] Matchs du jour: {len(df_matches_today)} lignes")
        
        # Créer un DataFrame avec les features nécessaires pour la prédiction
        # En utilisant les données du jour comme base
        prediction_data = []
        
        for _, match in df_matches_today.iterrows():
            home_team = match['home_team']
            away_team = match['away_team'] 
            league = match['league']
            
            print(f"[PROCESSING] {league}: {home_team} vs {away_team}")
            
            # Chercher les statistiques récentes pour chaque équipe dans les données preprocessed
            home_stats = df_processed[df_processed['team_name'].str.contains(home_team, case=False, na=False)]
            away_stats = df_processed[df_processed['team_name'].str.contains(away_team, case=False, na=False)]
            
            if home_stats.empty:
                print(f"  [WARNING] Pas de stats trouvées pour {home_team}")
            if away_stats.empty:
                print(f"  [WARNING] Pas de stats trouvées pour {away_team}")
            
            # Pour l'instant, créer une ligne de base avec les cotes
            match_features = {
                'team_name': home_team,
                'opponent_name': away_team,
                'league': league,
                'B365H': match.get('B365H'),
                'B365D': match.get('B365D'), 
                'B365A': match.get('B365A'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'year': 2025,
                'month': datetime.now().month
            }
            
            # Vérifier si on a les données nécessaires pour les deux équipes
            if home_stats.empty or away_stats.empty:
                print(f"  [SKIP] Données insuffisantes pour ce match - stats manquantes")
                continue
            
            # Utiliser seulement les vraies données des équipes
            latest_home = home_stats.iloc[-1]  # Stats les plus récentes de l'équipe domicile
            latest_away = away_stats.iloc[-1]  # Stats les plus récentes de l'équipe extérieure
            
            # Copier TOUTES les features de l'équipe domicile (team_*)
            for col in latest_home.index:
                if col.startswith('team_') or col in ['form_score_5', 'form_score_10', 'unbeaten_streak', 'winless_streak', 'current_streak']:
                    match_features[col] = latest_home[col]
                # Copier aussi les H2H features
                elif col.startswith('h2h_'):
                    match_features[col] = latest_home[col]
            
            # Copier les features de l'équipe extérieure (opponent_*)
            for col in latest_away.index:
                if col.startswith('team_'):
                    opponent_col = col.replace('team_', 'opponent_')
                    match_features[opponent_col] = latest_away[col]
                elif col in ['form_score_5', 'form_score_10', 'unbeaten_streak', 'winless_streak']:
                    opponent_col = f'opponent_{col}'
                    match_features[opponent_col] = latest_away[col]
            
            # Vérifier si on a assez de features essentielles
            essential_features = [
                'team_xG_last_5', 'team_xG_against_last_5',
                'opponent_xG_last_5', 'opponent_xG_against_last_5'
            ]
            
            missing_essential = [f for f in essential_features if f not in match_features]
            if missing_essential:
                print(f"  [SKIP] Features essentielles manquantes: {missing_essential}")
                continue
            
            print(f"  [OK] Données complètes trouvées pour ce match")
            prediction_data.append(match_features)
        
        df_for_prediction = pd.DataFrame(prediction_data)
        
        if df_for_prediction.empty:
            print(f"[ERROR] Aucun match avec données complètes trouvé")
            print(f"[INFO] Tous les matchs ont été skippés par manque de données")
            return None
        
        print(f"[SUCCESS] Données pour prédiction préparées: {len(df_for_prediction)} matches")
        return df_for_prediction
        
    except Exception as e:
        print(f"[ERROR] Merge with odds failed: {e}")
        traceback.print_exc()
        return None

def step5_enhanced_feature_engineering(df_merged):
    """STEP 5: Enhanced Feature Engineering après merge avec les cotes"""
    print_step(5, "ENHANCED FEATURE ENGINEERING")
    
    try:
        # Import des fonctions d'enhanced feature engineering
        from enhanced_feature_engineering import add_all_enhanced_features
        
        print("[RUNNING] Adding enhanced features...")
        
        # Ajouter les features avancées (nécessite les cotes B365H, B365D, B365A)
        df_enhanced = add_all_enhanced_features(df_merged)
        
        if df_enhanced is None or df_enhanced.empty:
            print("[ERROR] Enhanced feature engineering returned empty dataset")
            return None
        
        print(f"[SUCCESS] Enhanced features added: {len(df_enhanced)} matches, {df_enhanced.shape[1]} features")
        
        return df_enhanced
        
    except Exception as e:
        print(f"[ERROR] Enhanced feature engineering failed: {e}")
        traceback.print_exc()
        return None

def step6_run_predictions(df_for_prediction):
    """STEP 6: Lancer les prédictions avec le modèle et sauvegarder en CSV"""
    print_step(6, "PREDICTIONS AVEC LE MODELE")
    
    try:
        # Encoder les colonnes string avant la prédiction
        df_clean = df_for_prediction.copy()
        
        # Encoder h2h_last_result
        if 'h2h_last_result' in df_clean.columns:
            df_clean['h2h_last_result'] = df_clean['h2h_last_result'].map({
                'w': 1, 'd': 0, 'l': -1, 'N/A': 0
            }).fillna(0)
        
        # S'assurer que toutes les colonnes sont numériques sauf les colonnes importantes
        important_string_cols = ['team_name', 'opponent_name', 'league']
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' and col not in important_string_cols:
                print(f"[DEBUG] Converting string column: {col}")
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
        
        # Importer le module de prédiction fixe
        from predict_matches import predict_matches
        
        # Faire les prédictions (sauvegarde CSV automatique dans predict_matches)
        success = predict_matches(df_clean, df_for_prediction)
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Failed to run predictions: {e}")
        traceback.print_exc()
        return False


def main():
    """Pipeline principal de prédiction"""
    print_section("PIPELINE COMPLET DE PREDICTION")
    print(f"[START] Début du pipeline: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Vérification du modèle sera faite dans le module de prédiction
    
    success_steps = 0
    total_steps = 6
    
    try:
        # STEP 1: Récupérer les matchs du jour
        df_matches_today, step1_success = step1_get_todays_matches()
        if step1_success and df_matches_today is not None:
            success_steps += 1
        else:
            print("[STOP] Aucun match aujourd'hui - arrêt du pipeline")
            return False
        
        # STEP 2: Récupérer les statistiques actuelles
        if step2_get_current_statistics():
            success_steps += 1
        else:
            print("[WARNING] Step 2 failed, continuing with existing data...")
        
        # STEP 3: Preprocessing des données
        df_processed = step3_preprocess_data()
        if df_processed is not None:
            success_steps += 1
        else:
            print("[ERROR] Preprocessing failed - cannot continue")
            return False
        
        # STEP 4: Merger avec les cotes
        df_merged = step4_merge_with_odds(df_processed, df_matches_today)
        if df_merged is not None:
            success_steps += 1
        else:
            print("[ERROR] Merge failed - cannot continue")
            return False
        
        # STEP 5: Enhanced Feature Engineering
        df_for_prediction = step5_enhanced_feature_engineering(df_merged)
        if df_for_prediction is not None:
            success_steps += 1
        else:
            print("[ERROR] Enhanced feature engineering failed - cannot continue")
            return False
        
        # STEP 6: Faire les prédictions
        if step6_run_predictions(df_for_prediction):
            success_steps += 1
        else:
            print("[ERROR] Predictions failed")
            return False
        
        # Résumé final
        print_section("PIPELINE TERMINE AVEC SUCCES")
        print(f"[SUCCESS] {success_steps}/{total_steps} étapes réussies")
        print(f"[END] Pipeline terminé: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] Pipeline interrompu par l'utilisateur")
        return False
    except Exception as e:
        print(f"\n[ERROR] Erreur critique dans le pipeline: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)