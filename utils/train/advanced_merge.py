import pandas as pd
import glob
from tqdm import tqdm
from typing import Optional
import difflib

def fuzz_token_sort_ratio(s1: str, s2: str) -> int:
    """Implémentation simple du token sort ratio"""
    if not s1 or not s2:
        return 0
    tokens1 = sorted(s1.lower().split())
    tokens2 = sorted(s2.lower().split())
    return int(difflib.SequenceMatcher(None, ' '.join(tokens1), ' '.join(tokens2)).ratio() * 100)

def fuzz_partial_ratio(s1: str, s2: str) -> int:
    """Implémentation simple du partial ratio"""
    if not s1 or not s2:
        return 0
    s1, s2 = s1.lower(), s2.lower()
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    return int(max(difflib.SequenceMatcher(None, s1, s2[i:i+len(s1)]).ratio() for i in range(len(s2)-len(s1)+1)) * 100)

class fuzz:
    @staticmethod
    def token_sort_ratio(s1: str, s2: str) -> int:
        return fuzz_token_sort_ratio(s1, s2)
    
    @staticmethod
    def partial_ratio(s1: str, s2: str) -> int:
        return fuzz_partial_ratio(s1, s2)

def normalize_team_name(name: str) -> str:
    """
    Normalise le nom d'une équipe pour améliorer les correspondances.
    Retourne aussi le nom original pour le fallback.
    """
    if not name or pd.isna(name):
        return ""
        
    name = str(name).strip()
    
    specific_teams = {
        # Équipes françaises
        "Paris Saint Germain": "PSG",
        "Paris SG": "PSG",
        "Saint-Etienne": "St Etienne",
        "St-Etienne": "St Etienne",
        
        # Équipes espagnoles
        "Athletic Club": "Ath Bilbao",
        "Athletic Bilbao": "Ath Bilbao",
        "Celta Vigo": "Celta",
        "Real Betis": "Betis",
        "Rayo Vallecano": "Vallecano",
        "Real Valladolid": "Valladolid",
        "Deportivo La Coruna": "Deportivo",
        "Deportivo de La Coruña": "Deportivo",
        "Real Club Deportivo de La Coruña": "Deportivo",
        "SD Huesca": "Huesca",
        "FC Heidenheim": "Heidenheim",
        "Holstein Kiel": "Kiel",
        "St. Pauli": "St Pauli",
        
        # Équipes anglaises
        "Wolverhampton Wanderers": "Wolves",
        "Wolverhampton": "Wolves",
        "West Bromwich Albion": "West Brom",
        "West Bromwich": "West Brom",
        "Newcastle United": "Newcastle",
        "Norwich City": "Norwich",
        "Leicester City": "Leicester",
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Sheffield United": "Sheffield Utd",
        "Nottingham Forest": "Nott'm Forest",
        "Brighton & Hove Albion": "Brighton",
        "Crystal Palace": "Palace",
        
        # Équipes allemandes
        "Borussia MGladbach": "M'gladbach",
        "Monchengladbach": "M'gladbach",
        "Borussia M.Gladbach": "M'gladbach",
        "Borussia M.gladbach": "M'gladbach",
        "Borussia Monchengladbach": "M'gladbach",
        "RasenBallsport Leipzig": "RB Leipzig",
        "Red Bull Leipzig": "RB Leipzig",
        "Koln": "Cologne",
        "FC Koln": "Cologne",
        "FC Cologne": "Cologne",
        "1. FC Koln": "Cologne",
        "1. FC Cologne": "Cologne",
        "Bayern Munich": "Bayern",
        "Bayern Munchen": "Bayern",
        "FC Bayern": "Bayern",
        "Borussia Dortmund": "Dortmund",
        "Bor Dortmund": "Dortmund",
        "BVB": "Dortmund",
        "Hertha BSC": "Hertha",
        "Hertha BSC Berlin": "Hertha",
        "Hertha Berlin": "Hertha",
        "VfL Wolfsburg": "Wolfsburg",
        "TSG Hoffenheim": "Hoffenheim",
        "1899 Hoffenheim": "Hoffenheim",
        "FSV Mainz 05": "Mainz",
        "Mainz 05": "Mainz",
        "SC Freiburg": "Freiburg",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "Eintracht": "Ein Frankfurt",
        "Arminia Bielefeld": "Bielefeld",
        "Greuther Fuerth": "Fuerth",
        "Fortuna Duesseldorf": "Fortuna",
        
        # Équipes italiennes
        "Parma Calcio 1913": "Parma",
        "Parma Calcio": "Parma",
        "SPAL 2013": "SPAL",
        "Internazionale": "Inter",
        "Inter Milan": "Inter",
        "AC Milan": "Milan",
        "Associazione Calcio Milan": "Milan",
        "SSC Napoli": "Napoli",
        "Società Sportiva Calcio Napoli": "Napoli",
        "AS Roma": "Roma",
        "Associazione Sportiva Roma": "Roma",
        "SS Lazio": "Lazio",
        "Società Sportiva Lazio": "Lazio",
        "Bologna FC": "Bologna",
        "Bologna Football Club": "Bologna",
        "Udinese Calcio": "Udinese",
        "Torino FC": "Torino",
        "Torino Football Club": "Torino",
        "Atalanta BC": "Atalanta",
        "Atalanta Bergamo": "Atalanta",
        "Atalanta Bergamasca Calcio": "Atalanta",
        "AC Chievo Verona": "Chievo",
        "Chievo Verona": "Chievo",
        "Genoa CFC": "Genoa",
        "Genoa Cricket and Football Club": "Genoa",
        "ACF Fiorentina": "Fiorentina",
        "Cagliari Calcio": "Cagliari",
        "US Sassuolo": "Sassuolo",
        "Unione Sportiva Sassuolo Calcio": "Sassuolo",
        "Hellas Verona": "Verona",
        "Hellas Verona FC": "Verona",
        
        # Équipes espagnoles supplémentaires
        "Atlético Madrid": "Ath Madrid",
        "Atlético de Madrid": "Ath Madrid",
        "Club Atlético de Madrid": "Ath Madrid",
        "Atletico Madrid": "Ath Madrid",
        "Valencia CF": "Valencia",
        "Valencia Club de Fútbol": "Valencia",
        "FC Barcelona": "Barcelona",
        "Futbol Club Barcelona": "Barcelona",
        "Villarreal CF": "Villarreal",
        "Real Sociedad de Fútbol": "Real Sociedad",
        "RCD Espanyol": "Espanyol",
        "Real Club Deportivo Espanyol": "Espanyol",
        "UD Las Palmas": "Las Palmas",
        "CD Leganes": "Leganes",
        "Real Madrid CF": "Real Madrid",
        "Sevilla FC": "Sevilla",
        "Real Betis Balompie": "Betis",
        "CA Osasuna": "Osasuna",
        "Getafe CF": "Getafe",
        "CD Alaves": "Alaves",
        "RCD Mallorca": "Mallorca",
        "Girona FC": "Girona",
        
        # Équipes françaises supplémentaires
        "Lille OSC": "Lille",
        "LOSC Lille": "Lille",
        "SCO Angers": "Angers",
        "ES Troyes AC": "Troyes",
        "ESTAC Troyes": "Troyes",
        "Stade de Reims": "Reims",
        "AS Monaco": "Monaco",
        "Association Sportive de Monaco": "Monaco",
        "Olympique Lyonnais": "Lyon",
        "Olympique de Lyon": "Lyon",
        "Olympique de Marseille": "Marseille",
        "Stade Rennais FC": "Rennes",
        "Stade Rennais Football Club": "Rennes",
        "FC Nantes": "Nantes",
        "Montpellier HSC": "Montpellier",
        "OGC Nice": "Nice",
        "RC Lens": "Lens",
        "Stade Brestois": "Brest",
        "RC Strasbourg": "Strasbourg",
        "FC Metz": "Metz",
        "Clermont Foot": "Clermont",
        "AC Ajaccio": "Ajaccio",
        "AJ Auxerre": "Auxerre",
        "Le Havre AC": "Le Havre",
        "Toulouse FC": "Toulouse",
        "FC Lorient": "Lorient",
        "AS Saint-Etienne": "St Etienne",
        "Dijon FCO": "Dijon",
        "SM Caen": "Caen",
        "EA Guingamp": "Guingamp",
        "Girondins Bordeaux": "Bordeaux",
        "FC Girondins de Bordeaux": "Bordeaux",
        "Amiens SC": "Amiens",
        "Nimes Olympique": "Nimes",
        
        # Mappings supplémentaires basés sur l'analyse des équipes non trouvées
        # Équipes italiennes problématiques
        "Hellas Verona FC": "Verona",
        "Bologna FC 1909": "Bologna", 
        "Genoa CFC": "Genoa",
        "AC Fiorentina": "Fiorentina",
        "US Sassuolo Calcio": "Sassuolo",
        "FC Juventus": "Juventus",
        "Juventus FC": "Juventus",
        "Atalanta Bergamasca Calcio": "Atalanta",
        "UC Sampdoria": "Sampdoria",
        "Udinese Calcio": "Udinese",
        "SS Lazio": "Lazio",
        "S.S. Lazio": "Lazio",
        
        # Équipes espagnoles problématiques
        "Real Madrid Club de Fútbol": "Real Madrid",
        "Club de Fútbol Villarreal": "Villarreal",
        "Sociedad Deportiva Eibar": "Eibar",
        "Athletic Club Bilbao": "Ath Bilbao",
        
        # Équipes anglaises
        "Crystal Palace FC": "Crystal Palace",
        "West Bromwich Albion FC": "West Bromwich Albion", 
        "Stoke City FC": "Stoke",
        "Swansea City": "Swansea",
        "Bournemouth AFC": "Bournemouth",
        "Watford FC": "Watford",
        
        # Corrections de normalisation manquées
        "Chievo Verona": "Chievo",
        "Chievo": "Chievo",
        "Verona": "Verona", 
        "Hellas Verona": "Verona",
    }
    
    # D'abord vérifier si c'est une équipe spécifique
    if name in specific_teams:
        return specific_teams[name]
    
    # Nettoyer le nom de base
    cleaned_name = name
    
    # Remplacements sécurisés
    safe_replacements = [
        ("1.", ""),
        ("FC ", ""),
        (" FC", ""),
        ("CF ", ""),
        (" CF", ""),
        ("AC ", ""),
        (" AC", ""),
        ("SC ", ""),
        (" SC", ""),
        ("AS ", ""),
        (" AS", ""),
        ("US ", ""),
        (" US", ""),
        ("RC ", ""),
        (" RC", ""),
        ("CD ", ""),
        (" CD", ""),
        ("UD ", ""),
        (" UD", ""),
        ("Real ", ""),
        ("Club ", ""),
        ("Société ", ""),
        ("Sportiva ", ""),
        ("Calcio", ""),
        ("Football Club", ""),
        ("de Fútbol", ""),
        ("Balompie", ""),
        ("Olympique", "Oly"),
        ("Saint", "St"),
        ("Manchester", "Man"),
        ("United", "Utd"),
    ]
    
    for old, new in safe_replacements:
        cleaned_name = cleaned_name.replace(old, new)
    
    # Nettoyer les espaces multiples et caractères spéciaux
    import re
    cleaned_name = re.sub(r'[^\w\s\']', '', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    
    return cleaned_name if cleaned_name else name

def find_best_match_for_row(stats_row: pd.Series, odds_on_date: pd.DataFrame) -> Optional[pd.Series]:
    """
    Compare les noms des équipes entre les stats et les cotes avec plusieurs niveaux de matching.
    """
    team1_stats = normalize_team_name(stats_row['team_name'])
    team2_stats = normalize_team_name(stats_row['opponent_name'])
    team1_original = str(stats_row['team_name']).strip()
    team2_original = str(stats_row['opponent_name']).strip()
    
    best_match_score = 0
    best_match_row = None
    match_details = None
    
    for _, odds_row in odds_on_date.iterrows():
        team1_odds = normalize_team_name(odds_row['team_name'])
        team2_odds = normalize_team_name(odds_row['opponent_name'])
        team1_odds_original = str(odds_row['team_name']).strip()
        team2_odds_original = str(odds_row['opponent_name']).strip()
        
        # Niveau 1: Match exact avec noms normalisés
        if (team1_stats == team1_odds and team2_stats == team2_odds) or \
           (team1_stats == team2_odds and team2_stats == team1_odds):
            return odds_row
        
        # Niveau 2: Match avec noms originaux
        if (team1_original == team1_odds_original and team2_original == team2_odds_original) or \
           (team1_original == team2_odds_original and team2_original == team1_odds_original):
            return odds_row
        
        # Niveau 3: Match fuzzy avec différents algorithmes
        scores = []
        
        # Match direct normalisé
        score1_norm = fuzz.token_sort_ratio(team1_stats, team1_odds)
        score2_norm = fuzz.token_sort_ratio(team2_stats, team2_odds)
        scores.append(min(score1_norm, score2_norm))
        
        # Match inversé normalisé
        score3_norm = fuzz.token_sort_ratio(team1_stats, team2_odds)
        score4_norm = fuzz.token_sort_ratio(team2_stats, team1_odds)
        scores.append(min(score3_norm, score4_norm))
        
        # Match direct original
        score1_orig = fuzz.token_sort_ratio(team1_original, team1_odds_original)
        score2_orig = fuzz.token_sort_ratio(team2_original, team2_odds_original)
        scores.append(min(score1_orig, score2_orig))
        
        # Match inversé original
        score3_orig = fuzz.token_sort_ratio(team1_original, team2_odds_original)
        score4_orig = fuzz.token_sort_ratio(team2_original, team1_odds_original)
        scores.append(min(score3_orig, score4_orig))
        
        # Match avec ratio partiel pour gérer les noms courts
        partial_scores = []
        partial_scores.append(min(fuzz.partial_ratio(team1_stats, team1_odds), 
                                fuzz.partial_ratio(team2_stats, team2_odds)))
        partial_scores.append(min(fuzz.partial_ratio(team1_stats, team2_odds), 
                                fuzz.partial_ratio(team2_stats, team1_odds)))
        
        # Prendre le meilleur score
        match_strength = max(scores + partial_scores)
        
        if match_strength > best_match_score:
            best_match_score = match_strength
            best_match_row = odds_row
            match_details = {
                'normalized_direct': min(score1_norm, score2_norm),
                'normalized_inverse': min(score3_norm, score4_norm),
                'original_direct': min(score1_orig, score2_orig),
                'original_inverse': min(score3_orig, score4_orig),
                'partial_best': max(partial_scores),
                'teams_stats': f"{team1_stats} vs {team2_stats}",
                'teams_odds': f"{team1_odds} vs {team2_odds}"
            }
    
    # Seuils adaptatifs selon la qualité du match
    threshold = 50  # Seuil plus bas pour capturer plus de matches
    if best_match_score >= 90:
        threshold = 90
    elif best_match_score >= 80:
        threshold = 80
    elif best_match_score >= 70:
        threshold = 70
    elif best_match_score >= 60:
        threshold = 60
    
    return best_match_row if best_match_score >= threshold else None

def load_csv_files(path_pattern: str) -> pd.DataFrame:
    """
    Charge tous les fichiers CSV correspondants à un pattern donné.
    """
    files = glob.glob(path_pattern, recursive=True)
    dfs = []
    
    columns_to_keep = ['HomeTeam', 'AwayTeam', 'Date', 'B365H', 'B365D', 'B365A']
    
    for f in files:
        try:
            df = pd.read_csv(f, encoding='latin1', low_memory=False)
            # Sélectionner uniquement les colonnes nécessaires
            df = df[columns_to_keep]
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")
    
    if not dfs:
        print("Aucun fichier CSV trouvé avec le pattern spécifié")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)

def analyze_data_loss(stats_df: pd.DataFrame, odds_df: pd.DataFrame):
    """
    Analyse les causes potentielles de perte de données.
    """
    print("\n=== ANALYSE DES DONNÉES ===")
    
    # Analyse des dates
    stats_dates = set(stats_df['date'].unique())
    odds_dates = set(odds_df['date'].unique())
    
    print(f"Dates dans les stats: {len(stats_dates)} ({min(stats_dates)} à {max(stats_dates)})")
    print(f"Dates dans les cotes: {len(odds_dates)} ({min(odds_dates)} à {max(odds_dates)})")
    print(f"Dates communes: {len(stats_dates & odds_dates)}")
    print(f"Dates uniquement dans stats: {len(stats_dates - odds_dates)}")
    print(f"Dates uniquement dans cotes: {len(odds_dates - stats_dates)}")
    
    # Analyse des équipes
    stats_teams = set(stats_df['team_name'].unique()) | set(stats_df['opponent_name'].unique())
    odds_teams = set(odds_df['team_name'].unique()) | set(odds_df['opponent_name'].unique())
    
    print(f"\nÉquipes dans les stats: {len(stats_teams)}")
    print(f"Équipes dans les cotes: {len(odds_teams)}")
    
    # Équipes normalisées
    stats_teams_norm = {normalize_team_name(team) for team in stats_teams}
    odds_teams_norm = {normalize_team_name(team) for team in odds_teams}
    
    print(f"Équipes normalisées communes: {len(stats_teams_norm & odds_teams_norm)}")
    print(f"Équipes uniquement dans stats (normalisées): {len(stats_teams_norm - odds_teams_norm)}")
    print(f"Équipes uniquement dans cotes (normalisées): {len(odds_teams_norm - stats_teams_norm)}")

def load_odds_data():
    """Charge toutes les données de cotes et les retourne en DataFrame"""
    print("Loading odds data...")
    odds_df = load_csv_files("Data/odds/**/*.csv")
    if odds_df.empty:
        print("Aucune donnée de cotes trouvée")
        return pd.DataFrame()
    
    # Renommage des colonnes pour uniformiser
    odds_df.rename(columns={
        'HomeTeam': 'team_name', 
        'AwayTeam': 'opponent_name',
        'Date': 'date'
    }, inplace=True)
    
    # Conversion des dates pour les cotes
    odds_df['date'] = pd.to_datetime(odds_df['date'], dayfirst=True, errors='coerce').dt.date
    odds_df.dropna(subset=['date'], inplace=True)
    
    print(f"Odds data loaded: {len(odds_df)} records")
    return odds_df

def merge_with_odds(stats_df, odds_df):
    """
    Version optimisée du merge qui prend les DataFrames en entrée
    et retourne le DataFrame fusionné sans sauvegarde
    """
    if stats_df.empty or odds_df.empty:
        print("Error: Empty DataFrame provided")
        return pd.DataFrame()
    
    print("Starting optimized merge with odds...")
    
    # Conversion des dates pour les stats
    stats_df = stats_df.copy()
    stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce').dt.date
    stats_df.dropna(subset=['date'], inplace=True)
    
    # Analyse préliminaire
    analyze_data_loss(stats_df, odds_df)
    
    # Groupement des cotes par date pour optimiser les recherches
    odds_by_date = {date: group for date, group in odds_df.groupby('date')}
    
    matched_rows = []
    unmatched_teams = {}
    unmatched_details = []
    match_quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    print("Processing matches...")
    for _, row in tqdm(stats_df.iterrows(), total=len(stats_df), desc="Merging with odds"):
        match_date = row['date']
        
        if match_date not in odds_by_date:
            continue
        
        odds_on_date = odds_by_date[match_date]
        best_match = find_best_match_for_row(row, odds_on_date)
        
        if best_match is not None:
            # Combine row and best_match
            combined_row = {**row.to_dict(), **best_match.to_dict()}
            matched_rows.append(combined_row)
            
            # Calculate match quality
            team1_stats = normalize_team_name(row['team_name'])
            team2_stats = normalize_team_name(row['opponent_name'])
            team1_odds = normalize_team_name(best_match['team_name'])
            team2_odds = normalize_team_name(best_match['opponent_name'])
            
            score = max(
                min(fuzz.token_sort_ratio(team1_stats, team1_odds), fuzz.token_sort_ratio(team2_stats, team2_odds)),
                min(fuzz.token_sort_ratio(team1_stats, team2_odds), fuzz.token_sort_ratio(team2_stats, team1_odds))
            )
            
            if score >= 90:
                match_quality_stats["excellent"] += 1
            elif score >= 80:
                match_quality_stats["good"] += 1
            elif score >= 70:
                match_quality_stats["fair"] += 1
            else:
                match_quality_stats["poor"] += 1
        else:
            team_pair = f"{row['team_name']} vs {row['opponent_name']}"
            unmatched_teams[team_pair] = unmatched_teams.get(team_pair, 0) + 1
            
            unmatched_details.append({
                'date': match_date,
                'team1': row['team_name'],
                'team2': row['opponent_name'],
                'team1_norm': normalize_team_name(row['team_name']),
                'team2_norm': normalize_team_name(row['opponent_name'])
            })
    
    if matched_rows:
        merged_df = pd.DataFrame(matched_rows)
        
        print(f"\n=== RÉSULTATS DU MERGE OPTIMISÉ ===")
        print(f"Matches avec stats: {len(stats_df)}")
        print(f"Matches avec cotes trouvées: {len(merged_df)}")
        print(f"Taux de succès: {len(merged_df)/len(stats_df)*100:.1f}%")
        print(f"\nQualité des matches:")
        for quality, count in match_quality_stats.items():
            print(f"  {quality}: {count} ({count/len(merged_df)*100:.1f}%)")
        
        return merged_df
    else:
        print("Aucun match trouvé lors du merge!")
        return pd.DataFrame()

def main():
    # Chargement des données de stats
    try:
        stats_df = pd.read_csv("./preprocessed_data.csv")
        print(f"Stats chargées: {len(stats_df)} lignes")
        
        unmatched_teams = {}
        unmatched_details = []
        match_quality_stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
        
    except FileNotFoundError:
        print("Fichier preprocessed_data.csv non trouvé")
        return
    
    # Conversion des dates pour les stats
    stats_df['date'] = pd.to_datetime(stats_df['date'], errors='coerce').dt.date
    stats_df.dropna(subset=['date'], inplace=True)
    
    # Chargement des données de cotes
    odds_df = load_csv_files("Data/odds/**/*.csv")
    if odds_df.empty:
        print("Aucune donnée de cotes trouvée")
        return
    
    # Renommage des colonnes pour uniformiser
    odds_df.rename(columns={
        'HomeTeam': 'team_name', 
        'AwayTeam': 'opponent_name',
        'Date': 'date'
    }, inplace=True)
    
    # Conversion des dates pour les cotes
    odds_df['date'] = pd.to_datetime(odds_df['date'], dayfirst=True, errors='coerce').dt.date
    odds_df.dropna(subset=['date'], inplace=True)
    
    # Analyse préliminaire
    analyze_data_loss(stats_df, odds_df)
    
    # Groupement des cotes par date pour optimiser les recherches
    odds_by_date = {date: group for date, group in odds_df.groupby('date')}
    
    matched_rows = []
    no_date_matches = 0
    
    print(f"\n=== DÉBUT DU MATCHING ===")
    
    for _, stats_row in tqdm(stats_df.iterrows(), total=stats_df.shape[0], desc="Matching games"):
        odds_on_date = odds_by_date.get(stats_row['date'])
        
        if odds_on_date is None:
            no_date_matches += 1
            continue
            
        best_match = find_best_match_for_row(stats_row, odds_on_date)
        
        if best_match is not None:
            combined_row = {**stats_row.to_dict(), **best_match.to_dict()}
            matched_rows.append(combined_row)
            
            # Calculer la qualité du match pour les statistiques
            team1_stats = normalize_team_name(stats_row['team_name'])
            team2_stats = normalize_team_name(stats_row['opponent_name'])
            team1_odds = normalize_team_name(best_match['team_name'])
            team2_odds = normalize_team_name(best_match['opponent_name'])
            
            score = max(
                min(fuzz.token_sort_ratio(team1_stats, team1_odds), fuzz.token_sort_ratio(team2_stats, team2_odds)),
                min(fuzz.token_sort_ratio(team1_stats, team2_odds), fuzz.token_sort_ratio(team2_stats, team1_odds))
            )
            
            if score >= 90:
                match_quality_stats["excellent"] += 1
            elif score >= 80:
                match_quality_stats["good"] += 1
            elif score >= 70:
                match_quality_stats["fair"] += 1
            else:
                match_quality_stats["poor"] += 1
        else:
            # Enregistrer les détails des matches non trouvés
            team_key = f"{stats_row['team_name']} vs {stats_row['opponent_name']}"
            if team_key not in unmatched_teams:
                unmatched_teams[team_key] = 0
                unmatched_details.append({
                    'date': stats_row['date'],
                    'team1': stats_row['team_name'],
                    'team2': stats_row['opponent_name'],
                    'team1_norm': normalize_team_name(stats_row['team_name']),
                    'team2_norm': normalize_team_name(stats_row['opponent_name']),
                    'available_matches': len(odds_on_date)
                })
            unmatched_teams[team_key] += 1
    
    print(f"\n=== RÉSULTATS DU MATCHING ===")
    print(f"Total lignes stats: {len(stats_df)}")
    print(f"Pas de cotes pour la date: {no_date_matches}")
    print(f"Matches trouvés: {len(matched_rows)}")
    print(f"Matches non trouvés: {len(stats_df) - len(matched_rows) - no_date_matches}")
    print(f"Taux de succès (avec cotes disponibles): {len(matched_rows)/(len(stats_df)-no_date_matches)*100:.1f}%")
    print(f"Taux de succès global: {len(matched_rows)/len(stats_df)*100:.1f}%")
    
    print(f"\n=== QUALITÉ DES MATCHES ===")
    for quality, count in match_quality_stats.items():
        if len(matched_rows) > 0:
            print(f"{quality.capitalize()}: {count} ({count/len(matched_rows)*100:.1f}%)")
    
    if matched_rows:
        final_df = pd.DataFrame(matched_rows)
            
        # Suppression des doublons
        before_dedup = len(final_df)
        final_df = final_df.drop_duplicates(subset=['team_name', 'opponent_name', 'date']).reset_index(drop=True)
        print(f"\nDoublons supprimés: {before_dedup - len(final_df)}")
        print(f"Lignes finales: {len(final_df)}")
        
        final_df.to_csv("preprocessed_data_with_odds.csv", index=False)
        print(f"\nDonnees sauvegardees dans preprocessed_data_with_odds.csv")
        
        # Afficher les équipes qui n'ont pas trouvé de correspondance
        print(f"\n=== TOP 20 MATCHES NON TROUVÉS ===")
        sorted_unmatched = sorted(unmatched_teams.items(), key=lambda x: x[1], reverse=True)
        for team_match, count in sorted_unmatched[:20]:
            print(f"{team_match}: {count} occurrences")
        
        # Afficher quelques exemples de détails avec les équipes disponibles
        print(f"\n=== EXEMPLES DE MATCHES NON TROUVÉS (avec équipes disponibles) ===")
        for i, detail in enumerate(unmatched_details[:3]):
            print(f"\n--- Exemple {i+1} ---")
            print(f"Date: {detail['date']}")
            print(f"Stats teams: {detail['team1']} vs {detail['team2']}")
            print(f"Normalized: {detail['team1_norm']} vs {detail['team2_norm']}")
            
            # Afficher les équipes disponibles dans les cotes pour cette date
            date_odds = odds_by_date.get(detail['date'])
            if date_odds is not None:
                print(f"Available odds matches ({len(date_odds)}):")
                for _, odds_row in date_odds.iterrows():
                    odds_team1 = normalize_team_name(odds_row['team_name'])
                    odds_team2 = normalize_team_name(odds_row['opponent_name'])
                    print(f"  {odds_row['team_name']} vs {odds_row['opponent_name']}")
                    print(f"    -> {odds_team1} vs {odds_team2}")
                    
                    # Calculer les scores de similarité
                    score1 = fuzz.token_sort_ratio(detail['team1_norm'], odds_team1)
                    score2 = fuzz.token_sort_ratio(detail['team2_norm'], odds_team2)
                    score3 = fuzz.token_sort_ratio(detail['team1_norm'], odds_team2)
                    score4 = fuzz.token_sort_ratio(detail['team2_norm'], odds_team1)
                    
                    best_score = max(min(score1, score2), min(score3, score4))
                    print(f"    -> Best match score: {best_score}")
            print()
        
        # Analyser les équipes les plus problématiques
        print(f"\n=== ANALYSE DES ÉQUIPES PROBLÉMATIQUES ===")
        
        # Compter les équipes individuelles non matchées
        individual_teams = {}
        for detail in unmatched_details:
            team1_norm = detail['team1_norm']
            team2_norm = detail['team2_norm']
            if team1_norm not in individual_teams:
                individual_teams[team1_norm] = {'original': detail['team1'], 'count': 0}
            if team2_norm not in individual_teams:
                individual_teams[team2_norm] = {'original': detail['team2'], 'count': 0}
            individual_teams[team1_norm]['count'] += 1
            individual_teams[team2_norm]['count'] += 1
        
        # Trier par fréquence
        sorted_teams = sorted(individual_teams.items(), key=lambda x: x[1]['count'], reverse=True)
        
        print("Top 15 équipes avec le plus de matches non trouvés:")
        for team_norm, info in sorted_teams[:15]:
            print(f"  {info['original']} -> {team_norm}: {info['count']} occurrences")
            
            # Chercher des équipes similaires dans les cotes
            similar_odds_teams = []
            for odds_team in set(odds_df['team_name'].unique()) | set(odds_df['opponent_name'].unique()):
                odds_team_norm = normalize_team_name(odds_team)
                score = fuzz.token_sort_ratio(team_norm, odds_team_norm)
                if 40 <= score < 60:  # Équipes potentiellement similaires
                    similar_odds_teams.append((odds_team, odds_team_norm, score))
            
            if similar_odds_teams:
                similar_odds_teams.sort(key=lambda x: x[2], reverse=True)
                print(f"    Équipes similaires dans les cotes:")
                for orig, norm, score in similar_odds_teams[:3]:
                    print(f"      {orig} -> {norm} (score: {score})")
            print()
    else:
        print("Aucun match trouve!")

if __name__ == "__main__":
    main()