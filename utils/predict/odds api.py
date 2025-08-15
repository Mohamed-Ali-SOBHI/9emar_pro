import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time

class EuropeanLeaguesScrapper:
    def __init__(self, api_key=None):
        """
        Initialize API Football scrapper
        
        API Football offers free tier with 100 requests/day
        Sign up at: https://www.api-football.com/
        """
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'X-RapidAPI-Host': 'v3.football.api-sports.io',
            'X-RapidAPI-Key': api_key if api_key else 'be30eda384115ceb82c21273eced6248'
        }
        
        # Season (current season)
        # Août 2025 = saison 2024-2025 se terminant
        current_month = datetime.now().month
        current_year = datetime.now().year
        if current_month >= 8:  # Août et après = nouvelle saison
            self.season = current_year
        else:  # Janvier-Juillet = saison précédente
            self.season = current_year - 1
        
        # IDs des ligues européennes + Community Shield
        self.target_leagues = {
            'EPL': 39,              # Premier League
            'La_liga': 140,         # La Liga  
            'Bundesliga': 78,       # Bundesliga
            'Serie_A': 135,         # Serie A
            'Ligue_1': 61,          # Ligue 1
            'Community_Shield': 528 # Community Shield
        }
        
        # Noms alternatifs pour reconnaissance
        self.league_names = {
            'Premier League': 'EPL',
            'La Liga': 'La_liga', 
            'Bundesliga': 'Bundesliga',
            'Serie A': 'Serie_A',
            'Ligue 1': 'Ligue_1',
            'Community Shield': 'Community_Shield'
        }

    def test_api_connection(self):
        """Test API connection and show remaining quota"""
        try:
            url = f"{self.base_url}/status"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                data = response.json()
                quota = data.get('response', {})
                print(f"API Status: Connected")
                print(f"Requests used today: {quota.get('requests', {}).get('used', 'N/A')}")
                print(f"Requests limit: {quota.get('requests', {}).get('limit_day', 'N/A')}")
                return True
            else:
                print(f"API Error: {response.status_code}")
                print("Please check your API key")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_fixtures_by_date(self, date, league_id=None):
        """Get fixtures for a specific date and league"""
        try:
            url = f"{self.base_url}/fixtures"
            params = {'date': date}
            
            if league_id:
                # Essayer d'abord avec la saison actuelle
                params['league'] = league_id
                params['season'] = self.season
                
                response = requests.get(url, headers=self.headers, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    fixtures = data.get('response', [])
                    if fixtures:
                        return fixtures
                    
                    # Si aucun match trouvé, essayer avec la saison précédente
                    print(f"   Aucun match en saison {self.season}, essai saison {self.season-1}")
                    params['season'] = self.season - 1
                    response = requests.get(url, headers=self.headers, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        return data.get('response', [])
            else:
                response = requests.get(url, headers=self.headers, params=params)
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', [])
            
            print(f"Error fetching fixtures: {response.status_code}")
            return []
                
        except Exception as e:
            print(f"Error in get_fixtures_by_date: {e}")
            return []

    def get_odds_for_fixture(self, fixture_id):
        """Get odds for a specific fixture"""
        try:
            url = f"{self.base_url}/odds"
            params = {
                'fixture': fixture_id,
                'bet': 1  # Match Winner (1X2)
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('response', [])
            else:
                print(f"Error fetching odds: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in get_odds_for_fixture: {e}")
            return []

    def extract_match_data(self, fixture, odds_data=None):
        """Extract match data from API response"""
        try:
            # Basic match info
            home_team = fixture['teams']['home']['name']
            away_team = fixture['teams']['away']['name']
            league_name = fixture['league']['name']
            match_date = fixture['fixture']['date']
            fixture_id = fixture['fixture']['id']
            
            # Convert league name to our standard format
            std_league = self.get_standard_league_name(league_name)
            
            # Extract odds as separate B365 columns
            B365H = None  # Home odds
            B365D = None  # Draw odds
            B365A = None  # Away odds
            
            if odds_data:
                for odds_info in odds_data:
                    bookmakers = odds_info.get('bookmakers', [])
                    # Look for Bet365 or use first available bookmaker
                    preferred_bookmaker = None
                    fallback_bookmaker = None
                    
                    for bookmaker in bookmakers:
                        bookmaker_name = bookmaker.get('name', '').lower()
                        if 'bet365' in bookmaker_name or '365' in bookmaker_name:
                            preferred_bookmaker = bookmaker
                            break
                        elif fallback_bookmaker is None:
                            fallback_bookmaker = bookmaker
                    
                    # Use preferred bookmaker or fallback
                    selected_bookmaker = preferred_bookmaker or fallback_bookmaker
                    
                    if selected_bookmaker:
                        bets = selected_bookmaker.get('bets', [])
                        for bet in bets:
                            if bet.get('name') == 'Match Winner':
                                values = bet.get('values', [])
                                # Extract Home, Draw, Away odds
                                for value in values:
                                    outcome = value.get('value', '')
                                    odd = value.get('odd', '')
                                    
                                    if outcome == 'Home' or outcome == '1':
                                        B365H = float(odd) if odd else None
                                    elif outcome == 'Draw' or outcome == 'X':
                                        B365D = float(odd) if odd else None
                                    elif outcome == 'Away' or outcome == '2':
                                        B365A = float(odd) if odd else None
                                break
                        if B365H or B365D or B365A:
                            break
            
            return {
                'league': std_league,
                'home_team': home_team,
                'away_team': away_team,
                'B365H': B365H,
                'B365D': B365D,
                'B365A': B365A,
                'match_time': match_date,
                'fixture_id': fixture_id,
                'date': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error extracting match data: {e}")
            return None

    def get_standard_league_name(self, api_league_name):
        """Convert API league name to our standard format"""
        league_mapping = {
            'Premier League': 'EPL',
            'La Liga': 'La_liga',
            'Bundesliga': 'Bundesliga',
            'Serie A': 'Serie_A',
            'Ligue 1': 'Ligue_1',
            'Community Shield': 'Community_Shield'
        }
        
        return league_mapping.get(api_league_name, api_league_name)

    def get_matches_for_date_range(self, start_date, days_ahead=3):
        """Récupérer les matchs sur une période donnée"""
        print(f"=== RECHERCHE MATCHS LIGUES EUROPEENNES ===")
        print(f"Periode: {start_date} (+{days_ahead} jours)")
        print(f"Saison: {self.season}")
        print(f"Ligues: EPL, La Liga, Bundesliga, Serie A, Ligue 1, Community Shield")
        
        if not self.test_api_connection():
            return []
        
        all_matches = []
        
        # Pour chaque jour de la période
        for day in range(days_ahead + 1):
            current_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=day)).strftime('%Y-%m-%d')
            print(f"\n--- Recherche pour le {current_date} ---")
            
            day_matches = self.get_matches_for_single_date(current_date)
            all_matches.extend(day_matches)
            
            if day_matches:
                print(f"   {len(day_matches)} match(s) trouve(s)")
            else:
                print(f"   Aucun match trouve")
        
        return all_matches

    def get_matches_for_single_date(self, date):
        """Récupérer les matchs pour une date spécifique"""
        matches = []
        
        # Méthode 1: Recherche directe par ID de ligue
        for league_name, league_id in self.target_leagues.items():
            try:
                fixtures = self.get_fixtures_by_date(date, league_id)
                if fixtures:
                    print(f"   {league_name}: {len(fixtures)} match(s)")
                    for fixture in fixtures:
                        match_data = self.process_fixture(fixture)
                        if match_data:
                            matches.append(match_data)
            except Exception as e:
                print(f"   Erreur {league_name}: {e}")
                continue
        
        # Méthode 2: Recherche générale si rien trouvé
        if not matches:
            print(f"   Recherche generale...")
            all_fixtures = self.get_fixtures_by_date(date)
            
            for fixture in all_fixtures:
                # Vérifier si c'est une ligue cible (ligue + pays)
                if self.is_target_league(fixture):
                    match_data = self.process_fixture(fixture)
                    if match_data:
                        matches.append(match_data)
                        std_name = self.get_standard_name(fixture)
                        country = fixture['league'].get('country', '')
                        print(f"   Trouve {std_name} ({country}): {fixture['teams']['home']['name']} vs {fixture['teams']['away']['name']}")
        
        return matches

    def is_target_league(self, fixture):
        """Vérifier si la ligue ET le pays correspondent aux vraies ligues européennes"""
        league_name = fixture['league']['name']
        country = fixture['league'].get('country', '')
        
        # Correspondances exactes ligue + pays
        target_combinations = {
            ('Premier League', 'England'): 'EPL',
            ('La Liga', 'Spain'): 'La_liga',
            ('Bundesliga', 'Germany'): 'Bundesliga', 
            ('Serie A', 'Italy'): 'Serie_A',
            ('Ligue 1', 'France'): 'Ligue_1',
            ('Community Shield', 'England'): 'Community_Shield'
        }
        
        return (league_name, country) in target_combinations

    def get_standard_name(self, fixture):
        """Convertir nom API vers nom standard en vérifiant le pays"""
        league_name = fixture['league']['name']
        country = fixture['league'].get('country', '')
        
        target_combinations = {
            ('Premier League', 'England'): 'EPL',
            ('La Liga', 'Spain'): 'La_liga',
            ('Bundesliga', 'Germany'): 'Bundesliga',
            ('Serie A', 'Italy'): 'Serie_A', 
            ('Ligue 1', 'France'): 'Ligue_1',
            ('Community Shield', 'England'): 'Community_Shield'
        }
        
        return target_combinations.get((league_name, country), league_name)

    def process_fixture(self, fixture):
        """Traiter un fixture et récupérer les cotes"""
        try:
            home = fixture['teams']['home']['name']
            away = fixture['teams']['away']['name']
            league = fixture['league']['name']
            status = fixture['fixture']['status']['long']
            fixture_id = fixture['fixture']['id']
            
            # Forcer le bon nom de ligue
            std_league = self.get_standard_name(fixture)
            
            # Essayer de récupérer les cotes
            match_data = None
            try:
                odds_data = self.get_odds_for_fixture(fixture_id)
                if odds_data:
                    match_data = self.extract_match_data(fixture, odds_data)
                    if match_data:
                        match_data['league'] = std_league  # Forcer le nom standard
                    print(f"      -> {home} vs {away} (avec cotes)")
                else:
                    match_data = self.extract_match_data(fixture)
                    if match_data:
                        match_data['league'] = std_league  # Forcer le nom standard
                    print(f"      -> {home} vs {away} (sans cotes)")
            except:
                match_data = self.extract_match_data(fixture)
                if match_data:
                    match_data['league'] = std_league  # Forcer le nom standard
                print(f"      -> {home} vs {away} (erreur cotes)")
            
            return match_data
            
        except Exception as e:
            print(f"      Erreur traitement fixture: {e}")
            return None

    def save_matches(self, matches, filename=None):
        """Sauvegarder les matchs"""
        if not matches:
            print(f"\nAucun match a sauvegarder")
            return
        
        if not filename:
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f'european_leagues_matches_{date_str}.csv'
        
        # Créer DataFrame
        df = pd.DataFrame(matches)
        
        # Créer le dossier de sortie
        os.makedirs('Match of ze day', exist_ok=True)
        
        # Sauvegarder
        filepath = os.path.join('Match of ze day', filename)
        df.to_csv(filepath, index=False)
        
        print(f"\n=== SAUVEGARDE ===")
        print(f"Fichier: {filepath}")
        print(f"Matchs sauvegardes: {len(matches)}")
        
        # Statistiques par ligue
        if not df.empty:
            print(f"\nRepartition par ligue:")
            league_counts = df['league'].value_counts()
            for league, count in league_counts.items():
                print(f"  {league}: {count} match(s)")
        
        return filepath

    def display_matches(self, matches):
        """Afficher les matchs trouvés"""
        if not matches:
            print(f"\nAucun match trouve")
            return
        
        print(f"\n=== MATCHS TROUVES ({len(matches)}) ===")
        
        # Grouper par ligue
        by_league = {}
        for match in matches:
            league = match['league']
            if league not in by_league:
                by_league[league] = []
            by_league[league].append(match)
        
        # Afficher par ligue
        for league, league_matches in by_league.items():
            print(f"\n{league}:")
            for match in league_matches:
                odds_text = ""
                if match.get('B365H') and match.get('B365D') and match.get('B365A'):
                    odds_text = f" - Cotes: H:{match['B365H']:.2f} D:{match['B365D']:.2f} A:{match['B365A']:.2f}"
                elif any([match.get('B365H'), match.get('B365D'), match.get('B365A')]):
                    odds_text = f" - Cotes partielles: H:{match.get('B365H', 'N/A')} D:{match.get('B365D', 'N/A')} A:{match.get('B365A', 'N/A')}"
                else:
                    odds_text = " - Pas de cotes"
                print(f"  {match['home_team']} vs {match['away_team']}{odds_text}")

def main():
    """Fonction principale"""
    scrapper = EuropeanLeaguesScrapper()
    
    # Configuration
    start_date = datetime.now().strftime('%Y-%m-%d')  # Date du jour
    days_ahead = 4             # Nombre de jours à vérifier
    
    print(f"Recherche matchs ligues europeennes...")
    print(f"Date debut: {start_date}")
    print(f"Periode: {days_ahead + 1} jours")
    
    # Récupérer les matchs
    matches = scrapper.get_matches_for_date_range(start_date, days_ahead)
    
    # Afficher les résultats
    scrapper.display_matches(matches)
    
    # Sauvegarder
    if matches:
        filepath = scrapper.save_matches(matches)
        print(f"\nFICHIER CREE: {filepath}")
    else:
        print(f"\n=== AUCUN MATCH TROUVE ===")
        print(f"Causes possibles:")
        print(f"- Periode hors saison pour les ligues europeennes")
        print(f"- Matchs non encore programmes")
        print(f"- Probleme de connectivite API")
        
        print(f"\nINFO: Les saisons 2025-2026 commencent:")
        print(f"- Premier League: 15 aout 2025")
        print(f"- La Liga: 15 aout 2025") 
        print(f"- Ligue 1: 17 aout 2025")
        print(f"- Bundesliga: 22 aout 2025")
        print(f"- Serie A: 23 aout 2025")

if __name__ == '__main__':
    main()