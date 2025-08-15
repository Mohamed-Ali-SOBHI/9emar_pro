from datetime import datetime, timedelta
import json
import os
import time
import numpy as np  
import pandas as pd 
import requests  
from bs4 import BeautifulSoup
from tqdm import tqdm  
import re

class LivescoreScrapper:
    def __init__(self):
        self.base_url = "https://www.livescore.in"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.league_mapping = {
            'La_liga': ['LaLiga', 'La Liga', 'Spain Primera Division'],
            'Bundesliga': ['Bundesliga', 'Germany Bundesliga'], 
            'EPL': ['Premier League', 'England Premier League', 'EPL'],
            'Serie_A': ['Serie A', 'Italy Serie A'],
            'Ligue_1': ['Ligue 1', 'France Ligue 1']
        }

    def get_page(self, url):
        """Fetch and parse a webpage with better error handling"""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=15)
            print(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                print(f"HTTP {response.status_code} for {url}")
                return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

    def extract_match_data(self, soup):
        """Extract match data and odds from the main page"""
        matches = []
        
        try:
            # Debug: Print page structure to understand layout
            print(f"Page title: {soup.title.text if soup.title else 'No title'}")
            
            # Look for various possible match containers
            possible_selectors = [
                'div[class*="match"]',
                'div[class*="fixture"]', 
                'div[class*="game"]',
                'div[class*="event"]',
                'tr[class*="match"]',
                'tr[class*="row"]',
                '.leftMenu__item',  # Based on WebFetch analysis
                '[data-match]',
                '[data-id]'
            ]
            
            for selector in possible_selectors:
                elements = soup.select(selector)
                print(f"Found {len(elements)} elements with selector: {selector}")
                
                if elements:
                    for elem in elements[:5]:  # Check first 5 elements
                        match_data = self.parse_match_element(elem)
                        if match_data:
                            matches.append(match_data)
                            
            # Alternative: Look for script tags containing JSON data
            script_tags = soup.find_all('script')
            for script in script_tags:
                if script.string and ('match' in script.string.lower() or 'fixture' in script.string.lower()):
                    matches.extend(self.extract_from_script(script.string))
                    
        except Exception as e:
            print(f"Error extracting match data: {e}")
            
        return matches

    def parse_match_element(self, match_elem):
        """Parse individual match element to extract teams and odds"""
        try:
            # Extract league information
            league_elem = match_elem.find(class_=re.compile(r'league|competition|tournament'))
            league = league_elem.text.strip() if league_elem else ""
            
            # Extract team names
            team_elements = match_elem.find_all(class_=re.compile(r'team|club'))
            if len(team_elements) >= 2:
                home_team = team_elements[0].text.strip()
                away_team = team_elements[1].text.strip()
            else:
                return None
            
            # Extract odds
            odds_elements = match_elem.find_all(class_=re.compile(r'odd|bet|price'))
            odds = []
            for odd_elem in odds_elements:
                odd_text = odd_elem.text.strip()
                if self.is_valid_odd(odd_text):
                    odds.append(odd_text)
            
            # Extract match time/date
            time_elem = match_elem.find(class_=re.compile(r'time|date|schedule'))
            match_time = time_elem.text.strip() if time_elem else ""
            
            if self.is_target_league(league) and home_team and away_team:
                return {
                    'league': self.normalize_league_name(league),
                    'home_team': home_team,
                    'away_team': away_team,
                    'odds': odds,
                    'match_time': match_time,
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
                
        except Exception as e:
            print(f"Error parsing match element: {e}")
            
        return None

    def is_valid_odd(self, odd_text):
        """Check if text represents a valid betting odd"""
        try:
            odd_float = float(odd_text.replace(',', '.'))
            return 1.0 <= odd_float <= 50.0
        except:
            return False

    def is_target_league(self, league_name):
        """Check if league is one of our target European leagues"""
        league_lower = league_name.lower()
        for target_league, variations in self.league_mapping.items():
            for variation in variations:
                if variation.lower() in league_lower:
                    return True
        return False

    def normalize_league_name(self, league_name):
        """Convert league name to our standard format"""
        league_lower = league_name.lower()
        for standard_name, variations in self.league_mapping.items():
            for variation in variations:
                if variation.lower() in league_lower:
                    return standard_name
        return league_name

    def extract_from_script(self, script_content):
        """Extract match data from JavaScript/JSON in script tags"""
        matches = []
        try:
            # Look for JSON objects containing match data
            import json
            
            # Try to find JSON objects in the script
            json_patterns = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', script_content)
            
            for pattern in json_patterns:
                try:
                    data = json.loads(pattern)
                    if isinstance(data, dict) and ('team' in str(data).lower() or 'match' in str(data).lower()):
                        # Extract relevant match data
                        pass
                except:
                    continue
                    
        except Exception as e:
            print(f"Error extracting from script: {e}")
            
        return matches

    def scrape_matches(self, days_ahead=3):
        """Main method to scrape matches for the next few days"""
        all_matches = []
        
        # Try main football pages for different leagues
        print("Trying league-specific pages...")
        league_urls = [
            f"{self.base_url}/fr/football/espagne/laliga/",
            f"{self.base_url}/fr/football/allemagne/bundesliga/", 
            f"{self.base_url}/fr/football/angleterre/premier-league/",
            f"{self.base_url}/fr/football/italie/serie-a/",
            f"{self.base_url}/fr/football/france/ligue-1/"
        ]
        
        for url in league_urls:
            soup = self.get_page(url)
            if soup:
                matches = self.extract_match_data(soup)
                all_matches.extend(matches)
                if matches:
                    print(f"Found {len(matches)} matches from {url}")
        
        # Try main football page
        print("Trying main football page...")
        main_urls = [
            f"{self.base_url}/fr/football",
            f"{self.base_url}/fr/football/",
            f"{self.base_url}/fr"
        ]
        
        for url in main_urls:
            soup = self.get_page(url)
            if soup:
                matches = self.extract_match_data(soup)
                all_matches.extend(matches)
                if matches:
                    print(f"Found {len(matches)} matches from {url}")
                    break
                        
        return self.deduplicate_matches(all_matches)

    def deduplicate_matches(self, matches):
        """Remove duplicate matches"""
        seen = set()
        unique_matches = []
        
        for match in matches:
            key = f"{match['league']}_{match['home_team']}_{match['away_team']}_{match['date']}"
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
                
        return unique_matches

    def save_to_csv(self, matches, filename=None):
        """Save matches to CSV file"""
        if not matches:
            print("No matches to save")
            return
            
        if not filename:
            filename = f"livescore_matches_{datetime.now().strftime('%Y%m%d')}.csv"
            
        # Create DataFrame
        df = pd.DataFrame(matches)
        
        # Ensure output directory exists
        os.makedirs('Match of ze day', exist_ok=True)
        
        # Save to CSV
        filepath = os.path.join('Match of ze day', filename)
        df.to_csv(filepath, index=False)
        
        print(f"Saved {len(matches)} matches to {filepath}")
        
        # Display summary
        if not df.empty:
            print("\nLeague distribution:")
            print(df['league'].value_counts())


def main():
    """Main function to run the scraper"""
    scraper = LivescoreScrapper()
    
    print("Starting Livescore scraping...")
    matches = scraper.scrape_matches(days_ahead=3)
    
    if matches:
        scraper.save_to_csv(matches)
        print(f"\nTotal matches found: {len(matches)}")
        
        # Show sample matches
        print("\nSample matches:")
        for i, match in enumerate(matches[:5]):
            print(f"{i+1}. {match['league']}: {match['home_team']} vs {match['away_team']} - Odds: {match['odds']}")
    else:
        print("No matches found. The site structure may have changed.")


if __name__ == '__main__':
    main()