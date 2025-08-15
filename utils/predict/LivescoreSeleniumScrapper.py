from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import pandas as pd
import time
import json
import os
from datetime import datetime

class LivescoreSeleniumScrapper:
    def __init__(self):
        self.base_url = "https://www.livescore.in"
        self.driver = None
        self.wait = None
        
        self.league_mapping = {
            'La_liga': ['LaLiga', 'La Liga', 'Spain Primera Division', 'Espagne'],
            'Bundesliga': ['Bundesliga', 'Germany Bundesliga', 'Allemagne'], 
            'EPL': ['Premier League', 'England Premier League', 'EPL', 'Angleterre'],
            'Serie_A': ['Serie A', 'Italy Serie A', 'Italie'],
            'Ligue_1': ['Ligue 1', 'France Ligue 1', 'France']
        }

    def setup_driver(self):
        """Initialize Chrome WebDriver with options"""
        try:
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # Run in background
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # Try to create driver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.wait = WebDriverWait(self.driver, 15)
            
            print("Chrome WebDriver initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Make sure ChromeDriver is installed and in PATH")
            return False

    def load_page_and_wait(self, url, wait_for_selector=None):
        """Load page and wait for content to load"""
        try:
            print(f"Loading: {url}")
            self.driver.get(url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector)))
                    print(f"Found expected element: {wait_for_selector}")
                except:
                    print(f"Timeout waiting for: {wait_for_selector}")
            
            return True
            
        except Exception as e:
            print(f"Error loading page {url}: {e}")
            return False

    def extract_matches_from_page(self):
        """Extract match data from current page"""
        matches = []
        
        try:
            # Try multiple possible selectors for match elements
            selectors_to_try = [
                '[data-testid*="match"]',
                '[class*="match"]',
                '[class*="fixture"]',
                '[class*="event"]',
                'tr[class*="row"]',
                'div[class*="game"]'
            ]
            
            for selector in selectors_to_try:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                print(f"Found {len(elements)} elements with selector: {selector}")
                
                if elements:
                    for element in elements[:10]:  # Limit to first 10 to avoid timeout
                        try:
                            match_data = self.parse_match_element(element)
                            if match_data:
                                matches.append(match_data)
                        except Exception as e:
                            print(f"Error parsing match element: {e}")
                            continue
                            
            # Try to extract from JavaScript variables
            js_matches = self.extract_from_javascript()
            matches.extend(js_matches)
            
        except Exception as e:
            print(f"Error extracting matches: {e}")
            
        return matches

    def parse_match_element(self, element):
        """Parse individual match element"""
        try:
            # Get text content
            text = element.text.strip()
            
            if not text or len(text) < 5:
                return None
                
            # Look for team names and odds patterns
            lines = text.split('\n')
            
            # Simple pattern matching for common formats
            for i, line in enumerate(lines):
                if 'vs' in line.lower() or ' - ' in line:
                    # Found potential match line
                    parts = line.replace(' vs ', ' - ').split(' - ')
                    if len(parts) >= 2:
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                        
                        # Look for odds in surrounding lines
                        odds = []
                        for j in range(max(0, i-2), min(len(lines), i+3)):
                            odds.extend(self.extract_odds_from_text(lines[j]))
                        
                        if home_team and away_team:
                            return {
                                'league': 'Unknown',
                                'home_team': home_team,
                                'away_team': away_team,
                                'odds': odds[:3],  # Take first 3 odds
                                'match_time': '',
                                'date': datetime.now().strftime('%Y-%m-%d')
                            }
                            
        except Exception as e:
            print(f"Error parsing element: {e}")
            
        return None

    def extract_odds_from_text(self, text):
        """Extract odds from text using regex"""
        import re
        
        # Look for decimal odds pattern (1.50, 2.30, etc.)
        odds_pattern = r'\b(\d{1,2}\.\d{2})\b'
        matches = re.findall(odds_pattern, text)
        
        valid_odds = []
        for match in matches:
            try:
                odd_value = float(match)
                if 1.01 <= odd_value <= 50.0:
                    valid_odds.append(match)
            except:
                continue
                
        return valid_odds

    def extract_from_javascript(self):
        """Extract match data from JavaScript variables"""
        matches = []
        
        try:
            # Execute JavaScript to get window variables
            js_code = """
            var results = [];
            
            // Common variable names that might contain match data
            var vars_to_check = ['matches', 'fixtures', 'games', 'events', 'data'];
            
            for (var i = 0; i < vars_to_check.length; i++) {
                var varName = vars_to_check[i];
                if (window[varName] && typeof window[varName] === 'object') {
                    results.push({name: varName, data: window[varName]});
                }
            }
            
            return JSON.stringify(results);
            """
            
            result = self.driver.execute_script(js_code)
            data = json.loads(result)
            
            print(f"Found {len(data)} JavaScript variables with potential match data")
            
            for var_info in data:
                var_data = var_info['data']
                if isinstance(var_data, list):
                    for item in var_data[:10]:  # Limit processing
                        if isinstance(item, dict):
                            match_data = self.parse_js_match(item)
                            if match_data:
                                matches.append(match_data)
                                
        except Exception as e:
            print(f"Error extracting from JavaScript: {e}")
            
        return matches

    def parse_js_match(self, match_obj):
        """Parse match object from JavaScript"""
        try:
            # Look for common keys
            home_keys = ['home', 'homeTeam', 'team1', 'home_team']
            away_keys = ['away', 'awayTeam', 'team2', 'away_team'] 
            league_keys = ['league', 'competition', 'tournament']
            odds_keys = ['odds', 'betting', 'prices']
            
            home_team = ""
            away_team = ""
            league = ""
            odds = []
            
            # Extract team names
            for key in home_keys:
                if key in match_obj:
                    home_team = str(match_obj[key])
                    break
                    
            for key in away_keys:
                if key in match_obj:
                    away_team = str(match_obj[key])
                    break
                    
            # Extract league
            for key in league_keys:
                if key in match_obj:
                    league = str(match_obj[key])
                    break
                    
            # Extract odds
            for key in odds_keys:
                if key in match_obj:
                    odds_data = match_obj[key]
                    if isinstance(odds_data, list):
                        odds = [str(x) for x in odds_data[:3]]
                    elif isinstance(odds_data, dict):
                        odds = [str(v) for v in odds_data.values()][:3]
                    break
                    
            if home_team and away_team and self.is_target_league(league):
                return {
                    'league': self.normalize_league_name(league),
                    'home_team': home_team,
                    'away_team': away_team,
                    'odds': odds,
                    'match_time': '',
                    'date': datetime.now().strftime('%Y-%m-%d')
                }
                
        except Exception as e:
            print(f"Error parsing JS match: {e}")
            
        return None

    def is_target_league(self, league_name):
        """Check if league is one of our targets"""
        if not league_name:
            return False
            
        league_lower = league_name.lower()
        for target_league, variations in self.league_mapping.items():
            for variation in variations:
                if variation.lower() in league_lower:
                    return True
        return False

    def normalize_league_name(self, league_name):
        """Convert to standard league name"""
        league_lower = league_name.lower()
        for standard_name, variations in self.league_mapping.items():
            for variation in variations:
                if variation.lower() in league_lower:
                    return standard_name
        return league_name

    def scrape_all_matches(self):
        """Main method to scrape matches"""
        if not self.setup_driver():
            return []
            
        all_matches = []
        
        try:
            # Try main football page first
            urls_to_try = [
                f"{self.base_url}/fr/football",
                f"{self.base_url}/fr",
                f"{self.base_url}/fr/football/espagne/laliga/",
                f"{self.base_url}/fr/football/allemagne/bundesliga/",
                f"{self.base_url}/fr/football/angleterre/premier-league/",
                f"{self.base_url}/fr/football/italie/serie-a/",
                f"{self.base_url}/fr/football/france/ligue-1/"
            ]
            
            for url in urls_to_try:
                if self.load_page_and_wait(url):
                    matches = self.extract_matches_from_page()
                    all_matches.extend(matches)
                    
                    if matches:
                        print(f"Found {len(matches)} matches from {url}")
                        
                    # Wait between requests
                    time.sleep(2)
                    
        finally:
            if self.driver:
                self.driver.quit()
                print("WebDriver closed")
                
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
            filename = f"livescore_selenium_matches_{datetime.now().strftime('%Y%m%d')}.csv"
            
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
    """Main function"""
    scraper = LivescoreSeleniumScrapper()
    
    print("Starting Selenium-based Livescore scraping...")
    print("Note: This requires ChromeDriver to be installed")
    
    matches = scraper.scrape_all_matches()
    
    if matches:
        scraper.save_to_csv(matches)
        print(f"\nTotal unique matches found: {len(matches)}")
        
        # Show sample matches
        print("\nSample matches:")
        for i, match in enumerate(matches[:5]):
            print(f"{i+1}. {match['league']}: {match['home_team']} vs {match['away_team']} - Odds: {match['odds']}")
    else:
        print("No matches found with Selenium approach")


if __name__ == '__main__':
    main()