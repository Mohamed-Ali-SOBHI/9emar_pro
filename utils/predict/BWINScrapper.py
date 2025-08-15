from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv
import os


def extract_league_from_url(url):
    """
    Extract league name from BWIN URL
    """
    # Mapping des URLs vers les noms de ligues
    league_mappings = {
        'laliga': 'La Liga',
        'premier-league': 'Premier League',
        'ligue-1': 'Ligue 1',
        'bundesliga': 'Bundesliga',
        'serie-a': 'Serie A',
        'mls': 'MLS',
        'premier-league-102850': 'Russian Premier League'
    }
    
    url_lower = url.lower()
    for key, league_name in league_mappings.items():
        if key in url_lower:
            return league_name
    
    # Si pas trouvé, essayer d'extraire depuis l'URL
    if '/espagne-28/' in url_lower:
        return 'La Liga'
    elif '/angleterre-14/' in url_lower:
        return 'Premier League'
    elif '/france-16/' in url_lower:
        return 'Ligue 1'
    elif '/allemagne-17/' in url_lower:
        return 'Bundesliga'
    elif '/italie-20/' in url_lower:
        return 'Serie A'
    
    return 'Unknown League'

def scrappe(urls):
    """
    Scrappe the data from the url
    :param url: url to scrappe the data from
    :return: data
    """
    
    matches_data = []

    for url in urls:
        league_name = extract_league_from_url(url)
        # Configuration du navigateur pour le mode headless
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # Automatically download and setup ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        print('processing url:', url)
        driver.get(url)

        try:
            # Utiliser une attente explicite pour attendre que les matchs soient chargés
            element_present = EC.presence_of_element_located((By.CLASS_NAME, 'grid-event-wrapper'))
            try:
                WebDriverWait(driver, 10).until(element_present)
            except:
                print('Timed out waiting for page to load')

            # Obtenir le contenu de la page
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # Debug: Print page title and sample content
            page_title = soup.find('title')
            print(f"Page title: {page_title.text if page_title else 'No title'}")
            
            # Try different selectors to find matches
            matches = soup.find_all('div', class_='grid-event-wrapper')
            print(f"Found {len(matches)} matches with 'grid-event-wrapper' class")
            
            # If no matches found, try alternative selectors
            if not matches:
                matches = soup.find_all('div', class_='event-item')
                print(f"Found {len(matches)} matches with 'event-item' class")
            
            if not matches:
                matches = soup.find_all('div', class_='ms-event-pick')
                print(f"Found {len(matches)} matches with 'ms-event-pick' class")
            
            # Print sample HTML for debugging
            sample_html = str(soup)[:2000] + "..." if len(str(soup)) > 2000 else str(soup)
            print(f"Sample HTML: {sample_html}")
            
            for match in matches:
                    teams = match.find_all('div', class_='participant')
                    if teams:
                        team_1 = teams[0].get_text(strip=True)
                        team_2 = teams[1].get_text(strip=True) if len(teams) > 1 else None
                    else:
                        team_1, team_2 = None, None

                    # Cotes
                    odds = match.find_all('div', class_='option-value')
                    if odds:
                        odd_1 = odds[0].get_text(strip=True)
                        odd_2 = odds[1].get_text(strip=True) if len(odds) > 1 else None
                        odd_3 = odds[2].get_text(strip=True) if len(odds) > 2 else None
                    else:
                        odd_1, odd_2, odd_3 = None, None, None

                    match_data = {
                        'league': league_name,
                        'team_1': team_1,
                        'team_2': team_2,
                        'odd_1': odd_1,
                        'odd_2': odd_2,
                        'odd_3': odd_3,
                    }
                    matches_data.append(match_data)
                    
        finally:
            driver.quit()   

    return matches_data

def save_data(matches_data, day):
    """
    Save the matches data to a csv file
    :param matches_data: matches data to save
    :param day: string indicating the day (e.g. "today" or "tomorrow" or "plus_2")
    """
    if day == "today":
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    directory = "Match of ze day"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{directory}/matches_of_ze_day_{date_str}.csv"
    
    if not matches_data:
        print(f"No matches data found for {day}")
        return
        
    keys = matches_data[0].keys()
    
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(matches_data)
    
    print(f"Saved {len(matches_data)} matches to {filename}")

if __name__ == '__main__':
    
    urls_aujourdhui = [
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/espagne-28/laliga-102829", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd'hui/angleterre-14/premier-league-102841", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/france-16/ligue-1-102843", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/allemagne-17/bundesliga-102842", 
        "https://sports.bwin.fr/fr/sports/football-4/aujourd-hui/italie-20/serie-a-102846",
        
    ]

    # Pour aujourd'hui
    matches_data_today = scrappe(urls_aujourdhui)
    print(matches_data_today)
    save_data(matches_data_today, "today")
    