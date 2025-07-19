#!/usr/bin/env python3
"""
Scrape S&P 500 companies from Wikipedia table with GICS sectors.
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

def scrape_sp500_wikipedia():
    """
    Scrape the actual S&P 500 companies from Wikipedia table.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main table
        table = soup.find('table', {'class': 'wikitable'})
        
        if not table:
            print("Could not find S&P 500 table on Wikipedia")
            return None
        
        # Extract data
        data = []
        rows = table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 4:
                symbol = cells[0].get_text(strip=True)
                company = cells[1].get_text(strip=True)
                sector = cells[2].get_text(strip=True)
                sub_industry = cells[3].get_text(strip=True)
                
                # Clean up symbol (remove any extra characters)
                symbol = re.sub(r'[^\w.]', '', symbol)
                
                if symbol and sector:
                    data.append({
                        'symbol': symbol,
                        'company': company,
                        'sector': sector,
                        'sub_industry': sub_industry
                    })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return None

def create_sp500_dict(df):
    """
    Convert DataFrame to dictionary format for our analysis.
    """
    if df is None or df.empty:
        return {}, {}
    
    # Create symbol to sector mapping
    sector_map = {}
    symbols = []
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        sector = row['sector']
        
        if symbol and sector:
            sector_map[symbol] = sector
            symbols.append(symbol)
    
    return symbols, sector_map

def main():
    """Main function to scrape and display S&P 500 data."""
    print("Scraping S&P 500 companies from Wikipedia...")
    
    df = scrape_sp500_wikipedia()
    
    if df is not None:
        print(f"Successfully scraped {len(df)} companies")
        print("\nFirst 10 companies:")
        print(df.head(10))
        
        print("\nSector breakdown:")
        sector_counts = df['sector'].value_counts()
        for sector, count in sector_counts.items():
            print(f"{sector}: {count} companies")
        
        # Create the dictionaries for our analysis
        symbols, sector_map = create_sp500_dict(df)
        
        print(f"\nTotal symbols: {len(symbols)}")
        print(f"Total sectors: {len(set(sector_map.values()))}")
        
        # Save to file for use in other scripts
        df.to_csv('sp500_wikipedia_data.csv', index=False)
        print("\nData saved to 'sp500_wikipedia_data.csv'")
        
        return symbols, sector_map
    else:
        print("Failed to scrape S&P 500 data")
        return None, None

if __name__ == "__main__":
    main() 