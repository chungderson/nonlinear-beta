#!/usr/bin/env python3
"""
Alpaca Markets API functions for fetching historical bar data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json

def load_config():
    """Load API configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config['ALPACA_KEY'], config['ALPACA_SECRET']
    except FileNotFoundError:
        print("config.json not found. Please create it with your Alpaca API credentials.")
        return None, None

def getBars(symbol, start_date, end_date, timeframe='1Day'):
    """
    Fetch historical bar data from Alpaca Markets API.
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        timeframe (str): Bar timeframe ('1Min', '5Min', '15Min', '30Min', '1Hour', '1Day')
    
    Returns:
        pandas.DataFrame: Historical bar data with columns [open, high, low, close, volume]
    """
    api_key, secret_key = load_config()
    if not api_key or not secret_key:
        return None
    
    # Alpaca API base URL
    base_url = "https://data.alpaca.markets"
    
    # Headers for authentication
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key
    }
    
    # Convert dates to ISO 8601 format with timezone
    start_iso = f"{start_date}T00:00:00-04:00"
    end_iso = f"{end_date}T23:59:59-04:00"
    
    # API endpoint
    url = f"{base_url}/v2/stocks/{symbol}/bars"
    
    params = {
        "start": start_iso,
        "end": end_iso,
        "timeframe": timeframe,
        "limit": 10000
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'bars' not in data or not data['bars']:
            print(f"No data found for {symbol}")
            return None
        
        # Convert to DataFrame
        bars = data['bars']
        df = pd.DataFrame(bars)
        
        # Convert timestamp to datetime
        df['t'] = pd.to_datetime(df['t'])
        df.set_index('t', inplace=True)
        
        # Rename columns to match expected format
        df.rename(columns={
            'o': 'open',
            'h': 'high', 
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }, inplace=True)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for {symbol}: {e}")
        return None

def getBars5Min(symbol, start_date, end_date):
    """Fetch 5-minute bars."""
    return getBars(symbol, start_date, end_date, '5Min')

def getBars15Min(symbol, start_date, end_date):
    """Fetch 15-minute bars."""
    return getBars(symbol, start_date, end_date, '15Min')

def getBars30Min(symbol, start_date, end_date):
    """Fetch 30-minute bars."""
    return getBars(symbol, start_date, end_date, '30Min')

def getBars1Day(symbol, start_date, end_date):
    """Fetch daily bars."""
    return getBars(symbol, start_date, end_date, '1Day') 