#!/usr/bin/env python3
"""
Helper methods for trading day calculations and drift analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json

def load_config():
    """Load API configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        return config['alpaca']['api_key'], config['alpaca']['secret_key']
    except FileNotFoundError:
        print("config.json not found. Please create it with your Alpaca API credentials.")
        return None, None

def getTradingDays(start_date, end_date):
    """
    Get list of trading days between start_date and end_date.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
    
    Returns:
        list: List of trading day dates
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
    
    # API endpoint for calendar
    url = f"{base_url}/v2/stocks/SPY/calendar"
    
    params = {
        "start": start_iso,
        "end": end_iso
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'calendar' not in data:
            print("No calendar data found")
            return None
        
        # Extract trading days
        trading_days = []
        for day in data['calendar']:
            trading_days.append(day['date'])
        
        return trading_days
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching trading calendar: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching calendar: {e}")
        return None

def calculateDrift(df, window=20):
    """
    Calculate price drift over a rolling window.
    
    Args:
        df (pandas.DataFrame): Price data with 'close' column
        window (int): Rolling window size
    
    Returns:
        pandas.Series: Drift values
    """
    if 'close' not in df.columns:
        print("DataFrame must have 'close' column")
        return None
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Calculate rolling mean (drift)
    drift = returns.rolling(window=window).mean()
    
    return drift

def plotDrift(df, symbol, window=20):
    """
    Plot price drift analysis.
    
    Args:
        df (pandas.DataFrame): Price data
        symbol (str): Stock symbol for title
        window (int): Rolling window size
    """
    import matplotlib.pyplot as plt
    
    drift = calculateDrift(df, window)
    
    if drift is None:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Price plot
    ax1.plot(df.index, df['close'], label='Price')
    ax1.set_title(f'{symbol} Price and Drift Analysis')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Drift plot
    ax2.plot(drift.index, drift, label=f'{window}-day Drift', color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drift')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show() 