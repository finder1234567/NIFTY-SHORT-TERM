import os
import sys
import concurrent.futures
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from io import BytesIO
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

FALLBACK_NIFTY50 = {
    "RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank", "INFY.NS": "Infosys", "ITC.NS": "ITC Limited",
    "LT.NS": "Larsen & Toubro", "SBIN.NS": "State Bank of India", "MARUTI.NS": "Maruti Suzuki",
    "BAJAJFINSV.NS": "Bajaj Finserv", "WIPRO.NS": "Wipro", "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank", "BAJAJ-AUTO.NS": "Bajaj Auto", "BHARTIARTL.NS": "Bharti Airtel",
    "BPCL.NS": "Bharat Petroleum", "EICHERMOT.NS": "Eicher Motors", "GAIL.NS": "GAIL India",
    "GRASIM.NS": "Grasim Industries", "HCLTECH.NS": "HCL Technologies", "HDFC.NS": "HDFC Limited",
    "HEROMOTOCO.NS": "Hero MotoCorp", "HINDALCO.NS": "Hindalco Industries", "HINDUNILVR.NS": "Hindustan Unilever",
    "HONEYWELL.NS": "Honeywell Automation", "ICICIBANK.NS": "ICICI Bank", "INDIGO.NS": "IndiGo",
    "JSWSTEEL.NS": "JSW Steel", "KOTAKBANK.NS": "Kotak Mahindra Bank", "LTTS.NS": "L&T Technology",
    "LUPIN.NS": "Lupin Limited", "M&M.NS": "Mahindra & Mahindra", "MFSL.NS": "Max Financial Services",
    "MOTHERSON.NS": "Motherson Sumi Systems", "NTPC.NS": "NTPC Limited", "ONGC.NS": "Oil & Natural Gas",
    "POWERGRID.NS": "Power Grid Corporation", "SBICARD.NS": "SBI Card", "SUNPHARMA.NS": "Sun Pharmaceutical",
    "TATAMOTORS.NS": "Tata Motors", "TATAPOWER.NS": "Tata Power", "TATASTEEL.NS": "Tata Steel",
    "TECHM.NS": "Tech Mahindra", "TITAN.NS": "Titan Company", "TORNTPHARM.NS": "Torrent Pharmaceuticals",
    "ULTRACEMCO.NS": "UltraTech Cement", "UPL.NS": "UPL Limited", "YESBANK.NS": "Yes Bank"
}

class DummyOutput:
    def write(self, x): pass
    def flush(self): pass

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm