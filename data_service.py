"""
Yahoo Finance data fetching service for the Financial Digital Twin.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from models import FinancialInstrument

logger = logging.getLogger(__name__)


class YahooFinanceService:
    """Service for fetching and processing financial data from Yahoo Finance."""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
    
    def get_instrument_data(self, symbol: str, period: str = "5y") -> Optional[FinancialInstrument]:
        """
        Fetch financial data for a given symbol.
        
        Args:
            symbol: Stock/ETF symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            FinancialInstrument object or None if data unavailable
        """
        try:
            # Normalize symbol (uppercase, strip whitespace)
            symbol = symbol.strip().upper()
            
            # Check cache first
            cache_key = f"{symbol}_{period}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return cached_data
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.warning(f"No data available for symbol: {symbol}")
                return None
            
            # Get additional info
            info = ticker.info
            current_price = hist['Close'].iloc[-1]
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Calculate metrics
            volatility = self._calculate_volatility(returns)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(hist['Close'])
            
            # Create instrument object
            instrument = FinancialInstrument(
                symbol=symbol,
                name=info.get('longName', symbol),
                current_price=float(current_price),
                historical_returns=returns.tolist(),
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown
            )
            
            # Cache the result
            self.cache[cache_key] = (instrument, datetime.now())
            
            return instrument
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def get_multiple_instruments(self, symbols: List[str], period: str = "5y") -> List[FinancialInstrument]:
        """
        Fetch data for multiple instruments.
        
        Args:
            symbols: List of stock/ETF symbols
            period: Data period
            
        Returns:
            List of FinancialInstrument objects
        """
        instruments = []
        for symbol in symbols:
            instrument = self.get_instrument_data(symbol, period)
            if instrument:
                instruments.append(instrument)
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
        
        return instruments
    
    def get_popular_etfs(self) -> List[str]:
        """Get list of popular ETF symbols for analysis."""
        return [
            "SPY",  # SPDR S&P 500 ETF
            "QQQ",  # Invesco QQQ Trust
            "IWM",  # iShares Russell 2000 ETF
            "VTI",  # Vanguard Total Stock Market ETF
            "VEA",  # Vanguard FTSE Developed Markets ETF
            "VWO",  # Vanguard FTSE Emerging Markets ETF
            "BND",  # Vanguard Total Bond Market ETF
            "TLT",  # iShares 20+ Year Treasury Bond ETF
            "GLD",  # SPDR Gold Trust
            "VNQ",  # Vanguard Real Estate ETF
            "AAPL", # Apple Inc.
            "MSFT", # Microsoft Corporation
            "GOOGL", # Alphabet Inc.
            "AMZN", # Amazon.com Inc.
            "TSLA", # Tesla Inc.
            "NVDA", # NVIDIA Corporation
            "META", # Meta Platforms Inc.
            "JPM",  # JPMorgan Chase & Co.
            "JNJ",  # Johnson & Johnson
            "V",    # Visa Inc.
        ]
    
    def get_asset_class_instruments(self) -> Dict[str, List[Dict[str, str]]]:
        """Get instruments organized by asset class."""
        return {
            "stocks": [
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "description": "Large-cap US stocks"},
                {"symbol": "QQQ", "name": "Invesco QQQ Trust", "description": "NASDAQ-100 technology stocks"},
                {"symbol": "VTI", "name": "Vanguard Total Stock Market ETF", "description": "Total US stock market"},
                {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "description": "Small-cap US stocks"},
                {"symbol": "VEA", "name": "Vanguard FTSE Developed Markets ETF", "description": "International developed markets"},
                {"symbol": "VWO", "name": "Vanguard FTSE Emerging Markets ETF", "description": "Emerging markets"},
                {"symbol": "AAPL", "name": "Apple Inc.", "description": "Technology giant"},
                {"symbol": "MSFT", "name": "Microsoft Corporation", "description": "Cloud computing leader"},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "description": "Search and advertising"},
                {"symbol": "TSLA", "name": "Tesla Inc.", "description": "Electric vehicles"},
            ],
            "bonds": [
                {"symbol": "BND", "name": "Vanguard Total Bond Market ETF", "description": "Total US bond market"},
                {"symbol": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "description": "Long-term US treasuries"},
                {"symbol": "IEF", "name": "iShares 7-10 Year Treasury Bond ETF", "description": "Intermediate treasuries"},
                {"symbol": "SHY", "name": "iShares 1-3 Year Treasury Bond ETF", "description": "Short-term treasuries"},
                {"symbol": "LQD", "name": "iShares iBoxx $ Investment Grade Corporate Bond ETF", "description": "Corporate bonds"},
                {"symbol": "HYG", "name": "iShares iBoxx $ High Yield Corporate Bond ETF", "description": "High yield bonds"},
            ],
            "real_estate": [
                {"symbol": "VNQ", "name": "Vanguard Real Estate ETF", "description": "US real estate investment trusts"},
                {"symbol": "IYR", "name": "iShares U.S. Real Estate ETF", "description": "US real estate sector"},
                {"symbol": "SCHH", "name": "Schwab U.S. REIT ETF", "description": "US REITs"},
                {"symbol": "VNQI", "name": "Vanguard Global ex-U.S. Real Estate ETF", "description": "International real estate"},
                {"symbol": "RWO", "name": "SPDR Dow Jones Global Real Estate ETF", "description": "Global real estate"},
                {"symbol": "O", "name": "Realty Income Corporation", "description": "Monthly dividend REIT"},
            ],
            "commodities": [
                {"symbol": "GLD", "name": "SPDR Gold Trust", "description": "Physical gold"},
                {"symbol": "IAU", "name": "iShares Gold Trust", "description": "Physical gold (alternative)"},
                {"symbol": "SLV", "name": "iShares Silver Trust", "description": "Physical silver"},
                {"symbol": "DJP", "name": "iPath Bloomberg Commodity Index Total Return ETN", "description": "Broad commodities"},
                {"symbol": "USO", "name": "United States Oil Fund LP", "description": "Crude oil"},
                {"symbol": "UNG", "name": "United States Natural Gas Fund LP", "description": "Natural gas"},
            ],
            "cryptocurrency": [
                {"symbol": "BTC-USD", "name": "Bitcoin", "description": "Digital gold, store of value"},
                {"symbol": "ETH-USD", "name": "Ethereum", "description": "Smart contract platform"},
                {"symbol": "COIN", "name": "Coinbase Global Inc.", "description": "Cryptocurrency exchange"},
                {"symbol": "MSTR", "name": "MicroStrategy Incorporated", "description": "Bitcoin proxy stock"},
                {"symbol": "RIOT", "name": "Riot Platforms Inc.", "description": "Bitcoin mining"},
                {"symbol": "MARA", "name": "Marathon Digital Holdings Inc.", "description": "Bitcoin mining"},
            ],
            "alternative_investments": [
                {"symbol": "ARKK", "name": "ARK Innovation ETF", "description": "Disruptive innovation"},
                {"symbol": "TAN", "name": "Invesco Solar ETF", "description": "Solar energy"},
                {"symbol": "ICLN", "name": "iShares Global Clean Energy ETF", "description": "Clean energy"},
                {"symbol": "BLOK", "name": "Amplify Transformational Data Sharing ETF", "description": "Blockchain technology"},
                {"symbol": "ROBO", "name": "ROBO Global Robotics and Automation Index ETF", "description": "Robotics and AI"},
                {"symbol": "ESG", "name": "iShares ESG Aware MSCI USA ETF", "description": "ESG investing"},
            ]
        }
    
    def get_stock_indices(self) -> List[str]:
        """Get list of major stock indices."""
        return [
            "^GSPC",  # S&P 500
            "^DJI",   # Dow Jones Industrial Average
            "^IXIC",  # NASDAQ Composite
            "^RUT",   # Russell 2000
        ]
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        return float(returns.std() * np.sqrt(252))
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns.mean() * 252 - risk_free_rate
        volatility = returns.std() * np.sqrt(252)
        return float(excess_returns / volatility) if volatility > 0 else 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min())
    
    def get_correlation_matrix(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Calculate correlation matrix for given symbols.
        
        Args:
            symbols: List of symbols
            period: Data period
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            data = {}
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty:
                    data[symbol] = hist['Close'].pct_change().dropna()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            return df.corr()
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict[str, Any]:
        """
        Get current market overview with major indices.
        
        Returns:
            Dictionary with market overview data
        """
        try:
            overview = {}
            indices = self.get_stock_indices()
            
            for index in indices:
                ticker = yf.Ticker(index)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2]
                    change = current - previous
                    change_pct = (change / previous) * 100
                    
                    overview[index] = {
                        "current": float(current),
                        "change": float(change),
                        "change_pct": float(change_pct)
                    }
            
            return overview
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {str(e)}")
            return {}
