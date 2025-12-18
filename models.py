"""
Data models for the Financial Digital Twin API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class RiskAppetite(str, Enum):
    """Risk appetite levels for investment strategies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserInput(BaseModel):
    """User input for financial planning."""
    current_income: float = Field(..., gt=0, description="Current annual income")
    target_investment_amount: float = Field(..., gt=0, description="Target investment amount to achieve")
    investment_tenure_years: int = Field(..., gt=0, le=50, description="Investment tenure in years")
    risk_appetite: RiskAppetite = Field(..., description="Risk tolerance level")
    initial_investment: Optional[float] = Field(0, ge=0, description="Initial investment amount")


class FinancialInstrument(BaseModel):
    """Financial instrument data."""
    symbol: str = Field(..., description="Ticker symbol")
    name: str = Field(..., description="Instrument name")
    current_price: float = Field(..., description="Current price")
    historical_returns: List[float] = Field(..., description="Historical daily returns")
    volatility: float = Field(..., description="Annualized volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown")


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation result."""
    mean_final_value: float = Field(..., description="Mean final portfolio value")
    median_final_value: float = Field(..., description="Median final portfolio value")
    std_final_value: float = Field(..., description="Standard deviation of final values")
    probability_of_success: float = Field(..., description="Probability of achieving target")
    worst_case_5th_percentile: float = Field(..., description="5th percentile (worst case)")
    best_case_95th_percentile: float = Field(..., description="95th percentile (best case)")
    simulation_paths: List[List[float]] = Field(..., description="Sample simulation paths")


class InvestmentStrategy(BaseModel):
    """Investment strategy recommendation."""
    strategy_name: str = Field(..., description="Name of the strategy")
    description: str = Field(..., description="Strategy description")
    recommended_allocation: Dict[str, float] = Field(..., description="Asset allocation percentages")
    expected_annual_return: float = Field(..., description="Expected annual return")
    expected_volatility: float = Field(..., description="Expected volatility")
    sharpe_ratio: float = Field(..., description="Expected Sharpe ratio")
    max_drawdown: float = Field(..., description="Expected maximum drawdown")
    probability_of_success: float = Field(..., description="Probability of achieving target")
    monte_carlo_result: MonteCarloResult = Field(..., description="Monte Carlo simulation results")
    risk_level: RiskAppetite = Field(..., description="Risk level of this strategy")


class SimulationRequest(BaseModel):
    """Request for running simulations."""
    user_input: UserInput = Field(..., description="User financial input")
    symbols: List[str] = Field(..., description="List of financial instruments to analyze")
    simulation_count: int = Field(100, ge=10, le=100, description="Number of Monte Carlo simulations (limited to 100 for real-time performance)")
    use_yfinance: bool = Field(True, description="Whether to use real-time Yahoo Finance data")
    real_time_mode: bool = Field(False, description="Enable real-time mode for faster results")


class SimulationResponse(BaseModel):
    """Response containing simulation results and recommendations."""
    user_input: UserInput = Field(..., description="Original user input")
    available_instruments: List[FinancialInstrument] = Field(..., description="Available financial instruments")
    recommended_strategies: List[InvestmentStrategy] = Field(..., description="Recommended investment strategies")
    simulation_timestamp: datetime = Field(default_factory=datetime.now, description="When simulation was run")
    ai_insights: Optional[Dict[str, Any]] = Field(None, description="AI-generated financial insights")


class UserProfile(BaseModel):
    """User profile for saving preferences and history."""
    user_id: str = Field(..., description="Unique user identifier")
    name: str = Field(..., description="User name")
    email: Optional[str] = Field(None, description="User email")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    simulation_history: List[SimulationResponse] = Field(default_factory=list, description="Previous simulation results")
    created_at: datetime = Field(default_factory=datetime.now, description="Profile creation time")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")


class ChartData(BaseModel):
    """Chart data for frontend visualization."""
    labels: List[str] = Field(..., description="Chart labels (time periods)")
    datasets: List[Dict[str, Any]] = Field(..., description="Chart datasets")
    options: Dict[str, Any] = Field(default_factory=dict, description="Chart options")


class DashboardData(BaseModel):
    """Dashboard data for frontend."""
    user_profile: UserProfile = Field(..., description="User profile")
    market_overview: Dict[str, Any] = Field(..., description="Current market overview")
    recent_simulations: List[SimulationResponse] = Field(..., description="Recent simulation results")
    portfolio_chart: ChartData = Field(..., description="Portfolio performance chart")
    risk_analysis_chart: ChartData = Field(..., description="Risk analysis chart")


class AIInsights(BaseModel):
    """AI-generated financial insights."""
    ai_analysis: str = Field(..., description="Main AI analysis text")
    risk_assessment: str = Field(..., description="Risk level assessment")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    warnings: List[str] = Field(..., description="Risk warnings and cautions")
    next_steps: List[str] = Field(..., description="Recommended next steps")
    confidence_level: str = Field(..., description="Confidence level in recommendations")
    generated_at: datetime = Field(..., description="When insights were generated")


class MarketInsights(BaseModel):
    """AI-generated market insights."""
    market_sentiment: str = Field(..., description="Overall market sentiment")
    key_observations: List[str] = Field(..., description="Key market observations")
    recommendations: List[str] = Field(..., description="Market-based recommendations")
    generated_at: datetime = Field(..., description="When insights were generated")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
