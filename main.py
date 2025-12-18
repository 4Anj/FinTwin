"""
FastAPI Financial Digital Twin Server
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import json
import os

from models import (
    SimulationRequest, SimulationResponse, UserProfile, 
    DashboardData, ChartData, ErrorResponse, UserInput,
    AIInsights, MarketInsights
)
from data_service import YahooFinanceService
from strategy_engine import StrategyEngine
from diversified_strategy_engine import DiversifiedStrategyEngine
from monte_carlo import MonteCarloSimulator
from ai_insights_service import AIInsightsService
from blockchain_integrity import DataIntegrityService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Digital Twin API",
    description="AI-powered financial planning and investment simulation platform",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_service = YahooFinanceService()
strategy_engine = StrategyEngine()
diversified_strategy_engine = DiversifiedStrategyEngine()
monte_carlo = MonteCarloSimulator()
ai_insights_service = AIInsightsService()
integrity_service = DataIntegrityService()

# In-memory storage for demo (in production, use a database)
user_profiles: Dict[str, UserProfile] = {}
simulation_cache: Dict[str, SimulationResponse] = {}

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend application."""
    return FileResponse("static/index.html")


@app.post("/api/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """
    Run Monte Carlo simulation and generate investment recommendations.
    """
    try:
        logger.info(f"Running simulation for user input: {request.user_input}")
        
        # Get financial instruments data
        if request.use_yfinance:
            instruments = data_service.get_multiple_instruments(request.symbols)
        else:
            # Use mock data for testing without yfinance
            instruments = _get_mock_instruments(request.symbols)
        
        if not instruments:
            # Try with mock data as fallback
            logger.warning("No instruments found with real data, using mock data")
            instruments = _get_mock_instruments(request.symbols)
            
        if not instruments:
            raise HTTPException(
                status_code=400, 
                detail="No valid financial instruments found. Please check your symbols. Try: SPY, QQQ, VTI, BND, GLD"
            )
        
        # Generate investment strategies
        strategies = strategy_engine.generate_strategies(
            instruments, 
            request.user_input, 
            request.simulation_count
        )
        
        if not strategies:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate investment strategies"
            )
        
        # Generate AI insights for basic simulation too
        ai_insights = None
        try:
            simulation_data = {
                'user_input': request.user_input.dict(),
                'strategies': [strategy.dict() for strategy in strategies]
            }
            ai_insights_dict = ai_insights_service.generate_investment_insights(
                request.user_input.dict(),
                simulation_data,
                [strategy.dict() for strategy in strategies]
            )
            
            # Use the dictionary directly
            if ai_insights_dict:
                ai_insights = ai_insights_dict
        except Exception as e:
            logger.warning(f"AI insights generation failed: {str(e)}")
        
        # Create response
        response = SimulationResponse(
            user_input=request.user_input,
            available_instruments=instruments,
            recommended_strategies=strategies,
            ai_insights=ai_insights
        )
        
        # Cache the result
        simulation_id = str(uuid.uuid4())
        simulation_cache[simulation_id] = response
        
        # Create tamper-proof blockchain record
        try:
            blockchain_hash = integrity_service.create_simulation_record(
                simulation_id=simulation_id,
                user_id=request.user_input.user_id if hasattr(request.user_input, 'user_id') else 'anonymous',
                user_input=request.user_input.dict(),
                simulation_results=response.dict()
            )
            logger.info(f"Blockchain record created: {blockchain_hash}")
        except Exception as e:
            logger.warning(f"Failed to create blockchain record: {str(e)}")
        
        logger.info(f"Simulation completed successfully. ID: {simulation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/market-overview")
async def get_market_overview():
    """Get current market overview."""
    try:
        overview = data_service.get_market_overview()
        return {"market_overview": overview}
    except Exception as e:
        logger.error(f"Error fetching market overview: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instruments/popular")
async def get_popular_instruments():
    """Get list of popular financial instruments."""
    try:
        symbols = data_service.get_popular_etfs()
        # Return basic instrument info without fetching full data for performance
        instruments = []
        for symbol in symbols:
            instruments.append({
                "symbol": symbol,
                "name": _get_instrument_name(symbol)
            })
        return {"instruments": instruments}
    except Exception as e:
        logger.error(f"Error fetching popular instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/instruments/asset-classes")
async def get_asset_class_instruments():
    """Get instruments organized by asset class."""
    try:
        instruments_by_class = data_service.get_asset_class_instruments()
        return {"asset_classes": instruments_by_class}
    except Exception as e:
        logger.error(f"Error fetching asset class instruments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/user/profile", response_model=UserProfile)
async def create_user_profile(profile_data: Dict[str, Any]):
    """Create or update user profile."""
    try:
        user_id = profile_data.get("user_id", str(uuid.uuid4()))
        
        profile = UserProfile(
            user_id=user_id,
            name=profile_data.get("name", "Anonymous User"),
            email=profile_data.get("email"),
            preferences=profile_data.get("preferences", {}),
            simulation_history=profile_data.get("simulation_history", [])
        )
        
        user_profiles[user_id] = profile
        return profile
        
    except Exception as e:
        logger.error(f"Error creating user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/user/profile/{user_id}", response_model=UserProfile)
async def get_user_profile(user_id: str):
    """Get user profile by ID."""
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    return user_profiles[user_id]


@app.post("/api/user/profile/{user_id}/simulation")
async def save_simulation_to_profile(user_id: str, simulation_id: str):
    """Save a simulation result to user profile."""
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    user_profiles[user_id].simulation_history.append(simulation_cache[simulation_id])
    user_profiles[user_id].last_updated = datetime.now()
    
    return {"message": "Simulation saved to profile"}


@app.get("/api/dashboard/{user_id}", response_model=DashboardData)
async def get_dashboard_data(user_id: str):
    """Get dashboard data for a user."""
    if user_id not in user_profiles:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    try:
        profile = user_profiles[user_id]
        market_overview = data_service.get_market_overview()
        recent_simulations = profile.simulation_history[-5:]  # Last 5 simulations
        
        # Create portfolio chart data
        portfolio_chart = _create_portfolio_chart(recent_simulations)
        
        # Create risk analysis chart
        risk_chart = _create_risk_analysis_chart(recent_simulations)
        
        return DashboardData(
            user_profile=profile,
            market_overview=market_overview,
            recent_simulations=recent_simulations,
            portfolio_chart=portfolio_chart,
            risk_analysis_chart=risk_chart
        )
        
    except Exception as e:
        logger.error(f"Error creating dashboard data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulation/{simulation_id}", response_model=SimulationResponse)
async def get_simulation(simulation_id: str):
    """Get a specific simulation result."""
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return simulation_cache[simulation_id]


@app.post("/api/simulate/diversified", response_model=SimulationResponse)
async def run_diversified_simulation(request: SimulationRequest):
    """
    Run diversified Monte Carlo simulation across multiple asset classes.
    """
    try:
        logger.info(f"Running diversified simulation for user input: {request.user_input}")
        
        # Get instruments organized by asset class
        instruments_by_class = {}
        asset_classes = data_service.get_asset_class_instruments()
        
        for asset_class, instrument_list in asset_classes.items():
            symbols = [inst['symbol'] for inst in instrument_list]
            if request.use_yfinance:
                instruments = data_service.get_multiple_instruments(symbols)
            else:
                instruments = _get_mock_instruments(symbols)
            instruments_by_class[asset_class] = instruments
        
        # Generate diversified strategies
        strategies = diversified_strategy_engine.generate_diversified_strategies(
            instruments_by_class, 
            request.user_input, 
            request.simulation_count
        )
        
        if not strategies:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate diversified investment strategies"
            )
        
        # Get all instruments for response
        all_instruments = []
        for instruments in instruments_by_class.values():
            all_instruments.extend(instruments)
        
        # Generate AI insights
        ai_insights = None
        try:
            simulation_data = {
                'user_input': request.user_input.dict(),
                'strategies': [strategy.dict() for strategy in strategies]
            }
            ai_insights_dict = ai_insights_service.generate_investment_insights(
                request.user_input.dict(),
                simulation_data,
                [strategy.dict() for strategy in strategies]
            )
            
            # Use the dictionary directly
            if ai_insights_dict:
                ai_insights = ai_insights_dict
        except Exception as e:
            logger.warning(f"AI insights generation failed: {str(e)}")
        
        # Create response
        response = SimulationResponse(
            user_input=request.user_input,
            available_instruments=all_instruments,
            recommended_strategies=strategies,
            ai_insights=ai_insights
        )
        
        # Cache the result
        simulation_id = str(uuid.uuid4())
        simulation_cache[simulation_id] = response
        
        # Create tamper-proof blockchain record
        try:
            blockchain_hash = integrity_service.create_simulation_record(
                simulation_id=simulation_id,
                user_id=request.user_input.user_id if hasattr(request.user_input, 'user_id') else 'anonymous',
                user_input=request.user_input.dict(),
                simulation_results=response.dict()
            )
            logger.info(f"Blockchain record created: {blockchain_hash}")
        except Exception as e:
            logger.warning(f"Failed to create blockchain record: {str(e)}")
        
        logger.info(f"Diversified simulation completed successfully. ID: {simulation_id}")
        return response
        
    except Exception as e:
        logger.error(f"Error in diversified simulation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integrity/verify/{simulation_id}")
async def verify_simulation_integrity(simulation_id: str):
    """Verify the integrity of a simulation using blockchain."""
    try:
        verification_result = integrity_service.verify_simulation_integrity(simulation_id)
        return verification_result
    except Exception as e:
        logger.error(f"Error verifying simulation integrity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integrity/report")
async def get_integrity_report():
    """Get comprehensive blockchain integrity report."""
    try:
        report = integrity_service.get_integrity_report()
        return report
    except Exception as e:
        logger.error(f"Error generating integrity report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/integrity/chain")
async def get_blockchain_info():
    """Get blockchain information and status."""
    try:
        chain_info = integrity_service.blockchain.get_chain_info()
        return {
            "blockchain": chain_info,
            "public_key": integrity_service.blockchain.export_public_key(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting blockchain info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}


def _get_instrument_name(symbol: str) -> str:
    """Get display name for a financial instrument symbol."""
    names = {
        "SPY": "SPDR S&P 500 ETF Trust",
        "QQQ": "Invesco QQQ Trust",
        "IWM": "iShares Russell 2000 ETF",
        "VTI": "Vanguard Total Stock Market ETF",
        "VEA": "Vanguard FTSE Developed Markets ETF",
        "VWO": "Vanguard FTSE Emerging Markets ETF",
        "BND": "Vanguard Total Bond Market ETF",
        "TLT": "iShares 20+ Year Treasury Bond ETF",
        "GLD": "SPDR Gold Trust",
        "VNQ": "Vanguard Real Estate ETF",
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "GOOGL": "Alphabet Inc.",
        "AMZN": "Amazon.com Inc.",
        "TSLA": "Tesla Inc.",
        "NVDA": "NVIDIA Corporation",
        "META": "Meta Platforms Inc.",
        "JPM": "JPMorgan Chase & Co.",
        "JNJ": "Johnson & Johnson",
        "V": "Visa Inc."
    }
    return names.get(symbol, symbol)


def _get_mock_instruments(symbols: List[str]) -> List[Any]:
    """Generate mock instruments for testing without yfinance."""
    from models import FinancialInstrument
    import random
    
    mock_instruments = []
    for symbol in symbols:
        # Generate realistic mock data
        volatility = random.uniform(0.1, 0.3)
        sharpe_ratio = random.uniform(0.3, 1.2)
        
        instrument = FinancialInstrument(
            symbol=symbol,
            name=f"Mock {symbol}",
            current_price=random.uniform(50, 500),
            historical_returns=[random.gauss(0, volatility/252) for _ in range(252)],
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=random.uniform(0.1, 0.4)
        )
        mock_instruments.append(instrument)
    
    return mock_instruments


def _create_portfolio_chart(simulations: List[SimulationResponse]) -> ChartData:
    """Create portfolio performance chart data."""
    if not simulations:
        return ChartData(labels=[], datasets=[])
    
    # Use the most recent simulation
    latest_sim = simulations[-1]
    if not latest_sim.recommended_strategies:
        return ChartData(labels=[], datasets=[])
    
    # Get the best strategy's simulation paths
    best_strategy = latest_sim.recommended_strategies[0]
    paths = best_strategy.monte_carlo_result.simulation_paths
    
    if not paths:
        return ChartData(labels=[], datasets=[])
    
    # Create time labels
    years = latest_sim.user_input.investment_tenure_years
    labels = [f"Year {i+1}" for i in range(years)]
    
    # Create datasets for different percentiles
    datasets = []
    
    # Mean path
    mean_path = [sum(path[i] for path in paths) / len(paths) for i in range(len(paths[0]))]
    datasets.append({
        "label": "Mean Portfolio Value",
        "data": mean_path,
        "borderColor": "rgb(75, 192, 192)",
        "backgroundColor": "rgba(75, 192, 192, 0.2)",
        "fill": False
    })
    
    # Add some sample individual paths
    for i, path in enumerate(paths[:3]):  # Show first 3 paths
        datasets.append({
            "label": f"Simulation {i+1}",
            "data": path,
            "borderColor": f"rgba({100 + i*50}, {150 + i*30}, {200 + i*20}, 0.5)",
            "backgroundColor": f"rgba({100 + i*50}, {150 + i*30}, {200 + i*20}, 0.1)",
            "fill": False,
            "pointRadius": 0
        })
    
    return ChartData(
        labels=labels,
        datasets=datasets,
        options={
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": "Portfolio Value ($)"
                    }
                },
                "x": {
                    "title": {
                        "display": True,
                        "text": "Time"
                    }
                }
            }
        }
    )


def _create_risk_analysis_chart(simulations: List[SimulationResponse]) -> ChartData:
    """Create risk analysis chart data."""
    if not simulations:
        return ChartData(labels=[], datasets=[])
    
    # Collect risk metrics from all simulations
    strategy_names = []
    probabilities = []
    volatilities = []
    sharpe_ratios = []
    
    for sim in simulations:
        for strategy in sim.recommended_strategies:
            strategy_names.append(strategy.strategy_name)
            probabilities.append(strategy.probability_of_success * 100)
            volatilities.append(strategy.expected_volatility * 100)
            sharpe_ratios.append(strategy.sharpe_ratio)
    
    return ChartData(
        labels=strategy_names,
        datasets=[
            {
                "label": "Success Probability (%)",
                "data": probabilities,
                "backgroundColor": "rgba(54, 162, 235, 0.6)",
                "borderColor": "rgba(54, 162, 235, 1)",
                "borderWidth": 1
            },
            {
                "label": "Volatility (%)",
                "data": volatilities,
                "backgroundColor": "rgba(255, 99, 132, 0.6)",
                "borderColor": "rgba(255, 99, 132, 1)",
                "borderWidth": 1
            }
        ],
        options={
            "responsive": True,
            "scales": {
                "y": {
                    "beginAtZero": True,
                    "title": {
                        "display": True,
                        "text": "Percentage (%)"
                    }
                }
            }
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
