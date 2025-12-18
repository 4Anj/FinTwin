"""
Financial Digital Twin - FastAPI Server
---------------------------------------
This server simulates and recommends investment strategies using Monte Carlo
simulations on Yahoo Finance data. Limited to 1000 simulations to prevent overload.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Dict, Any
import uuid
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# -------------------------------------------
# Configuration
# -------------------------------------------
DEFAULT_TICKERS = ["SPY", "QQQ", "VTI", "BND"]
RISK_FREE_RATE = 0.02
TRADING_DAYS = 252
SIMULATION_LIMIT = 1000  # maximum to prevent overload

JOB_STORE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Financial Digital Twin API", version="1.0")

# -------------------------------------------
# Utility functions
# -------------------------------------------

def annualize_return(daily_mean: float) -> float:
    return (1 + daily_mean) ** TRADING_DAYS - 1


def annualize_vol(daily_std: float) -> float:
    return daily_std * np.sqrt(TRADING_DAYS)


def fetch_market_data(tickers: List[str]) -> pd.DataFrame:
    """Fetch historical data for tickers from Yahoo Finance"""
    data = yf.download(tickers, period="5y", interval="1d", progress=False)["Adj Close"]
    data.dropna(inplace=True)
    return data


def compute_stats(data: pd.DataFrame) -> pd.DataFrame:
    """Compute annualized returns and volatility"""
    returns = np.log(data / data.shift(1)).dropna()
    mean_daily = returns.mean()
    std_daily = returns.std()
    stats = pd.DataFrame({
        "annual_return": (1 + mean_daily) ** TRADING_DAYS - 1,
        "annual_volatility": std_daily * np.sqrt(TRADING_DAYS)
    })
    return stats


def monte_carlo_simulation(mu, sigma, start_value, years, n_simulations):
    """Simulate future portfolio values using Geometric Brownian Motion."""
    dt = 1 / TRADING_DAYS
    n_days = int(TRADING_DAYS * years)
    results = np.zeros((n_days, n_simulations))
    results[0] = start_value

    for t in range(1, n_days):
        rand = np.random.normal(0, 1, n_simulations)
        results[t] = results[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)

    return results


def portfolio_metrics(simulated_values: np.ndarray, target_amount: float):
    """Compute metrics like probability of success, Sharpe ratio, etc."""
    final_values = simulated_values[-1]
    mean_final = np.mean(final_values)
    prob_success = np.mean(final_values >= target_amount)
    worst_case = np.percentile(final_values, 5)
    volatility = np.std(final_values)
    sharpe = (mean_final - target_amount) / (volatility + 1e-6)

    return {
        "expected_final_value": float(mean_final),
        "probability_of_success": float(prob_success),
        "worst_case_5pct": float(worst_case),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe)
    }


def suggest_strategies(stats: pd.DataFrame, risk_appetite: str) -> Dict[str, Any]:
    """Suggest portfolio weights and descriptions based on risk appetite."""
    if risk_appetite == "low":
        weights = np.array([0.2, 0.2, 0.2, 0.4])  # more bonds
    elif risk_appetite == "high":
        weights = np.array([0.4, 0.3, 0.3, 0.0])  # more equities
    else:
        weights = np.array([0.3, 0.25, 0.25, 0.2])  # balanced

    portfolio_return = np.dot(stats["annual_return"], weights)
    portfolio_vol = np.dot(stats["annual_volatility"], weights)

    return {
        "weights": dict(zip(stats.index, weights.round(2))),
        "portfolio_return": float(portfolio_return),
        "portfolio_volatility": float(portfolio_vol)
    }

# -------------------------------------------
# Pydantic Models
# -------------------------------------------

class SubmitRequest(BaseModel):
    current_income: float = Field(..., gt=0)
    target_amount: float = Field(..., gt=0)
    tenure_years: conint(gt=0)
    tickers: Optional[List[str]] = None
    monthly_contribution: Optional[float] = Field(0, ge=0)
    risk_appetite: Optional[str] = Field("medium", regex="^(low|medium|high)$")
    n_simulations: conint(gt=0, le=SIMULATION_LIMIT) = SIMULATION_LIMIT


class SimulationResult(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# -------------------------------------------
# Simulation Logic
# -------------------------------------------

def run_simulation(job_id: str, payload: SubmitRequest):
    """Background simulation job."""
    try:
        tickers = payload.tickers or DEFAULT_TICKERS
        n_sim = min(payload.n_simulations, SIMULATION_LIMIT)

        data = fetch_market_data(tickers)
        stats = compute_stats(data)

        strategy = suggest_strategies(stats, payload.risk_appetite)

        mu = strategy["portfolio_return"]
        sigma = strategy["portfolio_volatility"]

        results = monte_carlo_simulation(mu, sigma, payload.current_income, payload.tenure_years, n_sim)
        metrics = portfolio_metrics(results, payload.target_amount)

        JOB_STORE[job_id]["status"] = "completed"
        JOB_STORE[job_id]["result"] = {
            "strategy": strategy,
            "metrics": metrics
        }

    except Exception as e:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["error"] = str(e)


# -------------------------------------------
# API Endpoints
# -------------------------------------------

@app.post("/api/submit", response_model=SimulationResult)
async def submit_simulation(payload: SubmitRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {
        "created_at": datetime.datetime.utcnow(),
        "status": "running",
        "payload": payload.dict()
    }

    background_tasks.add_task(run_simulation, job_id, payload)
    return SimulationResult(job_id=job_id, status="running")


@app.get("/api/simulations/{job_id}", response_model=SimulationResult)
async def get_simulation(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return SimulationResult(job_id=job_id, status=job["status"], result=job.get("result"), error=job.get("error"))


@app.get("/api/strategies")
async def get_strategies(risk: str = "medium"):
    dummy_data = pd.DataFrame({
        "annual_return": [0.08, 0.1, 0.09, 0.03],
        "annual_volatility": [0.15, 0.18, 0.16, 0.05]
    }, index=DEFAULT_TICKERS)
    return suggest_strategies(dummy_data, risk)

# -------------------------------------------
# Run
# -------------------------------------------
# To run: uvicorn financial_digital_twin_fastapi:app --reload --port 8000
