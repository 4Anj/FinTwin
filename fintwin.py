"""
Financial Digital Twin - FastAPI server
File: financial_digital_twin_fastapi.py

Features:
- Fetches market data using yfinance
- Runs Monte Carlo simulations (GBM) for portfolios and individual tickers
- Suggests strategies by risk appetite (low, medium, high) with allocations
- Exposes REST endpoints for submitting user input, retrieving simulation results,
  and viewing suggested strategies & risk profiles
- Modular and documented for easy extension (e.g., ML models)
- Limited to maximum 1000 simulations per request to prevent overload

Run:
1. pip install fastapi uvicorn yfinance numpy pandas scipy pydantic python-multipart
2. uvicorn financial_digital_twin_fastapi:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, conint
from typing import List, Optional, Dict, Any
import uuid
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import asyncio

# ----------------------------- Configuration -----------------------------
DEFAULT_TICKERS = ["SPY", "QQQ", "VTI", "BND"]
RISK_FREE_RATE = 0.02  # annual
SIMULATION_COUNT = 1000  # maximum limit to avoid overload
TRADING_DAYS = 252

# In-memory job store (demo only)
JOB_STORE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Financial Digital Twin API", version="0.2")


# ----------------------------- Utility Functions -----------------------------
def annualize_return(daily_mean: float) -> float:
    return (1 + daily_mean) ** TRADING_DAYS - 1


def annualize_vol(daily_std: float) -> float:
    return daily_std * np.sqrt(TRADING_DAYS)


# ----------------------------- Pydantic Models -----------------------------
class SubmitRequest(BaseModel):
    current_income: float = Field(..., gt=0, description="Annual gross income")
    target_amount: float = Field(..., gt=0, description="Target investment amount")
    tenure_years: conint(gt=0) = Field(..., description="Investment tenure in years")
    tickers: Optional[List[str]] = None
    monthly_contribution: Optional[float] = Field(None, gt=0)
    risk_appetite: Optional[str] = Field("medium", description="low | medium | high")
    n_simulations: Optional[conint(gt=0, le=1000)] = Field(
        SIMULATION_COUNT, description="Monte Carlo simulations (max 1000)"
    )


class StrategySummary(BaseModel):
    name: str
    tickers: List[str]
    allocations: List[float]
    projected_mean_final: float
    probability_reaching_target: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    worst_case_percentile: float


# ----------------------------- Market Data -----------------------------
async def fetch_history(tickers: List[str], period: str = "5y") -> pd.DataFrame:
    """Fetch historical adjusted close prices using yfinance."""
    loop = asyncio.get_event_loop()

    def _download():
        data = yf.download(tickers, period=period, progress=False, threads=True, auto_adjust=True)
        if isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
            adj = data["Adj Close"]
        else:
            adj = data
        return adj

    df = await loop.run_in_executor(None, _download)

    if isinstance(df, pd.Series):
        df = df.to_frame(df.name)
    df = df.dropna(how="all")
    df = df[tickers]
    return df


def compute_return_stats(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns, mean and std for each asset."""
    returns = np.log(price_df / price_df.shift(1)).dropna()
    stats = pd.DataFrame(index=price_df.columns)
    stats["daily_mean"] = returns.mean()
    stats["daily_std"] = returns.std()
    stats["annual_mean"] = stats["daily_mean"].apply(annualize_return)
    stats["annual_vol"] = stats["daily_std"].apply(annualize_vol)
    return stats


# ----------------------------- Monte Carlo Simulation -----------------------------
def simulate_gbm(initial_price: float, mu: float, sigma: float, days: int, n_sims: int) -> np.ndarray:
    """Simulate Geometric Brownian Motion paths."""
    dt = 1 / TRADING_DAYS
    rand = np.random.normal(0, 1, size=(n_sims, days))
    increments = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand
    log_paths = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(increments, axis=1)], axis=1)
    price_paths = initial_price * np.exp(log_paths)
    return price_paths


def simulate_portfolio(tickers: List[str], allocations: List[float], prices: pd.DataFrame, stats: pd.DataFrame,
                       tenure_years: int, monthly_contribution: float, n_sims: int) -> Dict[str, Any]:
    """Simulate portfolio growth with monthly contributions."""
    days = int(tenure_years * TRADING_DAYS)
    n_assets = len(tickers)
    asset_final_values = np.zeros((n_sims, n_assets))

    for i, t in enumerate(tickers):
        init_price = prices[t].iloc[-1]
        daily_mu = stats.loc[t, "daily_mean"]
        daily_sigma = stats.loc[t, "daily_std"]
        paths = simulate_gbm(init_price, daily_mu, daily_sigma, days, n_sims)
        asset_final_values[:, i] = paths[:, -1]

    asset_returns = (asset_final_values / prices.iloc[-1].values) - 1
    portfolio_returns = np.dot(asset_returns, np.array(allocations))

    monthly_rate_per_sim = (1 + portfolio_returns) ** (1 / tenure_years) - 1
    monthly_rate_per_sim = monthly_rate_per_sim / 12.0

    months = tenure_years * 12
    fv_contrib = monthly_contribution * (((1 + monthly_rate_per_sim) ** months - 1) / monthly_rate_per_sim)
    zero_mask = np.isclose(monthly_rate_per_sim, 0)
    fv_contrib[zero_mask] = monthly_contribution * months

    final_values = fv_contrib
    return {"final_values": final_values, "portfolio_returns": portfolio_returns, "n_sims": n_sims}


# ----------------------------- Strategies -----------------------------
def default_strategies() -> List[Dict[str, Any]]:
    return [
        {"name": "Conservative", "tickers": ["BND", "VTI"], "allocations": [0.7, 0.3]},
        {"name": "Balanced", "tickers": ["BND", "VTI", "SPY"], "allocations": [0.4, 0.4, 0.2]},
        {"name": "Aggressive", "tickers": ["VTI", "QQQ"], "allocations": [0.5, 0.5]},
    ]


def select_strategies_by_risk(risk: str) -> List[Dict[str, Any]]:
    s = default_strategies()
    if risk == "low":
        return [s[0], s[1]]
    elif risk == "high":
        return [s[2]]
    else:
        return [s[1], s[2]]


# ----------------------------- Metrics -----------------------------
def summarize_simulation(final_values: np.ndarray, portfolio_returns: np.ndarray) -> Dict[str, Any]:
    mean_final = float(np.mean(final_values))
    ann_return_mean = float(np.mean((1 + portfolio_returns) - 1))
    ann_vol = float(np.std((1 + portfolio_returns) - 1))
    sharpe = (ann_return_mean - RISK_FREE_RATE) / ann_vol if ann_vol != 0 else 0.0
    worst_case = float(np.percentile(final_values, 5))
    return {
        "mean_final": mean_final,
        "annualized_return_mean": ann_return_mean,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "worst_case_5pct": worst_case,
    }


# ----------------------------- Simulation Job -----------------------------
async def run_simulation_job(job_id: str, payload: SubmitRequest):
    JOB_STORE[job_id]["status"] = "running"
    try:
        tickers = payload.tickers if payload.tickers else DEFAULT_TICKERS
        monthly_contribution = payload.monthly_contribution or (payload.current_income / 12.0) * 0.2

        prices = await fetch_history(tickers, period="5y")
        stats = compute_return_stats(prices)
        strategies = select_strategies_by_risk(payload.risk_appetite.lower())

        strategy_summaries: List[StrategySummary] = []

        for s in strategies:
            allocations = np.array(s["allocations"]) / np.sum(s["allocations"])
            sim = simulate_portfolio(
                s["tickers"], allocations.tolist(), prices[s["tickers"]], stats,
                payload.tenure_years, monthly_contribution, payload.n_simulations
            )
            final_values = sim["final_values"]
            portfolio_returns = sim["portfolio_returns"]
            prob = float(np.mean(final_values >= payload.target_amount))

            metrics = summarize_simulation(final_values, portfolio_returns)
            strategy_summary = StrategySummary(
                name=s["name"],
                tickers=s["tickers"],
                allocations=allocations.tolist(),
                projected_mean_final=float(metrics["mean_final"]),
                probability_reaching_target=prob,
                annualized_return=float(metrics["annualized_return_mean"]),
                annualized_volatility=float(metrics["annualized_volatility"]),
                sharpe_ratio=float(metrics["sharpe_ratio"]),
                worst_case_percentile=float(metrics["worst_case_5pct"]),
            )
            strategy_summaries.append(strategy_summary)

        JOB_STORE[job_id]["status"] = "finished"
        JOB_STORE[job_id]["result"] = {
            "strategies": [s.dict() for s in strategy_summaries],
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "params": payload.dict(),
        }
    except Exception as e:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["error"] = str(e)


# ----------------------------- API Endpoints -----------------------------
@app.post("/api/submit")
async def submit(payload: SubmitRequest, background_tasks: BackgroundTasks):
    """Submit user inputs and trigger Monte Carlo simulations."""
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "pending",
        "params": payload.dict(),
    }
    background_tasks.add_task(asyncio.ensure_future, run_simulation_job(job_id, payload))
    return {"job_id": job_id, "status": "started"}


@app.get("/api/simulations/{job_id}")
async def get_simulation(job_id: str):
    entry = JOB_STORE.get(job_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return entry


@app.get("/api/strategies")
async def get_strategies(risk: Optional[str] = "medium"):
    return {"strategies": select_strategies_by_risk(risk.lower())}


@app.get("/")
async def root():
    return {"msg": "Financial Digital Twin API - POST /api/submit to run simulations"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
