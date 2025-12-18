"""
Monte Carlo simulation engine for financial modeling.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy import stats
import logging
from models import MonteCarloResult, FinancialInstrument, UserInput

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """Monte Carlo simulation engine for financial modeling."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            random_seed: Random seed for reproducible results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_portfolio(
        self,
        instruments: List[FinancialInstrument],
        allocation: Dict[str, float],
        user_input: UserInput,
        num_simulations: int = 100,
        num_paths_to_return: int = 10
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a portfolio.
        
        Args:
            instruments: List of financial instruments
            allocation: Dictionary mapping symbols to allocation percentages
            user_input: User financial input
            num_simulations: Number of simulation runs
            num_paths_to_return: Number of sample paths to return
            
        Returns:
            MonteCarloResult object with simulation results
        """
        try:
            # Validate allocation
            if abs(sum(allocation.values()) - 1.0) > 1e-6:
                raise ValueError("Allocation percentages must sum to 1.0")
            
            # Create instrument lookup
            instrument_map = {inst.symbol: inst for inst in instruments}
            
            # Calculate portfolio-level metrics
            portfolio_returns, portfolio_volatility = self._calculate_portfolio_metrics(
                instruments, allocation, instrument_map
            )
            
            # Generate random returns for simulation
            random_returns = self._generate_random_returns(
                portfolio_returns, portfolio_volatility, 
                user_input.investment_tenure_years, num_simulations
            )
            
            # Calculate portfolio values over time
            portfolio_values = self._calculate_portfolio_values(
                random_returns, user_input.initial_investment, user_input.current_income
            )
            
            # Calculate final values
            final_values = portfolio_values[:, -1]
            
            # Calculate statistics
            mean_final_value = np.mean(final_values)
            median_final_value = np.median(final_values)
            std_final_value = np.std(final_values)
            
            # Calculate probability of success
            target = user_input.target_investment_amount
            probability_of_success = np.mean(final_values >= target)
            
            # Calculate percentiles
            worst_case_5th = np.percentile(final_values, 5)
            best_case_95th = np.percentile(final_values, 95)
            
            # Sample paths for visualization
            sample_paths = self._sample_paths(portfolio_values, num_paths_to_return)
            
            return MonteCarloResult(
                mean_final_value=float(mean_final_value),
                median_final_value=float(median_final_value),
                std_final_value=float(std_final_value),
                probability_of_success=float(probability_of_success),
                worst_case_5th_percentile=float(worst_case_5th),
                best_case_95th_percentile=float(best_case_95th),
                simulation_paths=sample_paths.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            raise
    
    def _calculate_portfolio_metrics(
        self, 
        instruments: List[FinancialInstrument], 
        allocation: Dict[str, float],
        instrument_map: Dict[str, FinancialInstrument]
    ) -> Tuple[float, float]:
        """Calculate portfolio-level expected return and volatility."""
        portfolio_return = 0.0
        portfolio_variance = 0.0
        
        # Calculate weighted average return
        for symbol, weight in allocation.items():
            if symbol in instrument_map:
                instrument = instrument_map[symbol]
                # Convert annual volatility to daily
                daily_vol = instrument.volatility / np.sqrt(252)
                # Estimate daily return from annual return (simplified)
                daily_return = (1 + instrument.sharpe_ratio * daily_vol) ** (1/252) - 1
                portfolio_return += weight * daily_return
        
        # Calculate portfolio variance (simplified - assumes independence)
        for symbol, weight in allocation.items():
            if symbol in instrument_map:
                instrument = instrument_map[symbol]
                daily_vol = instrument.volatility / np.sqrt(252)
                portfolio_variance += (weight * daily_vol) ** 2
        
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # Annualize
        
        return portfolio_return * 252, portfolio_volatility  # Annualize return
    
    def _generate_random_returns(
        self, 
        expected_return: float, 
        volatility: float, 
        years: int, 
        num_simulations: int
    ) -> np.ndarray:
        """Generate random returns using geometric Brownian motion."""
        # Convert to daily parameters
        dt = 1/252  # Daily time step
        mu = expected_return
        sigma = volatility
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (num_simulations, years * 252))
        
        # Calculate daily returns using geometric Brownian motion
        daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
        
        return daily_returns
    
    def _calculate_portfolio_values(
        self, 
        daily_returns: np.ndarray, 
        initial_investment: float,
        annual_income: float
    ) -> np.ndarray:
        """Calculate portfolio values over time with regular contributions."""
        num_simulations, num_days = daily_returns.shape
        portfolio_values = np.zeros((num_simulations, num_days + 1))
        
        # Initial investment
        portfolio_values[:, 0] = initial_investment
        
        # Monthly contribution
        monthly_contribution = annual_income / 12
        
        for day in range(num_days):
            # Apply return first
            portfolio_values[:, day + 1] = portfolio_values[:, day] * (1 + daily_returns[:, day])
            
            # Add monthly contribution on the 1st of each month (every 30 days)
            if day % 30 == 0 and day > 0:
                portfolio_values[:, day + 1] += monthly_contribution
        
        return portfolio_values
    
    def _sample_paths(self, portfolio_values: np.ndarray, num_paths: int) -> np.ndarray:
        """Sample random paths for visualization."""
        num_simulations = portfolio_values.shape[0]
        if num_simulations <= num_paths:
            return portfolio_values
        else:
            indices = np.random.choice(num_simulations, num_paths, replace=False)
            return portfolio_values[indices]
    
    def simulate_multiple_strategies(
        self,
        instruments: List[FinancialInstrument],
        strategies: List[Dict[str, float]],
        user_input: UserInput,
        num_simulations: int = 100
    ) -> List[MonteCarloResult]:
        """
        Run Monte Carlo simulation for multiple strategies.
        
        Args:
            instruments: List of financial instruments
            strategies: List of allocation dictionaries
            user_input: User financial input
            num_simulations: Number of simulation runs
            
        Returns:
            List of MonteCarloResult objects
        """
        results = []
        
        for i, allocation in enumerate(strategies):
            try:
                result = self.simulate_portfolio(
                    instruments, allocation, user_input, num_simulations
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error simulating strategy {i}: {str(e)}")
                # Create a default result for failed strategies
                results.append(MonteCarloResult(
                    mean_final_value=0.0,
                    median_final_value=0.0,
                    std_final_value=0.0,
                    probability_of_success=0.0,
                    worst_case_5th_percentile=0.0,
                    best_case_95th_percentile=0.0,
                    simulation_paths=[]
                ))
        
        return results
    
    def calculate_risk_metrics(
        self, 
        final_values: np.ndarray, 
        target_value: float
    ) -> Dict[str, float]:
        """
        Calculate additional risk metrics.
        
        Args:
            final_values: Array of final portfolio values
            target_value: Target investment amount
            
        Returns:
            Dictionary of risk metrics
        """
        metrics = {}
        
        # Value at Risk (VaR) - 5% and 1%
        metrics['var_5pct'] = float(np.percentile(final_values, 5))
        metrics['var_1pct'] = float(np.percentile(final_values, 1))
        
        # Expected Shortfall (Conditional VaR)
        var_5pct = metrics['var_5pct']
        metrics['expected_shortfall_5pct'] = float(
            np.mean(final_values[final_values <= var_5pct])
        )
        
        # Probability of loss
        metrics['probability_of_loss'] = float(np.mean(final_values < 0))
        
        # Probability of achieving target
        metrics['probability_of_target'] = float(np.mean(final_values >= target_value))
        
        # Skewness and Kurtosis
        metrics['skewness'] = float(stats.skew(final_values))
        metrics['kurtosis'] = float(stats.kurtosis(final_values))
        
        return metrics
