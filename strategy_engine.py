"""
Investment strategy recommendation engine for the Financial Digital Twin.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import logging
from models import (
    FinancialInstrument, InvestmentStrategy, UserInput, 
    RiskAppetite, MonteCarloResult
)
from monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Engine for generating investment strategy recommendations."""
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
    
    def generate_strategies(
        self,
        instruments: List[FinancialInstrument],
        user_input: UserInput,
        simulation_count: int = 100
    ) -> List[InvestmentStrategy]:
        """
        Generate investment strategies based on user input and available instruments.
        
        Args:
            instruments: Available financial instruments
            user_input: User financial requirements
            simulation_count: Number of Monte Carlo simulations
            
        Returns:
            List of recommended investment strategies
        """
        try:
            strategies = []
            
            # Generate strategies based on risk appetite
            if user_input.risk_appetite == RiskAppetite.LOW:
                strategies.extend(self._generate_conservative_strategies(instruments))
            elif user_input.risk_appetite == RiskAppetite.MEDIUM:
                strategies.extend(self._generate_balanced_strategies(instruments))
            else:  # HIGH risk
                strategies.extend(self._generate_aggressive_strategies(instruments))
            
            # Add diversified strategies
            strategies.extend(self._generate_diversified_strategies(instruments))
            
            # Run Monte Carlo simulations for each strategy
            strategies_with_simulations = []
            for strategy in strategies:
                try:
                    monte_carlo_result = self.monte_carlo.simulate_portfolio(
                        instruments, strategy['allocation'], user_input, simulation_count
                    )
                    
                    # Create InvestmentStrategy object
                    investment_strategy = InvestmentStrategy(
                        strategy_name=strategy['name'],
                        description=strategy['description'],
                        recommended_allocation=strategy['allocation'],
                        expected_annual_return=strategy['expected_return'],
                        expected_volatility=strategy['expected_volatility'],
                        sharpe_ratio=strategy['sharpe_ratio'],
                        max_drawdown=strategy['max_drawdown'],
                        probability_of_success=monte_carlo_result.probability_of_success,
                        monte_carlo_result=monte_carlo_result,
                        risk_level=strategy['risk_level']
                    )
                    
                    strategies_with_simulations.append(investment_strategy)
                    
                except Exception as e:
                    logger.error(f"Error simulating strategy {strategy['name']}: {str(e)}")
                    continue
            
            # Sort by probability of success
            strategies_with_simulations.sort(
                key=lambda x: x.probability_of_success, reverse=True
            )
            
            return strategies_with_simulations[:5]  # Return top 5 strategies
            
        except Exception as e:
            logger.error(f"Error generating strategies: {str(e)}")
            return []
    
    def _generate_conservative_strategies(self, instruments: List[FinancialInstrument]) -> List[Dict[str, Any]]:
        """Generate conservative investment strategies."""
        strategies = []
        
        if not instruments:
            return strategies
        
        # Find bond ETFs and low-volatility instruments
        bond_etfs = [inst for inst in instruments if any(keyword in inst.name.lower() 
                     for keyword in ['bond', 'treasury', 'fixed income'])]
        low_vol_etfs = [inst for inst in instruments if inst.volatility < 0.15]
        
        if bond_etfs:
            # Conservative: 80% bonds, 20% low-vol stocks
            allocation = {bond_etfs[0].symbol: 0.8}
            if low_vol_etfs:
                allocation[low_vol_etfs[0].symbol] = 0.2
            else:
                allocation[instruments[0].symbol] = 0.2
        else:
            # If no bonds, use lowest volatility instruments
            if low_vol_etfs:
                allocation = {low_vol_etfs[0].symbol: 1.0}
            else:
                allocation = {instruments[0].symbol: 1.0}
        
        strategies.append({
            'name': 'Conservative Income',
            'description': 'Focus on bonds and low-volatility instruments for steady income',
            'allocation': allocation,
            'expected_return': 0.05,
            'expected_volatility': 0.08,
            'sharpe_ratio': 0.6,
            'max_drawdown': 0.05,
            'risk_level': RiskAppetite.LOW
        })
        
        return strategies
    
    def _generate_balanced_strategies(self, instruments: List[FinancialInstrument]) -> List[Dict[str, Any]]:
        """Generate balanced investment strategies."""
        strategies = []
        
        if not instruments:
            return strategies
        
        # 60/40 balanced portfolio
        if len(instruments) >= 2:
            strategies.append({
                'name': 'Balanced Growth',
                'description': '60% stocks, 40% bonds for balanced growth and income',
                'allocation': {
                    instruments[0].symbol: 0.6,
                    instruments[1].symbol: 0.4
                },
                'expected_return': 0.08,
                'expected_volatility': 0.12,
                'sharpe_ratio': 0.7,
                'max_drawdown': 0.15,
                'risk_level': RiskAppetite.MEDIUM
            })
        else:
            # Single instrument balanced approach
            strategies.append({
                'name': 'Single Asset Balanced',
                'description': 'Focused approach with single high-quality instrument',
                'allocation': {
                    instruments[0].symbol: 1.0
                },
                'expected_return': 0.08,
                'expected_volatility': 0.12,
                'sharpe_ratio': 0.7,
                'max_drawdown': 0.15,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        # Core satellite approach
        if len(instruments) >= 3:
            strategies.append({
                'name': 'Core Satellite',
                'description': 'Core holding with satellite positions for diversification',
                'allocation': {
                    instruments[0].symbol: 0.5,  # Core
                    instruments[1].symbol: 0.3,  # Satellite 1
                    instruments[2].symbol: 0.2   # Satellite 2
                },
                'expected_return': 0.09,
                'expected_volatility': 0.14,
                'sharpe_ratio': 0.65,
                'max_drawdown': 0.18,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        return strategies
    
    def _generate_aggressive_strategies(self, instruments: List[FinancialInstrument]) -> List[Dict[str, Any]]:
        """Generate aggressive investment strategies."""
        strategies = []
        
        if not instruments:
            return strategies
        
        # Growth-focused portfolio
        if len(instruments) >= 2:
            strategies.append({
                'name': 'Growth Focus',
                'description': 'High allocation to growth stocks and ETFs',
                'allocation': {
                    instruments[0].symbol: 0.8,
                    instruments[1].symbol: 0.2
                },
                'expected_return': 0.12,
                'expected_volatility': 0.20,
                'sharpe_ratio': 0.6,
                'max_drawdown': 0.25,
                'risk_level': RiskAppetite.HIGH
            })
        else:
            # Single instrument aggressive approach
            strategies.append({
                'name': 'Single Asset Growth',
                'description': 'Focused growth approach with single high-potential instrument',
                'allocation': {
                    instruments[0].symbol: 1.0
                },
                'expected_return': 0.12,
                'expected_volatility': 0.20,
                'sharpe_ratio': 0.6,
                'max_drawdown': 0.25,
                'risk_level': RiskAppetite.HIGH
            })
        
        # Momentum strategy
        if len(instruments) >= 3:
            strategies.append({
                'name': 'Momentum Strategy',
                'description': 'Focus on high-performing instruments with momentum',
                'allocation': {
                    instruments[0].symbol: 0.4,
                    instruments[1].symbol: 0.4,
                    instruments[2].symbol: 0.2
                },
                'expected_return': 0.14,
                'expected_volatility': 0.22,
                'sharpe_ratio': 0.65,
                'max_drawdown': 0.30,
                'risk_level': RiskAppetite.HIGH
            })
        
        return strategies
    
    def _generate_diversified_strategies(self, instruments: List[FinancialInstrument]) -> List[Dict[str, Any]]:
        """Generate diversified investment strategies."""
        strategies = []
        
        # Equal weight portfolio
        if len(instruments) >= 3:
            # Take top 5 instruments and ensure weights sum to 1.0
            selected_instruments = instruments[:5]
            weight_per_instrument = 1.0 / len(selected_instruments)
            allocation = {inst.symbol: weight_per_instrument for inst in selected_instruments}
            
            strategies.append({
                'name': 'Equal Weight Diversified',
                'description': 'Equal allocation across multiple asset classes',
                'allocation': allocation,
                'expected_return': 0.10,
                'expected_volatility': 0.16,
                'sharpe_ratio': 0.65,
                'max_drawdown': 0.20,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        # Risk parity approach (simplified)
        if len(instruments) >= 4:
            # Weight by inverse volatility
            weights = []
            selected_instruments = instruments[:4]
            for inst in selected_instruments:
                weight = 1.0 / (inst.volatility + 0.01)  # Add small constant to avoid division by zero
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w/total_weight for w in weights]
                
                allocation = {selected_instruments[i].symbol: weights[i] for i in range(len(selected_instruments))}
                
                strategies.append({
                    'name': 'Risk Parity',
                    'description': 'Risk-weighted allocation for balanced risk exposure',
                    'allocation': allocation,
                    'expected_return': 0.09,
                    'expected_volatility': 0.13,
                    'sharpe_ratio': 0.7,
                    'max_drawdown': 0.16,
                    'risk_level': RiskAppetite.MEDIUM
                })
        
        return strategies
    
    def calculate_portfolio_metrics(
        self, 
        instruments: List[FinancialInstrument], 
        allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level metrics.
        
        Args:
            instruments: List of financial instruments
            allocation: Asset allocation dictionary
            
        Returns:
            Dictionary of portfolio metrics
        """
        instrument_map = {inst.symbol: inst for inst in instruments}
        
        portfolio_return = 0.0
        portfolio_volatility = 0.0
        portfolio_sharpe = 0.0
        portfolio_max_dd = 0.0
        
        for symbol, weight in allocation.items():
            if symbol in instrument_map:
                inst = instrument_map[symbol]
                portfolio_return += weight * (inst.sharpe_ratio * inst.volatility + 0.02)  # Rough annual return
                portfolio_volatility += (weight * inst.volatility) ** 2
                portfolio_sharpe += weight * inst.sharpe_ratio
                portfolio_max_dd = max(portfolio_max_dd, inst.max_drawdown)
        
        portfolio_volatility = np.sqrt(portfolio_volatility)
        
        return {
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'sharpe_ratio': portfolio_sharpe,
            'max_drawdown': portfolio_max_dd
        }
