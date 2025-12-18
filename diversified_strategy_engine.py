"""
Diversified Investment Strategy Engine for Multiple Asset Classes
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


class DiversifiedStrategyEngine:
    """Engine for generating diversified investment strategies across multiple asset classes."""
    
    def __init__(self):
        self.monte_carlo = MonteCarloSimulator()
        self.asset_class_weights = {
            'stocks': {'low': 0.3, 'medium': 0.5, 'high': 0.7},
            'bonds': {'low': 0.5, 'medium': 0.3, 'high': 0.1},
            'real_estate': {'low': 0.1, 'medium': 0.15, 'high': 0.2},
            'commodities': {'low': 0.05, 'medium': 0.05, 'high': 0.1},
            'cryptocurrency': {'low': 0.0, 'medium': 0.0, 'high': 0.1},
            'alternative_investments': {'low': 0.05, 'medium': 0.0, 'high': 0.0}
        }
    
    def generate_diversified_strategies(
        self,
        instruments_by_class: Dict[str, List[FinancialInstrument]],
        user_input: UserInput,
        simulation_count: int = 100
    ) -> List[InvestmentStrategy]:
        """
        Generate diversified investment strategies across multiple asset classes.
        
        Args:
            instruments_by_class: Instruments organized by asset class
            user_input: User financial requirements
            simulation_count: Number of Monte Carlo simulations
            
        Returns:
            List of diversified investment strategies
        """
        try:
            strategies = []
            
            # Generate different diversification approaches
            strategies.extend(self._generate_risk_based_strategies(instruments_by_class, user_input))
            strategies.extend(self._generate_goal_based_strategies(instruments_by_class, user_input))
            strategies.extend(self._generate_balanced_strategies(instruments_by_class, user_input))
            
            # Run Monte Carlo simulations for each strategy
            strategies_with_simulations = []
            for strategy in strategies:
                try:
                    # Get all instruments for this strategy
                    all_instruments = []
                    for asset_class, instruments in instruments_by_class.items():
                        for symbol in strategy['allocation']:
                            matching_instruments = [inst for inst in instruments if inst.symbol == symbol]
                            all_instruments.extend(matching_instruments)
                    
                    if not all_instruments:
                        continue
                    
                    monte_carlo_result = self.monte_carlo.simulate_portfolio(
                        all_instruments, strategy['allocation'], user_input, simulation_count
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
            
            return strategies_with_simulations[:6]  # Return top 6 strategies
            
        except Exception as e:
            logger.error(f"Error generating diversified strategies: {str(e)}")
            return []
    
    def _generate_risk_based_strategies(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]], 
        user_input: UserInput
    ) -> List[Dict[str, Any]]:
        """Generate strategies based on risk appetite."""
        strategies = []
        risk_level = user_input.risk_appetite.value
        
        # Conservative Strategy
        if risk_level in ['low', 'medium']:
            allocation = self._create_allocation_by_risk(
                instruments_by_class, 'low', user_input
            )
            if allocation:
                strategies.append({
                    'name': 'Conservative Diversified',
                    'description': 'Low-risk portfolio with bonds, REITs, and stable stocks',
                    'allocation': allocation,
                    'expected_return': 0.06,
                    'expected_volatility': 0.10,
                    'sharpe_ratio': 0.6,
                    'max_drawdown': 0.12,
                    'risk_level': RiskAppetite.LOW
                })
        
        # Balanced Strategy
        allocation = self._create_allocation_by_risk(
            instruments_by_class, 'medium', user_input
        )
        if allocation:
            strategies.append({
                'name': 'Balanced Diversified',
                'description': 'Balanced portfolio across all major asset classes',
                'allocation': allocation,
                'expected_return': 0.08,
                'expected_volatility': 0.14,
                'sharpe_ratio': 0.65,
                'max_drawdown': 0.18,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        # Aggressive Strategy
        if risk_level in ['medium', 'high']:
            allocation = self._create_allocation_by_risk(
                instruments_by_class, 'high', user_input
            )
            if allocation:
                strategies.append({
                    'name': 'Growth Diversified',
                    'description': 'Growth-focused portfolio with stocks, crypto, and alternatives',
                    'allocation': allocation,
                    'expected_return': 0.12,
                    'expected_volatility': 0.20,
                    'sharpe_ratio': 0.6,
                    'max_drawdown': 0.25,
                    'risk_level': RiskAppetite.HIGH
                })
        
        return strategies
    
    def _generate_goal_based_strategies(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]], 
        user_input: UserInput
    ) -> List[Dict[str, Any]]:
        """Generate strategies based on investment goals and timeline."""
        strategies = []
        
        # Calculate goal metrics
        target_amount = user_input.target_investment_amount
        current_income = user_input.current_income
        tenure = user_input.investment_tenure_years
        
        # Income vs Target ratio
        income_ratio = target_amount / current_income if current_income > 0 else 0
        
        # Short-term strategy (less than 5 years)
        if tenure < 5:
            allocation = self._create_short_term_allocation(instruments_by_class)
            if allocation:
                strategies.append({
                    'name': 'Short-term Focused',
                    'description': 'Conservative approach for short-term goals with bonds and stable assets',
                    'allocation': allocation,
                    'expected_return': 0.05,
                    'expected_volatility': 0.08,
                    'sharpe_ratio': 0.7,
                    'max_drawdown': 0.10,
                    'risk_level': RiskAppetite.LOW
                })
        
        # High-income strategy (if target is very high relative to income)
        if income_ratio > 3:
            allocation = self._create_high_income_allocation(instruments_by_class)
            if allocation:
                strategies.append({
                    'name': 'High-Income Strategy',
                    'description': 'Aggressive approach for ambitious goals with growth assets',
                    'allocation': allocation,
                    'expected_return': 0.15,
                    'expected_volatility': 0.25,
                    'sharpe_ratio': 0.6,
                    'max_drawdown': 0.30,
                    'risk_level': RiskAppetite.HIGH
                })
        
        # Long-term strategy (more than 10 years)
        if tenure > 10:
            allocation = self._create_long_term_allocation(instruments_by_class)
            if allocation:
                strategies.append({
                    'name': 'Long-term Wealth Building',
                    'description': 'Diversified approach for long-term wealth accumulation',
                    'allocation': allocation,
                    'expected_return': 0.10,
                    'expected_volatility': 0.16,
                    'sharpe_ratio': 0.7,
                    'max_drawdown': 0.20,
                    'risk_level': RiskAppetite.MEDIUM
                })
        
        return strategies
    
    def _generate_balanced_strategies(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]], 
        user_input: UserInput
    ) -> List[Dict[str, Any]]:
        """Generate balanced diversification strategies."""
        strategies = []
        
        # Equal weight across asset classes
        allocation = self._create_equal_weight_allocation(instruments_by_class)
        if allocation:
            strategies.append({
                'name': 'Equal Weight Diversified',
                'description': 'Equal allocation across all available asset classes',
                'allocation': allocation,
                'expected_return': 0.09,
                'expected_volatility': 0.15,
                'sharpe_ratio': 0.65,
                'max_drawdown': 0.18,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        # Risk parity approach
        allocation = self._create_risk_parity_allocation(instruments_by_class)
        if allocation:
            strategies.append({
                'name': 'Risk Parity Diversified',
                'description': 'Risk-weighted allocation for balanced risk exposure',
                'allocation': allocation,
                'expected_return': 0.08,
                'expected_volatility': 0.12,
                'sharpe_ratio': 0.7,
                'max_drawdown': 0.15,
                'risk_level': RiskAppetite.MEDIUM
            })
        
        return strategies
    
    def _create_allocation_by_risk(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]], 
        risk_level: str,
        user_input: UserInput
    ) -> Dict[str, float]:
        """Create allocation based on risk level."""
        allocation = {}
        total_weight = 0.0
        
        for asset_class, instruments in instruments_by_class.items():
            if not instruments:
                continue
                
            target_weight = self.asset_class_weights[asset_class][risk_level]
            if target_weight > 0:
                # Select best instrument from this asset class
                best_instrument = self._select_best_instrument(instruments, asset_class)
                if best_instrument:
                    allocation[best_instrument.symbol] = target_weight
                    total_weight += target_weight
        
        # Normalize weights to sum to 1.0
        if total_weight > 0:
            for symbol in allocation:
                allocation[symbol] /= total_weight
        
        return allocation
    
    def _create_short_term_allocation(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]]
    ) -> Dict[str, float]:
        """Create allocation for short-term goals."""
        allocation = {}
        
        # Focus on bonds and stable assets
        if 'bonds' in instruments_by_class and instruments_by_class['bonds']:
            best_bond = self._select_best_instrument(instruments_by_class['bonds'], 'bonds')
            if best_bond:
                allocation[best_bond.symbol] = 0.6
        
        if 'real_estate' in instruments_by_class and instruments_by_class['real_estate']:
            best_reit = self._select_best_instrument(instruments_by_class['real_estate'], 'real_estate')
            if best_reit:
                allocation[best_reit.symbol] = 0.3
        
        if 'stocks' in instruments_by_class and instruments_by_class['stocks']:
            best_stock = self._select_best_instrument(instruments_by_class['stocks'], 'stocks')
            if best_stock:
                allocation[best_stock.symbol] = 0.1
        
        return allocation
    
    def _create_high_income_allocation(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]]
    ) -> Dict[str, float]:
        """Create allocation for high-income goals."""
        allocation = {}
        
        # Focus on growth assets
        if 'stocks' in instruments_by_class and instruments_by_class['stocks']:
            best_stock = self._select_best_instrument(instruments_by_class['stocks'], 'stocks')
            if best_stock:
                allocation[best_stock.symbol] = 0.5
        
        if 'cryptocurrency' in instruments_by_class and instruments_by_class['cryptocurrency']:
            best_crypto = self._select_best_instrument(instruments_by_class['cryptocurrency'], 'cryptocurrency')
            if best_crypto:
                allocation[best_crypto.symbol] = 0.2
        
        if 'alternative_investments' in instruments_by_class and instruments_by_class['alternative_investments']:
            best_alt = self._select_best_instrument(instruments_by_class['alternative_investments'], 'alternative_investments')
            if best_alt:
                allocation[best_alt.symbol] = 0.2
        
        if 'real_estate' in instruments_by_class and instruments_by_class['real_estate']:
            best_reit = self._select_best_instrument(instruments_by_class['real_estate'], 'real_estate')
            if best_reit:
                allocation[best_reit.symbol] = 0.1
        
        return allocation
    
    def _create_long_term_allocation(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]]
    ) -> Dict[str, float]:
        """Create allocation for long-term goals."""
        allocation = {}
        
        # Diversified approach
        asset_weights = {
            'stocks': 0.4,
            'bonds': 0.2,
            'real_estate': 0.2,
            'commodities': 0.1,
            'alternative_investments': 0.1
        }
        
        for asset_class, weight in asset_weights.items():
            if asset_class in instruments_by_class and instruments_by_class[asset_class]:
                best_instrument = self._select_best_instrument(instruments_by_class[asset_class], asset_class)
                if best_instrument:
                    allocation[best_instrument.symbol] = weight
        
        return allocation
    
    def _create_equal_weight_allocation(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]]
    ) -> Dict[str, float]:
        """Create equal weight allocation across asset classes."""
        allocation = {}
        available_classes = [k for k, v in instruments_by_class.items() if v]
        
        if not available_classes:
            return allocation
        
        weight_per_class = 1.0 / len(available_classes)
        
        for asset_class in available_classes:
            best_instrument = self._select_best_instrument(instruments_by_class[asset_class], asset_class)
            if best_instrument:
                allocation[best_instrument.symbol] = weight_per_class
        
        # Ensure weights sum to 1.0
        total_weight = sum(allocation.values())
        if total_weight > 0:
            for symbol in allocation:
                allocation[symbol] /= total_weight
        
        return allocation
    
    def _create_risk_parity_allocation(
        self, 
        instruments_by_class: Dict[str, List[FinancialInstrument]]
    ) -> Dict[str, float]:
        """Create risk parity allocation."""
        allocation = {}
        
        # Simplified risk parity - weight by inverse volatility
        instruments_with_weights = []
        
        for asset_class, instruments in instruments_by_class.items():
            for instrument in instruments:
                # Use inverse volatility as weight (higher volatility = lower weight)
                weight = 1.0 / (instrument.volatility + 0.01)  # Add small constant to avoid division by zero
                instruments_with_weights.append((instrument, weight))
        
        if not instruments_with_weights:
            return allocation
        
        # Normalize weights
        total_weight = sum(weight for _, weight in instruments_with_weights)
        
        if total_weight > 0:
            for instrument, weight in instruments_with_weights:
                allocation[instrument.symbol] = weight / total_weight
        
        # Ensure weights sum to 1.0 (double-check)
        total_allocation = sum(allocation.values())
        if total_allocation > 0:
            for symbol in allocation:
                allocation[symbol] /= total_allocation
        
        return allocation
    
    def _select_best_instrument(
        self, 
        instruments: List[FinancialInstrument], 
        asset_class: str
    ) -> FinancialInstrument:
        """Select the best instrument from a list based on Sharpe ratio."""
        if not instruments:
            return None
        
        # Sort by Sharpe ratio (higher is better)
        sorted_instruments = sorted(instruments, key=lambda x: x.sharpe_ratio, reverse=True)
        return sorted_instruments[0]
    
    def get_asset_class_breakdown(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Get breakdown of allocation by asset class."""
        # This would need to be implemented based on your instrument classification
        # For now, return a simple breakdown
        breakdown = {
            'stocks': 0.0,
            'bonds': 0.0,
            'real_estate': 0.0,
            'commodities': 0.0,
            'cryptocurrency': 0.0,
            'alternative_investments': 0.0
        }
        
        # This is a simplified version - in practice, you'd need to map symbols to asset classes
        for symbol, weight in allocation.items():
            if symbol in ['SPY', 'QQQ', 'VTI', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']:
                breakdown['stocks'] += weight
            elif symbol in ['BND', 'TLT', 'IEF', 'SHY']:
                breakdown['bonds'] += weight
            elif symbol in ['VNQ', 'IYR', 'SCHH', 'O']:
                breakdown['real_estate'] += weight
            elif symbol in ['GLD', 'IAU', 'SLV']:
                breakdown['commodities'] += weight
            elif symbol in ['BTC-USD', 'ETH-USD', 'COIN']:
                breakdown['cryptocurrency'] += weight
            elif symbol in ['ARKK', 'TAN', 'ICLN']:
                breakdown['alternative_investments'] += weight
        
        return breakdown
