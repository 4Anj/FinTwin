"""
AI Insights Service for Financial Digital Twin
Uses free AI APIs to provide grounded financial advice and insights.
"""

import requests
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class AIInsightsService:
    """Service for generating AI-powered financial insights using free APIs."""
    
    def __init__(self):
        self.huggingface_api_url = "https://api-inference.huggingface.co/models"
        self.fallback_models = [
            "microsoft/DialoGPT-medium",
            "distilbert-base-uncased",
            "facebook/blenderbot-400M-distill"
        ]
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
    
    def generate_investment_insights(
        self, 
        user_input: Dict[str, Any], 
        simulation_results: Dict[str, Any],
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate AI-powered investment insights based on user data and simulation results.
        
        Args:
            user_input: User's financial input
            simulation_results: Monte Carlo simulation results
            strategies: Recommended investment strategies
            
        Returns:
            Dictionary containing AI insights and recommendations
        """
        try:
            # Create a grounded prompt for conservative financial advice
            prompt = self._create_grounded_prompt(user_input, simulation_results, strategies)
            
            # Get AI response
            ai_response = self._get_ai_response(prompt)
            
            # Parse and structure the response
            insights = self._parse_ai_response(ai_response, user_input, strategies)
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            return self._get_fallback_insights(user_input, strategies)
    
    def _create_grounded_prompt(
        self, 
        user_input: Dict[str, Any], 
        simulation_results: Dict[str, Any],
        strategies: List[Dict[str, Any]]
    ) -> str:
        """Create a conservative, grounded prompt for AI analysis."""
        
        # Extract key metrics
        target_amount = user_input.get('target_investment_amount', 0)
        current_income = user_input.get('current_income', 0)
        tenure = user_input.get('investment_tenure_years', 0)
        risk_appetite = user_input.get('risk_appetite', 'medium')
        
        # Get best strategy metrics
        best_strategy = strategies[0] if strategies else {}
        success_prob = best_strategy.get('probability_of_success', 0) * 100
        expected_return = best_strategy.get('expected_annual_return', 0) * 100
        volatility = best_strategy.get('expected_volatility', 0) * 100
        
        prompt = f"""
        As a conservative financial advisor, provide grounded investment advice based on these facts:
        
        Client Profile:
        - Annual Income: ${current_income:,.0f}
        - Target Amount: ${target_amount:,.0f}
        - Investment Period: {tenure} years
        - Risk Tolerance: {risk_appetite}
        
        Analysis Results:
        - Probability of Success: {success_prob:.1f}%
        - Expected Annual Return: {expected_return:.1f}%
        - Portfolio Volatility: {volatility:.1f}%
        
        Please provide:
        1. A realistic assessment of achieving the goal
        2. Conservative risk warnings
        3. Practical next steps
        4. Alternative approaches if needed
        
        Keep advice grounded, conservative, and realistic. Avoid overly optimistic projections.
        Focus on risk management and practical implementation.
        """
        
        return prompt.strip()
    
    def _get_ai_response(self, prompt: str) -> str:
        """Get response from free AI API."""
        try:
            # Try Hugging Face Inference API (free tier)
            response = requests.post(
                f"{self.huggingface_api_url}/microsoft/DialoGPT-medium",
                headers={"Authorization": f"Bearer hf_demo"},  # Demo token
                json={"inputs": prompt, "parameters": {"max_length": 500, "temperature": 0.7}}
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', '')
            
        except Exception as e:
            logger.warning(f"Hugging Face API failed: {str(e)}")
        
        # Fallback to rule-based insights
        return self._generate_rule_based_insights(prompt)
    
    def _generate_rule_based_insights(self, prompt: str) -> str:
        """Generate insights using rule-based logic as fallback."""
        
        # Extract key information from prompt
        income_match = re.search(r'Annual Income: \$([0-9,]+)', prompt)
        target_match = re.search(r'Target Amount: \$([0-9,]+)', prompt)
        tenure_match = re.search(r'Investment Period: (\d+) years', prompt)
        success_match = re.search(r'Probability of Success: ([\d.]+)%', prompt)
        
        income = int(income_match.group(1).replace(',', '')) if income_match else 0
        target = int(target_match.group(1).replace(',', '')) if target_match else 0
        tenure = int(tenure_match.group(1)) if tenure_match else 0
        success_prob = float(success_match.group(1)) if success_match else 0
        
        # Generate conservative insights
        insights = []
        
        # Goal achievability assessment
        if success_prob > 70:
            insights.append("✓ Your goal appears achievable with the recommended strategy.")
        elif success_prob > 50:
            insights.append("⚠️ Your goal is moderately achievable. Consider adjusting expectations or increasing contributions.")
        else:
            insights.append("⚠️ Your goal may be challenging to achieve. Consider extending the timeline or increasing monthly contributions.")
        
        # Risk warnings
        insights.append("⚠️ All investments carry risk. Past performance doesn't guarantee future results.")
        insights.append("💡 Consider diversifying across different asset classes to reduce risk.")
        
        # Practical advice
        if target > income * 2:
            insights.append("💡 Your target is ambitious relative to your income. Consider a phased approach.")
        
        if tenure < 5:
            insights.append("⚠️ Short-term investing carries higher risk. Consider your risk tolerance carefully.")
        
        # Next steps
        insights.append("📋 Next steps: 1) Review your budget, 2) Start with small amounts, 3) Monitor regularly, 4) Adjust as needed.")
        
        return "\n\n".join(insights)
    
    def _parse_ai_response(self, ai_response: str, user_input: Dict[str, Any], strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse AI response into structured insights."""
        
        # Clean up the response
        cleaned_response = ai_response.replace("Generated text:", "").strip()
        
        # Extract key insights using simple parsing
        insights = {
            "ai_analysis": cleaned_response,
            "risk_assessment": self._assess_risk_level(user_input, strategies),
            "recommendations": self._extract_recommendations(cleaned_response),
            "warnings": self._extract_warnings(cleaned_response),
            "next_steps": self._extract_next_steps(cleaned_response),
            "confidence_level": self._calculate_confidence(user_input, strategies),
            "generated_at": datetime.now().isoformat()
        }
        
        return insights
    
    def _assess_risk_level(self, user_input: Dict[str, Any], strategies: List[Dict[str, Any]]) -> str:
        """Assess overall risk level of the investment plan."""
        risk_appetite = user_input.get('risk_appetite', 'medium')
        tenure = user_input.get('investment_tenure_years', 0)
        
        if tenure < 3:
            return "HIGH - Short-term investing carries significant risk"
        elif risk_appetite == 'high' and tenure > 10:
            return "MEDIUM-HIGH - Aggressive strategy with long timeline"
        elif risk_appetite == 'low':
            return "LOW - Conservative approach"
        else:
            return "MEDIUM - Balanced risk approach"
    
    def _extract_recommendations(self, response: str) -> List[str]:
        """Extract actionable recommendations from AI response."""
        recommendations = []
        
        # Look for numbered lists or bullet points
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['recommend', 'suggest', 'consider', 'try']):
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _extract_warnings(self, response: str) -> List[str]:
        """Extract risk warnings from AI response."""
        warnings = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['warning', 'risk', 'caution', 'be careful']):
                warnings.append(line)
        
        return warnings[:3]  # Limit to 3 warnings
    
    def _extract_next_steps(self, response: str) -> List[str]:
        """Extract next steps from AI response."""
        steps = []
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if any(indicator in line.lower() for indicator in ['step', 'next', 'action', 'do']):
                steps.append(line)
        
        return steps[:4]  # Limit to 4 steps
    
    def _calculate_confidence(self, user_input: Dict[str, Any], strategies: List[Dict[str, Any]]) -> str:
        """Calculate confidence level in the recommendations."""
        if not strategies:
            return "LOW"
        
        best_strategy = strategies[0]
        success_prob = best_strategy.get('probability_of_success', 0)
        tenure = user_input.get('investment_tenure_years', 0)
        
        if success_prob > 0.8 and tenure > 5:
            return "HIGH"
        elif success_prob > 0.6 and tenure > 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_fallback_insights(self, user_input: Dict[str, Any], strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Provide fallback insights when AI service is unavailable."""
        return {
            "ai_analysis": "AI analysis temporarily unavailable. Using conservative financial guidance.",
            "risk_assessment": "Please consult with a qualified financial advisor for personalized advice.",
            "recommendations": [
                "Start with small, regular investments",
                "Diversify across different asset classes",
                "Review and adjust your strategy regularly"
            ],
            "warnings": [
                "All investments carry risk of loss",
                "Past performance doesn't guarantee future results",
                "Consider your risk tolerance carefully"
            ],
            "next_steps": [
                "Research your investment options thoroughly",
                "Start with a small amount to test your strategy",
                "Monitor your investments regularly",
                "Consider professional financial advice"
            ],
            "confidence_level": "MEDIUM",
            "generated_at": datetime.now().isoformat()
        }
    
    def generate_market_insights(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI insights about current market conditions."""
        try:
            # Simple market analysis based on available data
            insights = {
                "market_sentiment": "NEUTRAL",
                "key_observations": [],
                "recommendations": [],
                "generated_at": datetime.now().isoformat()
            }
            
            # Analyze market data
            for index, data in market_data.items():
                change_pct = data.get('change_pct', 0)
                if change_pct > 2:
                    insights["key_observations"].append(f"{index} is up {change_pct:.1f}% - strong performance")
                elif change_pct < -2:
                    insights["key_observations"].append(f"{index} is down {change_pct:.1f}% - market volatility")
                else:
                    insights["key_observations"].append(f"{index} is relatively stable at {change_pct:.1f}%")
            
            # Generate recommendations based on market conditions
            if any(data.get('change_pct', 0) < -3 for data in market_data.values()):
                insights["recommendations"].append("Market shows volatility - consider dollar-cost averaging")
                insights["market_sentiment"] = "BEARISH"
            elif any(data.get('change_pct', 0) > 3 for data in market_data.values()):
                insights["recommendations"].append("Market shows strength - but avoid FOMO investing")
                insights["market_sentiment"] = "BULLISH"
            else:
                insights["recommendations"].append("Market is stable - good time for regular investing")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating market insights: {str(e)}")
            return {
                "market_sentiment": "UNKNOWN",
                "key_observations": ["Market data analysis unavailable"],
                "recommendations": ["Consult multiple sources for market analysis"],
                "generated_at": datetime.now().isoformat()
            }
