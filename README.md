# Financial Digital Twin

A comprehensive FastAPI-based financial planning and investment simulation platform that uses Monte Carlo simulations to model market behavior and provide personalized investment recommendations.

## Features

### Core Functionality
- **Real-time Market Data**: Integration with Yahoo Finance API for live market data
- **Monte Carlo Simulations**: Advanced financial modeling with configurable simulation limits (max 100 for real-time performance)
- **Investment Strategy Recommendations**: AI-powered suggestions based on risk appetite and financial goals
- **Risk Analysis**: Comprehensive risk metrics including Sharpe ratio, volatility, and drawdown analysis
- **User Profiles**: Persistent user data and simulation history

### Frontend Features
- **Interactive Dashboard**: Modern, responsive web interface
- **Real-time Charts**: Portfolio performance and risk analysis visualizations using Chart.js
- **User Profile Management**: Save preferences and simulation history
- **Settings Panel**: Toggle real-time data and simulation modes
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

### API Endpoints
- `POST /api/simulate` - Run Monte Carlo simulations
- `GET /api/market-overview` - Get current market data
- `GET /api/instruments/popular` - Get popular financial instruments
- `POST /api/user/profile` - Create/update user profile
- `GET /api/user/profile/{user_id}` - Get user profile
- `GET /api/dashboard/{user_id}` - Get dashboard data
- `GET /api/health` - Health check

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

4. **Access the application**:
   - Open your browser and go to `http://localhost:8000`
   - The API documentation is available at `http://localhost:8000/docs`

## Usage

### Basic Workflow

1. **Set Up Profile**: Click the profile icon to create your user profile
2. **Configure Settings**: Use the settings panel to enable/disable real-time data
3. **Input Financial Data**:
   - Current annual income
   - Target investment amount
   - Investment tenure (years)
   - Risk appetite (low/medium/high)
   - Financial instruments to analyze
4. **Run Simulation**: Click "Run Simulation" to generate recommendations
5. **Analyze Results**: Review charts, strategies, and risk metrics

### Simulation Parameters

- **Simulation Count**: Limited to 100 for real-time performance
- **Real-time Mode**: Faster results with reduced accuracy
- **Yahoo Finance**: Toggle real-time data fetching
- **Risk Levels**: Low (conservative), Medium (balanced), High (aggressive)

### Investment Strategies

The system generates multiple strategies based on your risk profile:

- **Conservative**: Focus on bonds and low-volatility instruments
- **Balanced**: 60/40 stock/bond allocation
- **Aggressive**: High allocation to growth stocks
- **Diversified**: Equal weight or risk-parity approaches

## Architecture

### Backend Components

- **`main.py`**: FastAPI application with REST endpoints
- **`models.py`**: Pydantic data models for type safety
- **`data_service.py`**: Yahoo Finance integration and data processing
- **`monte_carlo.py`**: Monte Carlo simulation engine
- **`strategy_engine.py`**: Investment strategy recommendation system

### Frontend Components

- **`static/index.html`**: Main application interface
- **`static/app.js`**: Frontend JavaScript application
- **Charts**: Interactive visualizations using Chart.js
- **Responsive Design**: Mobile-first approach with Tailwind CSS

### Key Features

#### Monte Carlo Simulation
- Geometric Brownian Motion for price modeling
- Configurable simulation parameters
- Real-time performance optimization
- Risk metrics calculation

#### Risk Analysis
- Sharpe ratio calculation
- Maximum drawdown analysis
- Value at Risk (VaR)
- Probability of success metrics

#### Data Management
- Yahoo Finance API integration
- Caching for performance
- Error handling and fallbacks
- Mock data for testing

## Configuration

### Environment Variables
- No environment variables required for basic operation
- Yahoo Finance API is used without authentication

### Customization
- Modify `strategy_engine.py` for custom investment strategies
- Update `data_service.py` for different data sources
- Customize frontend in `static/` directory

## Performance Considerations

### Simulation Limits
- Maximum 100 simulations per run for real-time performance
- Configurable simulation count (10-100)
- Background processing for large simulations

### Caching
- Yahoo Finance data cached for 1 hour
- User profiles stored in memory (use database in production)
- Simulation results cached temporarily

### Real-time Mode
- Faster execution with reduced accuracy
- Optimized for quick decision making
- Suitable for exploratory analysis

## Error Handling

### API Errors
- Comprehensive error responses with details
- Graceful fallbacks for data service failures
- Input validation and sanitization

### Frontend Errors
- User-friendly error messages
- Loading states and progress indicators
- Retry mechanisms for failed requests

## Future Enhancements

### Planned Features
- Machine Learning integration for predictions
- Database persistence for user data
- Advanced portfolio optimization
- Real-time notifications
- Multi-currency support
- Social features and sharing

### Extensibility
- Modular architecture for easy extension
- Plugin system for custom strategies
- API versioning for backward compatibility
- Microservices architecture support

## Troubleshooting

### Common Issues

1. **Yahoo Finance API Errors**:
   - Check internet connection
   - Verify symbol names are correct
   - Enable mock data mode in settings

2. **Simulation Timeout**:
   - Reduce simulation count
   - Enable real-time mode
   - Check system resources

3. **Chart Display Issues**:
   - Refresh the page
   - Check browser console for errors
   - Ensure JavaScript is enabled

### Debug Mode
- Enable browser developer tools
- Check API responses in Network tab
- Review console logs for errors

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Style
- Follow PEP 8 for Python code
- Use TypeScript for frontend enhancements
- Document all public functions
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Create an issue in the repository
- Contact the development team

## Disclaimer

This software is for educational and informational purposes only. It does not constitute financial advice. Always consult with qualified financial professionals before making investment decisions. Past performance does not guarantee future results.
