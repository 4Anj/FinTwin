/**
 * Financial Digital Twin Frontend Application
 */

class FinancialDigitalTwin {
    constructor() {
        this.apiBase = '/api';
        this.currentUserId = null;
        this.portfolioChart = null;
        this.riskChart = null;
        this.currentSimulation = null;
        this.selectedInstruments = new Set();
        this.availableInstruments = [];
        this.currentStrategy = 'basic'; // 'basic' or 'diversified'
        this.assetClassPreferences = {
            stocks: true,
            bonds: true,
            real_estate: true,
            commodities: true,
            crypto: false,
            alternatives: false
        };
        
        this.initializeEventListeners();
        this.loadUserProfile();
        this.loadPopularInstruments();
        
        // Add some default instruments
        setTimeout(() => {
            this.addInstrument('SPY', 'SPDR S&P 500 ETF Trust');
            this.addInstrument('QQQ', 'Invesco QQQ Trust');
            this.addInstrument('VTI', 'Vanguard Total Stock Market ETF');
        }, 1000);
    }

    initializeEventListeners() {
        // Navigation
        document.getElementById('profileBtn').addEventListener('click', () => this.toggleSection('profileSection'));
        document.getElementById('settingsBtn').addEventListener('click', () => this.toggleSection('settingsSection'));
        
        // Profile
        document.getElementById('saveProfileBtn').addEventListener('click', () => this.saveUserProfile());
        
        // Simulation form
        document.getElementById('simulationForm').addEventListener('submit', (e) => this.runSimulation(e));
        
        // Symbol search and dropdown
        const symbolSearch = document.getElementById('symbolSearch');
        symbolSearch.addEventListener('input', (e) => this.handleSymbolSearch(e));
        symbolSearch.addEventListener('focus', () => this.showDropdown());
        symbolSearch.addEventListener('blur', () => this.hideDropdown());
        
        // Click outside to close dropdown
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#symbolSearch') && !e.target.closest('#symbolDropdown')) {
                this.hideDropdown();
            }
        });
        
        // Strategy selection buttons
        document.getElementById('basicStrategyBtn').addEventListener('click', () => this.selectStrategy('basic'));
        document.getElementById('diversifiedStrategyBtn').addEventListener('click', () => this.selectStrategy('diversified'));
        
        // Asset class preference checkboxes
        const assetClassCheckboxes = ['includeStocks', 'includeBonds', 'includeRealEstate', 'includeCommodities', 'includeCrypto', 'includeAlternatives'];
        assetClassCheckboxes.forEach(id => {
            document.getElementById(id).addEventListener('change', (e) => this.updateAssetClassPreferences(e));
        });
    }

    toggleSection(sectionId) {
        const sections = ['profileSection', 'settingsSection'];
        sections.forEach(id => {
            const section = document.getElementById(id);
            if (id === sectionId) {
                section.classList.toggle('hidden');
            } else {
                section.classList.add('hidden');
            }
        });
    }

    async loadUserProfile() {
        // Try to load existing profile or create a new one
        const storedUserId = localStorage.getItem('userId');
        if (storedUserId) {
            try {
                const response = await fetch(`${this.apiBase}/user/profile/${storedUserId}`);
                if (response.ok) {
                    const profile = await response.json();
                    this.currentUserId = profile.user_id;
                    this.populateProfileForm(profile);
                }
            } catch (error) {
                console.error('Error loading profile:', error);
            }
        }
        
        if (!this.currentUserId) {
            this.currentUserId = this.generateUserId();
            localStorage.setItem('userId', this.currentUserId);
        }
    }

    generateUserId() {
        return 'user_' + Math.random().toString(36).substr(2, 9);
    }

    populateProfileForm(profile) {
        document.getElementById('userName').value = profile.name || '';
        document.getElementById('userEmail').value = profile.email || '';
    }

    async saveUserProfile() {
        const profileData = {
            user_id: this.currentUserId,
            name: document.getElementById('userName').value,
            email: document.getElementById('userEmail').value,
            preferences: {
                use_yfinance: document.getElementById('useYFinance').checked,
                real_time_mode: document.getElementById('realTimeMode').checked
            }
        };

        try {
            const response = await fetch(`${this.apiBase}/user/profile`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(profileData)
            });

            if (response.ok) {
                alert('Profile saved successfully!');
                this.toggleSection('profileSection');
            } else {
                throw new Error('Failed to save profile');
            }
        } catch (error) {
            console.error('Error saving profile:', error);
            alert('Error saving profile. Please try again.');
        }
    }

    async loadPopularInstruments() {
        try {
            const response = await fetch(`${this.apiBase}/instruments/popular`);
            if (response.ok) {
                const data = await response.json();
                this.availableInstruments = data.instruments || [];
                this.populateDropdown();
            }
        } catch (error) {
            console.error('Error loading popular instruments:', error);
            // Fallback to default instruments
            this.availableInstruments = [
                { symbol: 'SPY', name: 'SPDR S&P 500 ETF Trust' },
                { symbol: 'QQQ', name: 'Invesco QQQ Trust' },
                { symbol: 'VTI', name: 'Vanguard Total Stock Market ETF' },
                { symbol: 'BND', name: 'Vanguard Total Bond Market ETF' },
                { symbol: 'GLD', name: 'SPDR Gold Trust' },
                { symbol: 'AAPL', name: 'Apple Inc.' },
                { symbol: 'MSFT', name: 'Microsoft Corporation' },
                { symbol: 'GOOGL', name: 'Alphabet Inc.' },
                { symbol: 'AMZN', name: 'Amazon.com Inc.' },
                { symbol: 'TSLA', name: 'Tesla Inc.' }
            ];
            this.populateDropdown();
        }
    }

    populateDropdown() {
        const dropdown = document.getElementById('symbolDropdown');
        dropdown.innerHTML = '';
        
        this.availableInstruments.forEach(instrument => {
            if (!this.selectedInstruments.has(instrument.symbol)) {
                const option = document.createElement('div');
                option.className = 'px-3 py-2 hover:bg-gray-100 cursor-pointer flex justify-between items-center';
                option.innerHTML = `
                    <div>
                        <div class="font-medium text-gray-900">${instrument.symbol}</div>
                        <div class="text-sm text-gray-500">${instrument.name}</div>
                    </div>
                    <i class="fas fa-plus text-blue-500"></i>
                `;
                option.addEventListener('click', () => this.addInstrument(instrument.symbol, instrument.name));
                dropdown.appendChild(option);
            }
        });
    }

    handleSymbolSearch(event) {
        const query = event.target.value.toLowerCase();
        const dropdown = document.getElementById('symbolDropdown');
        const options = dropdown.querySelectorAll('div');
        
        options.forEach(option => {
            const symbol = option.querySelector('.font-medium').textContent.toLowerCase();
            const name = option.querySelector('.text-sm').textContent.toLowerCase();
            
            if (symbol.includes(query) || name.includes(query)) {
                option.style.display = 'flex';
            } else {
                option.style.display = 'none';
            }
        });
        
        this.showDropdown();
    }

    showDropdown() {
        const dropdown = document.getElementById('symbolDropdown');
        dropdown.classList.remove('hidden');
    }

    hideDropdown() {
        setTimeout(() => {
            const dropdown = document.getElementById('symbolDropdown');
            dropdown.classList.add('hidden');
        }, 200);
    }

    addInstrument(symbol, name) {
        if (!this.selectedInstruments.has(symbol)) {
            this.selectedInstruments.add(symbol);
            this.updateSelectedInstrumentsDisplay();
            this.populateDropdown();
            document.getElementById('symbolSearch').value = '';
        }
    }

    removeInstrument(symbol) {
        this.selectedInstruments.delete(symbol);
        this.updateSelectedInstrumentsDisplay();
        this.populateDropdown();
    }

    updateSelectedInstrumentsDisplay() {
        const container = document.getElementById('selectedInstruments');
        container.innerHTML = '';
        
        this.selectedInstruments.forEach(symbol => {
            const instrument = this.availableInstruments.find(inst => inst.symbol === symbol);
            const name = instrument ? instrument.name : symbol;
            
            const tag = document.createElement('div');
            tag.className = 'inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800';
            tag.innerHTML = `
                <span class="mr-2">${symbol}</span>
                <button type="button" onclick="app.removeInstrument('${symbol}')" class="ml-1 text-blue-600 hover:text-blue-800">
                    <i class="fas fa-times"></i>
                </button>
            `;
            container.appendChild(tag);
        });
    }

    getSelectedInstrumentsArray() {
        return Array.from(this.selectedInstruments);
    }

    selectStrategy(strategy) {
        this.currentStrategy = strategy;
        
        // Update button styles
        document.getElementById('basicStrategyBtn').className = 
            strategy === 'basic' ? 'px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700' : 
            'px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300';
        
        document.getElementById('diversifiedStrategyBtn').className = 
            strategy === 'diversified' ? 'px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700' : 
            'px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300';
        
        // Show/hide sections
        document.getElementById('basicStrategySection').classList.toggle('hidden', strategy !== 'basic');
        document.getElementById('diversifiedStrategySection').classList.toggle('hidden', strategy !== 'diversified');
    }

    updateAssetClassPreferences(event) {
        const checkboxId = event.target.id;
        const isChecked = event.target.checked;
        
        // Map checkbox IDs to preference keys
        const mapping = {
            'includeStocks': 'stocks',
            'includeBonds': 'bonds',
            'includeRealEstate': 'real_estate',
            'includeCommodities': 'commodities',
            'includeCrypto': 'crypto',
            'includeAlternatives': 'alternatives'
        };
        
        if (mapping[checkboxId]) {
            this.assetClassPreferences[mapping[checkboxId]] = isChecked;
        }
    }

    async runSimulation(event) {
        event.preventDefault();
        
        // Show loading indicator
        document.getElementById('loadingIndicator').classList.remove('hidden');
        document.getElementById('resultsSection').classList.add('hidden');
        
        try {
            // Validate inputs based on strategy type
            if (this.currentStrategy === 'basic') {
                const selectedSymbols = this.getSelectedInstrumentsArray();
                if (selectedSymbols.length === 0) {
                    alert('Please select at least one financial instrument.');
                    return;
                }
            } else {
                // Check if at least one asset class is selected
                const selectedAssetClasses = Object.values(this.assetClassPreferences).some(Boolean);
                if (!selectedAssetClasses) {
                    alert('Please select at least one asset class for diversified strategy.');
                    return;
                }
            }

            // Get form data
            const formData = {
                user_input: {
                    current_income: parseFloat(document.getElementById('currentIncome').value),
                    target_investment_amount: parseFloat(document.getElementById('targetAmount').value),
                    investment_tenure_years: parseInt(document.getElementById('tenure').value),
                    risk_appetite: document.getElementById('riskAppetite').value,
                    initial_investment: parseFloat(document.getElementById('initialInvestment').value) || 0
                },
                symbols: this.currentStrategy === 'basic' ? this.getSelectedInstrumentsArray() : [],
                simulation_count: 100, // Limited to 100 for real-time performance
                use_yfinance: document.getElementById('useYFinance').checked,
                real_time_mode: document.getElementById('realTimeMode').checked
            };

            // Choose API endpoint based on strategy
            const apiEndpoint = this.currentStrategy === 'basic' ? 
                `${this.apiBase}/simulate` : 
                `${this.apiBase}/simulate/diversified`;

            // Run simulation
            const response = await fetch(apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Simulation failed');
            }

            const result = await response.json();
            this.currentSimulation = result;
            
            // Display results
            await this.displayResults(result);
            
            // Save to user profile
            await this.saveSimulationToProfile(result);
            
        } catch (error) {
            console.error('Simulation error:', error);
            alert(`Simulation failed: ${error.message}`);
        } finally {
            // Hide loading indicator
            document.getElementById('loadingIndicator').classList.add('hidden');
        }
    }

    async displayResults(result) {
        // Show results section
        document.getElementById('resultsSection').classList.remove('hidden');
        
        // Load market overview
        await this.loadMarketOverview();
        
        // Create charts
        this.createPortfolioChart(result);
        this.createRiskChart(result);
        
        // Display AI insights if available
        if (result.ai_insights) {
            this.displayAIInsights(result.ai_insights);
        }
        
        // Display strategies
        this.displayStrategies(result.recommended_strategies);
        
        // Display blockchain integrity information
        this.displayBlockchainIntegrity(result);
    }

    async loadMarketOverview() {
        try {
            const response = await fetch(`${this.apiBase}/market-overview`);
            if (response.ok) {
                const data = await response.json();
                this.displayMarketOverview(data.market_overview);
            }
        } catch (error) {
            console.error('Error loading market overview:', error);
        }
    }

    displayMarketOverview(overview) {
        const container = document.getElementById('marketOverview');
        container.innerHTML = '';

        Object.entries(overview).forEach(([index, data]) => {
            const changeClass = data.change >= 0 ? 'text-green-600' : 'text-red-600';
            const changeIcon = data.change >= 0 ? 'fas fa-arrow-up' : 'fas fa-arrow-down';
            
            const card = document.createElement('div');
            card.className = 'bg-gray-50 rounded-lg p-4 card-hover';
            card.innerHTML = `
                <div class="flex justify-between items-center">
                    <div>
                        <h3 class="font-semibold text-gray-800">${index}</h3>
                        <p class="text-2xl font-bold text-gray-900">$${data.current.toFixed(2)}</p>
                    </div>
                    <div class="text-right">
                        <div class="${changeClass}">
                            <i class="${changeIcon} mr-1"></i>
                            ${data.change_pct.toFixed(2)}%
                        </div>
                        <div class="text-sm text-gray-600">
                            ${data.change >= 0 ? '+' : ''}$${data.change.toFixed(2)}
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
    }

    createPortfolioChart(result) {
        const ctx = document.getElementById('portfolioChart').getContext('2d');
        
        if (this.portfolioChart) {
            this.portfolioChart.destroy();
        }

        const bestStrategy = result.recommended_strategies[0];
        const paths = bestStrategy.monte_carlo_result.simulation_paths;
        
        if (!paths || paths.length === 0) {
            return;
        }

        // Create time labels
        const years = result.user_input.investment_tenure_years;
        const labels = Array.from({length: years}, (_, i) => `Year ${i + 1}`);
        
        // Calculate mean path
        const meanPath = [];
        for (let i = 0; i < paths[0].length; i++) {
            const sum = paths.reduce((acc, path) => acc + path[i], 0);
            meanPath.push(sum / paths.length);
        }

        // Create datasets
        const datasets = [
            {
                label: 'Mean Portfolio Value',
                data: meanPath,
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: false,
                tension: 0.1
            }
        ];

        // Add sample individual paths
        paths.slice(0, 3).forEach((path, index) => {
            const colors = [
                'rgba(16, 185, 129, 0.5)',
                'rgba(245, 158, 11, 0.5)',
                'rgba(239, 68, 68, 0.5)'
            ];
            
            datasets.push({
                label: `Simulation ${index + 1}`,
                data: path,
                borderColor: colors[index],
                backgroundColor: colors[index].replace('0.5', '0.1'),
                fill: false,
                tension: 0.1,
                pointRadius: 0
            });
        });

        this.portfolioChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Portfolio Value ($)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    createRiskChart(result) {
        const ctx = document.getElementById('riskChart').getContext('2d');
        
        if (this.riskChart) {
            this.riskChart.destroy();
        }

        const strategies = result.recommended_strategies;
        const labels = strategies.map(s => s.strategy_name);
        const probabilities = strategies.map(s => s.probability_of_success * 100);
        const volatilities = strategies.map(s => s.expected_volatility * 100);

        this.riskChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Success Probability (%)',
                        data: probabilities,
                        backgroundColor: 'rgba(59, 130, 246, 0.6)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Volatility (%)',
                        data: volatilities,
                        backgroundColor: 'rgba(239, 68, 68, 0.6)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Percentage (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }

    displayStrategies(strategies) {
        const container = document.getElementById('strategiesContainer');
        container.innerHTML = '';

        strategies.forEach((strategy, index) => {
            const card = document.createElement('div');
            card.className = 'bg-gray-50 rounded-lg p-6 card-hover';
            
            const riskColor = this.getRiskColor(strategy.risk_level);
            const successColor = strategy.probability_of_success > 0.7 ? 'text-green-600' : 
                                strategy.probability_of_success > 0.4 ? 'text-yellow-600' : 'text-red-600';
            
            card.innerHTML = `
                <div class="flex justify-between items-start mb-4">
                    <h3 class="text-xl font-bold text-gray-800">${strategy.strategy_name}</h3>
                    <span class="px-3 py-1 rounded-full text-sm font-medium ${riskColor}">
                        ${strategy.risk_level.toUpperCase()} RISK
                    </span>
                </div>
                
                <p class="text-gray-600 mb-4">${strategy.description}</p>
                
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                    <div class="text-center">
                        <div class="text-2xl font-bold ${successColor}">
                            ${(strategy.probability_of_success * 100).toFixed(1)}%
                        </div>
                        <div class="text-sm text-gray-600">Success Probability</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-blue-600">
                            ${(strategy.expected_annual_return * 100).toFixed(1)}%
                        </div>
                        <div class="text-sm text-gray-600">Expected Return</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-orange-600">
                            ${(strategy.expected_volatility * 100).toFixed(1)}%
                        </div>
                        <div class="text-sm text-gray-600">Volatility</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-600">
                            ${strategy.sharpe_ratio.toFixed(2)}
                        </div>
                        <div class="text-sm text-gray-600">Sharpe Ratio</div>
                    </div>
                </div>
                
                <div class="mb-4">
                    <h4 class="font-semibold text-gray-800 mb-2">Asset Allocation:</h4>
                    <div class="flex flex-wrap gap-2">
                        ${Object.entries(strategy.recommended_allocation).map(([symbol, allocation]) => `
                            <span class="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                                ${symbol}: ${(allocation * 100).toFixed(1)}%
                            </span>
                        `).join('')}
                    </div>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                        <span class="font-medium text-gray-700">Worst Case (5th percentile):</span>
                        <span class="text-red-600 font-bold">$${strategy.monte_carlo_result.worst_case_5th_percentile.toLocaleString()}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Best Case (95th percentile):</span>
                        <span class="text-green-600 font-bold">$${strategy.monte_carlo_result.best_case_95th_percentile.toLocaleString()}</span>
                    </div>
                    <div>
                        <span class="font-medium text-gray-700">Max Drawdown:</span>
                        <span class="text-orange-600 font-bold">${(strategy.max_drawdown * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
            
            container.appendChild(card);
        });
    }

    displayAIInsights(insights) {
        const container = document.getElementById('aiInsightsContainer');
        const section = document.getElementById('aiInsightsSection');
        
        if (!insights) {
            section.classList.add('hidden');
            return;
        }
        
        section.classList.remove('hidden');
        
        container.innerHTML = `
            <div class="space-y-6">
                <!-- Main Analysis -->
                <div class="bg-blue-50 rounded-lg p-4">
                    <h3 class="font-semibold text-blue-800 mb-2">Analysis</h3>
                    <p class="text-blue-700">${insights.ai_analysis}</p>
                </div>
                
                <!-- Risk Assessment -->
                <div class="bg-yellow-50 rounded-lg p-4">
                    <h3 class="font-semibold text-yellow-800 mb-2">Risk Assessment</h3>
                    <p class="text-yellow-700">${insights.risk_assessment}</p>
                </div>
                
                <!-- Recommendations -->
                <div class="bg-green-50 rounded-lg p-4">
                    <h3 class="font-semibold text-green-800 mb-2">Recommendations</h3>
                    <ul class="list-disc list-inside space-y-1">
                        ${insights.recommendations.map(rec => `<li class="text-green-700">${rec}</li>`).join('')}
                    </ul>
                </div>
                
                <!-- Warnings -->
                <div class="bg-red-50 rounded-lg p-4">
                    <h3 class="font-semibold text-red-800 mb-2">Important Warnings</h3>
                    <ul class="list-disc list-inside space-y-1">
                        ${insights.warnings.map(warning => `<li class="text-red-700">${warning}</li>`).join('')}
                    </ul>
                </div>
                
                <!-- Next Steps -->
                <div class="bg-purple-50 rounded-lg p-4">
                    <h3 class="font-semibold text-purple-800 mb-2">Next Steps</h3>
                    <ul class="list-disc list-inside space-y-1">
                        ${insights.next_steps.map(step => `<li class="text-purple-700">${step}</li>`).join('')}
                    </ul>
                </div>
                
                <!-- Confidence Level -->
                <div class="flex justify-between items-center bg-gray-50 rounded-lg p-4">
                    <span class="font-medium text-gray-700">Confidence Level:</span>
                    <span class="px-3 py-1 rounded-full text-sm font-medium ${
                        insights.confidence_level === 'HIGH' ? 'bg-green-100 text-green-800' :
                        insights.confidence_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                    }">
                        ${insights.confidence_level}
                    </span>
                </div>
            </div>
        `;
    }

    async displayBlockchainIntegrity(simulationResult) {
        const container = document.getElementById('integrityContainer');
        const section = document.getElementById('integritySection');
        
        try {
            // Get blockchain integrity report
            const response = await fetch(`${this.apiBase}/integrity/report`);
            const integrityReport = await response.json();
            
            section.classList.remove('hidden');
            
            container.innerHTML = `
                <div class="space-y-6">
                    <!-- Blockchain Status -->
                    <div class="bg-green-50 rounded-lg p-4">
                        <h3 class="font-semibold text-green-800 mb-2">🔗 Blockchain Status</h3>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                            <div>
                                <span class="font-medium">Chain Length:</span>
                                <span class="text-green-700">${integrityReport.blockchain_info?.chain_length || 'N/A'}</span>
                            </div>
                            <div>
                                <span class="font-medium">Pending Transactions:</span>
                                <span class="text-green-700">${integrityReport.blockchain_info?.pending_transactions || 'N/A'}</span>
                            </div>
                            <div>
                                <span class="font-medium">Difficulty:</span>
                                <span class="text-green-700">${integrityReport.blockchain_info?.difficulty || 'N/A'}</span>
                            </div>
                            <div>
                                <span class="font-medium">Chain Integrity:</span>
                                <span class="text-green-700">${integrityReport.blockchain_info?.chain_integrity ? '✅ Verified' : '❌ Compromised'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Security Features -->
                    <div class="bg-blue-50 rounded-lg p-4">
                        <h3 class="font-semibold text-blue-800 mb-2">🛡️ Security Features</h3>
                        <ul class="list-disc list-inside space-y-1 text-blue-700">
                            <li>Cryptographic hashing (SHA-256) for data integrity</li>
                            <li>Digital signatures for block verification</li>
                            <li>Proof of Work consensus mechanism</li>
                            <li>Merkle tree for transaction verification</li>
                            <li>Tamper-proof audit trail</li>
                        </ul>
                    </div>
                    
                    <!-- Verification Status -->
                    <div class="bg-purple-50 rounded-lg p-4">
                        <h3 class="font-semibold text-purple-800 mb-2">✅ Verification Status</h3>
                        <div class="flex items-center space-x-4">
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                                <span class="text-purple-700">Simulation data cryptographically secured</span>
                            </div>
                            <div class="flex items-center">
                                <span class="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                                <span class="text-purple-700">Immutable blockchain record created</span>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Public Key for Verification -->
                    <div class="bg-gray-50 rounded-lg p-4">
                        <h3 class="font-semibold text-gray-800 mb-2">🔑 Public Key (for verification)</h3>
                        <div class="bg-white rounded p-3 font-mono text-xs break-all text-gray-600">
                            ${integrityReport.public_key ? integrityReport.public_key.substring(0, 100) + '...' : 'Not available'}
                        </div>
                        <p class="text-sm text-gray-600 mt-2">
                            Use this public key to verify the authenticity of blockchain records
                        </p>
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error loading blockchain integrity:', error);
            section.classList.add('hidden');
        }
    }

    getRiskColor(riskLevel) {
        switch (riskLevel) {
            case 'low': return 'bg-green-100 text-green-800';
            case 'medium': return 'bg-yellow-100 text-yellow-800';
            case 'high': return 'bg-red-100 text-red-800';
            default: return 'bg-gray-100 text-gray-800';
        }
    }

    async saveSimulationToProfile(result) {
        if (!this.currentUserId) return;
        
        try {
            // Generate a simulation ID (in a real app, this would come from the API)
            const simulationId = 'sim_' + Date.now();
            
            const response = await fetch(`${this.apiBase}/user/profile/${this.currentUserId}/simulation`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ simulation_id: simulationId })
            });

            if (!response.ok) {
                console.warn('Failed to save simulation to profile');
            }
        } catch (error) {
            console.error('Error saving simulation to profile:', error);
        }
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new FinancialDigitalTwin();
});
