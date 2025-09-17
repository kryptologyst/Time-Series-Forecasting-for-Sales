# Time Series Forecasting for Sales

A comprehensive machine learning system for predicting future sales based on historical data patterns. This project combines advanced time series analysis with an interactive web interface for business decision-making.

## Features

- **Multiple ML Models**: Linear Regression and Random Forest algorithms
- **Interactive Dashboard**: Streamlit-based web interface with real-time visualizations
- **Mock Database**: SQLite database with realistic sales patterns and seasonal trends
- **Advanced Analytics**: Feature importance, trend analysis, and performance metrics
- **Future Forecasting**: Predict sales for up to 12 months ahead
- **Data Export**: Download forecasts and historical data in CSV format

## What This Project Demonstrates

- Time series forecasting using lag-based features
- Machine learning model comparison and selection
- Interactive data visualization with Plotly
- Database integration for realistic data simulation
- Web application development with Streamlit
- Statistical analysis and trend decomposition

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd 0065_Time_series_forecasting_for_sales
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # For web interface
   streamlit run app.py
   
   # For command line forecasting
   python sales_forecaster.py
   
   # For data generation testing
   python data_generator.py
   ```

## 📁 Project Structure

```
├── app.py                 # Streamlit web application
├── sales_forecaster.py    # Main forecasting engine
├── data_generator.py      # Mock database and data generation
├── 0065.py               # Original simple implementation
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── sales_data.db         # SQLite database (auto-generated)
```

## Usage

### Web Interface

1. Launch the Streamlit app: `streamlit run app.py`
2. Navigate through the tabs:
   - **Overview**: Historical sales analysis and trends
   - **Forecasting**: Generate ML-powered predictions
   - **Analysis**: Advanced statistical insights
   - **Data**: Manage and export data

### Command Line

```python
from sales_forecaster import SalesForecaster
from data_generator import generate_sales_data

# Load data
df = generate_sales_data(use_database=True)

# Initialize and train forecaster
forecaster = SalesForecaster()
X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
forecaster.train_models(X_train, y_train)

# Generate forecasts
future_forecasts = forecaster.forecast_future(df, periods=6)
print(future_forecasts)
```

## Machine Learning Pipeline

1. **Data Preprocessing**
   - Lag feature creation (1-3 months)
   - Rolling statistics (mean, std)
   - Seasonal features (month, quarter)

2. **Model Training**
   - Cross-validation for model selection
   - Feature scaling with StandardScaler
   - Performance evaluation with multiple metrics

3. **Forecasting**
   - Multi-step ahead predictions
   - Confidence intervals and uncertainty quantification
   - Visualization of results

## Key Metrics

- **R² Score**: Model explanation power
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: 5-fold validation for robustness

## Database Schema

The SQLite database includes:

- **sales**: Transaction-level sales data
- **products**: Product categories and pricing
- **regions**: Geographic sales regions

## Visualizations

- Interactive time series plots
- Seasonal decomposition analysis
- Feature importance charts
- Performance metrics dashboard
- Forecast confidence intervals

## 🔧 Configuration

Customize the forecasting system:

```python
# Adjust lag features
forecaster.create_lag_features(df, lags=5)

# Change forecast horizon
forecasts = forecaster.forecast_future(df, periods=12)

# Use different models
forecaster.models['xgboost'] = XGBRegressor()
```

## Sample Output

```
🛒 Sales Forecasting System
==================================================
📊 Loaded 48 months of sales data
📈 Sales range: $850 - $1,847
🔄 Training set: 35 samples
🔄 Test set: 8 samples

🤖 Training Models...
Linear Regression: CV R² = 0.8234 (±0.1156)
Random Forest: CV R² = 0.8567 (±0.0987)

Best model: Random Forest

📊 Evaluating Model...
📈 Mean Squared Error: 2847.32
📈 Root Mean Squared Error: 53.35
📈 Mean Absolute Error: 41.23
📈 R² Score: 0.8745
```

## Future Enhancements

- [ ] ARIMA and SARIMA models
- [ ] Prophet integration for advanced seasonality
- [ ] Real-time data streaming
- [ ] A/B testing framework
- [ ] Automated model retraining
- [ ] API endpoints for integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Scikit-learn for machine learning algorithms
- Streamlit for the interactive web interface
- Plotly for beautiful visualizations
- Pandas for data manipulation

## Contact

For questions or suggestions, please open an issue in the repository.

# Time-Series-Forecasting-for-Sales
