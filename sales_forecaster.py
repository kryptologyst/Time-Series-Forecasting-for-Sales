"""
Time Series Forecasting for Sales
A comprehensive sales forecasting system using machine learning techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    """
    A class for time series sales forecasting using multiple ML algorithms.
    """
    
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def create_lag_features(self, df, target_col='Sales', lags=3):
        """
        Create lag features for time series forecasting.
        
        Args:
            df (pd.DataFrame): Input dataframe with time series data
            target_col (str): Name of the target column
            lags (int): Number of lag features to create
            
        Returns:
            pd.DataFrame: DataFrame with lag features
        """
        df_copy = df.copy()
        
        # Create lag features
        for lag in range(1, lags + 1):
            df_copy[f'Lag_{lag}'] = df_copy[target_col].shift(lag)
        
        # Create rolling statistics
        for window in [3, 6]:
            df_copy[f'Rolling_Mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
            df_copy[f'Rolling_Std_{window}'] = df_copy[target_col].rolling(window=window).std()
        
        # Create trend features
        df_copy['Month'] = df_copy.index.month
        df_copy['Quarter'] = df_copy.index.quarter
        df_copy['Year'] = df_copy.index.year
        
        # Drop rows with NaN values
        df_copy.dropna(inplace=True)
        
        return df_copy
    
    def prepare_data(self, df, target_col='Sales'):
        """
        Prepare data for training by creating features and splitting.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Create features
        df_features = self.create_lag_features(df, target_col)
        
        # Define features (exclude target column)
        feature_cols = [col for col in df_features.columns if col != target_col]
        self.feature_names = feature_cols
        
        X = df_features[feature_cols]
        y = df_features[target_col]
        
        # Split data (preserve time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and select the best one.
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        best_score = -np.inf
        
        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            avg_score = cv_scores.mean()
            
            print(f"{name.replace('_', ' ').title()}: CV R² = {avg_score:.4f} (±{cv_scores.std():.4f})")
            
            if avg_score > best_score:
                best_score = avg_score
                self.best_model = model
                self.best_model_name = name
        
        # Train the best model on full training data
        self.best_model.fit(X_train, y_train)
        print(f"\nBest model: {self.best_model_name.replace('_', ' ').title()}")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return y_pred, metrics
    
    def plot_results(self, y_test, y_pred, metrics):
        """
        Create visualization plots for the forecasting results.
        
        Args:
            y_test: Actual test values
            y_pred: Predicted values
            metrics: Evaluation metrics dictionary
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Actual vs Predicted
        axes[0, 0].plot(y_test.index, y_test.values, label='Actual Sales', marker='o')
        axes[0, 0].plot(y_test.index, y_pred, label='Predicted Sales', marker='s', linestyle='--')
        axes[0, 0].set_title('Actual vs Predicted Sales')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Sales')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        residuals = y_test.values - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].set_xlabel('Predicted Sales')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Feature Importance (if Random Forest)
        if self.best_model_name == 'random_forest':
            importance = self.best_model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=True)
            
            axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
            axes[1, 0].set_title('Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance\nnot available for\nLinear Regression', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')
        
        # Plot 4: Metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].set_ylabel('Value')
        
        # Add metric values on bars
        for i, v in enumerate(metric_values):
            axes[1, 1].text(i, v + max(metric_values) * 0.01, f'{v:.3f}', 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def forecast_future(self, df, periods=6):
        """
        Forecast future sales for the specified number of periods.
        
        Args:
            df (pd.DataFrame): Historical data
            periods (int): Number of future periods to forecast
            
        Returns:
            pd.DataFrame: Future forecasts
        """
        if self.best_model is None:
            raise ValueError("Model must be trained before forecasting")
        
        # Get the last few values for creating lag features
        last_values = df['Sales'].tail(10).values
        future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), 
                                   periods=periods, freq='ME')
        
        forecasts = []
        
        # Get the most recent complete feature set from training data
        df_with_features = self.create_lag_features(df)
        if len(df_with_features) == 0:
            raise ValueError("Not enough historical data to create features")
        
        last_feature_row = df_with_features[self.feature_names].iloc[-1:].values
        
        for i in range(periods):
            # For simplicity, use the last known feature pattern
            # In a production system, you'd implement proper recursive forecasting
            if i == 0:
                features = last_feature_row
            else:
                # Use a simple approach: adjust the lag features with recent forecasts
                features = last_feature_row.copy()
                # Update lag features with recent forecasts if available
                if len(forecasts) >= 1:
                    features[0][0] = forecasts[-1]  # Lag_1
                if len(forecasts) >= 2:
                    features[0][1] = forecasts[-2]  # Lag_2
                if len(forecasts) >= 3:
                    features[0][2] = forecasts[-3]  # Lag_3
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            forecast = self.best_model.predict(features_scaled)[0]
            forecasts.append(forecast)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Forecasted_Sales': forecasts
        })
        forecast_df.set_index('Date', inplace=True)
        
        return forecast_df

def main():
    """
    Main function to run the sales forecasting pipeline.
    """
    print("🛒 Sales Forecasting System")
    print("=" * 50)
    
    # Initialize forecaster
    forecaster = SalesForecaster()
    
    # Load data (this will be replaced with database data)
    from data_generator import generate_sales_data
    df = generate_sales_data()
    
    print(f"📊 Loaded {len(df)} months of sales data")
    print(f"📈 Sales range: ${df['Sales'].min():.0f} - ${df['Sales'].max():.0f}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
    print(f"🔄 Training set: {len(X_train)} samples")
    print(f"🔄 Test set: {len(X_test)} samples")
    
    # Train models
    print("\n🤖 Training Models...")
    forecaster.train_models(X_train, y_train)
    
    # Evaluate
    print("\n📊 Evaluating Model...")
    y_pred, metrics = forecaster.evaluate_model(X_test, y_test)
    
    print(f"📈 Mean Squared Error: {metrics['mse']:.2f}")
    print(f"📈 Root Mean Squared Error: {metrics['rmse']:.2f}")
    print(f"📈 Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"📈 R² Score: {metrics['r2']:.4f}")
    
    # Plot results
    forecaster.plot_results(y_test, y_pred, metrics)
    
    # Future forecasting
    print("\n🔮 Generating Future Forecasts...")
    future_forecasts = forecaster.forecast_future(df, periods=6)
    print(future_forecasts)
    
    # Plot future forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-12:], df['Sales'].tail(12), label='Historical Sales', marker='o')
    plt.plot(future_forecasts.index, future_forecasts['Forecasted_Sales'], 
             label='Forecasted Sales', marker='s', linestyle='--', color='red')
    plt.title('Sales Forecast - Next 6 Months')
    plt.xlabel('Date')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return forecaster, df, future_forecasts

if __name__ == "__main__":
    forecaster, historical_data, forecasts = main()
