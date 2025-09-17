"""
Streamlit Web Application for Sales Forecasting
Interactive UI for time series sales forecasting with visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sales_forecaster import SalesForecaster
from data_generator import generate_sales_data, SalesDataGenerator

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(use_database=False):
    """Load sales data with caching."""
    return generate_sales_data(use_database=use_database)

@st.cache_resource
def initialize_forecaster():
    """Initialize forecaster with caching."""
    return SalesForecaster()

def create_interactive_plot(df, forecasts=None, title="Sales Data"):
    """Create interactive plotly chart."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Sales'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Forecasts
    if forecasts is not None:
        fig.add_trace(go.Scatter(
            x=forecasts.index,
            y=forecasts['Forecasted_Sales'],
            mode='lines+markers',
            name='Forecasted Sales',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def create_metrics_dashboard(df, metrics=None):
    """Create metrics dashboard."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales",
            value=f"${df['Sales'].sum():,.0f}",
            delta=f"{df['Sales'].pct_change().iloc[-1]*100:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Average Monthly Sales",
            value=f"${df['Sales'].mean():,.0f}",
            delta=f"{df['Sales'].std():,.0f} std"
        )
    
    with col3:
        st.metric(
            label="Max Sales",
            value=f"${df['Sales'].max():,.0f}",
            delta=f"Month {df['Sales'].idxmax().strftime('%Y-%m')}"
        )
    
    with col4:
        if metrics:
            st.metric(
                label="Model R² Score",
                value=f"{metrics['r2']:.3f}",
                delta=f"RMSE: {metrics['rmse']:.0f}"
            )
        else:
            st.metric(
                label="Data Points",
                value=len(df),
                delta="Months"
            )

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Data source selection
    use_database = st.sidebar.checkbox("Use Database Data", value=False, 
                                     help="Use SQLite database with realistic sales data")
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    forecast_periods = st.sidebar.slider("Forecast Periods (months)", 1, 12, 6)
    
    # Load data
    with st.spinner("Loading sales data..."):
        df = load_data(use_database=use_database)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 Forecasting", "📈 Analysis", "🗄️ Data"])
    
    with tab1:
        st.header("Sales Overview")
        
        # Metrics
        create_metrics_dashboard(df)
        
        # Historical sales chart
        fig = create_interactive_plot(df, title="Historical Sales Data")
        st.plotly_chart(fig, use_container_width=True)
        
        # Recent trends
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Trends (Last 12 Months)")
            recent_data = df.tail(12)
            
            # Calculate trend
            trend = recent_data['Sales'].pct_change().mean() * 100
            trend_color = "green" if trend > 0 else "red"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>Monthly Growth Rate</h4>
                <h2 style="color: {trend_color};">{trend:.2f}%</h2>
                <p>Average monthly change in sales</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Seasonal Patterns")
            monthly_avg = df.groupby(df.index.month)['Sales'].mean()
            
            fig_seasonal = px.bar(
                x=monthly_avg.index,
                y=monthly_avg.values,
                title="Average Sales by Month",
                labels={'x': 'Month', 'y': 'Average Sales ($)'}
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with tab2:
        st.header("Sales Forecasting")
        
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Training models and generating forecasts..."):
                # Initialize forecaster
                forecaster = initialize_forecaster()
                
                # Prepare data
                X_train, X_test, y_train, y_test = forecaster.prepare_data(df)
                
                # Train models
                forecaster.train_models(X_train, y_train)
                
                # Evaluate
                y_pred, metrics = forecaster.evaluate_model(X_test, y_test)
                
                # Store results in session state
                st.session_state['forecaster'] = forecaster
                st.session_state['metrics'] = metrics
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                
                # Generate future forecasts
                future_forecasts = forecaster.forecast_future(df, periods=forecast_periods)
                st.session_state['forecasts'] = future_forecasts
            
            st.success("✅ Forecasting complete!")
        
        # Display results if available
        if 'forecaster' in st.session_state:
            metrics = st.session_state['metrics']
            forecasts = st.session_state['forecasts']
            
            # Model performance
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{metrics['r2']:.4f}")
            with col2:
                st.metric("RMSE", f"{metrics['rmse']:.0f}")
            with col3:
                st.metric("MAE", f"{metrics['mae']:.0f}")
            with col4:
                st.metric("MSE", f"{metrics['mse']:.0f}")
            
            # Forecast visualization
            st.subheader("Sales Forecast")
            
            # Combine historical and forecast data for plotting
            combined_fig = create_interactive_plot(df.tail(24), forecasts, 
                                                 "Sales Forecast - Next 6 Months")
            st.plotly_chart(combined_fig, use_container_width=True)
            
            # Forecast table
            st.subheader("Forecast Details")
            forecast_display = forecasts.copy()
            forecast_display['Forecasted_Sales'] = forecast_display['Forecasted_Sales'].round(0)
            forecast_display['Month'] = forecast_display.index.strftime('%Y-%m')
            st.dataframe(forecast_display[['Month', 'Forecasted_Sales']], use_container_width=True)
    
    with tab3:
        st.header("Advanced Analysis")
        
        # Decomposition analysis
        st.subheader("Sales Decomposition")
        
        # Simple trend analysis
        df_analysis = df.copy()
        df_analysis['Rolling_Mean_6'] = df_analysis['Sales'].rolling(window=6).mean()
        df_analysis['Rolling_Mean_12'] = df_analysis['Sales'].rolling(window=12).mean()
        
        fig_decomp = go.Figure()
        fig_decomp.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['Sales'], 
                                      name='Original', line=dict(color='blue')))
        fig_decomp.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['Rolling_Mean_6'], 
                                      name='6-Month Trend', line=dict(color='red')))
        fig_decomp.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['Rolling_Mean_12'], 
                                      name='12-Month Trend', line=dict(color='green')))
        
        fig_decomp.update_layout(title="Sales Trends Analysis", xaxis_title="Date", 
                               yaxis_title="Sales ($)", height=400)
        st.plotly_chart(fig_decomp, use_container_width=True)
        
        # Correlation analysis
        if 'forecaster' in st.session_state:
            st.subheader("Feature Importance")
            forecaster = st.session_state['forecaster']
            
            if forecaster.best_model_name == 'random_forest':
                importance_df = pd.DataFrame({
                    'Feature': forecaster.feature_names,
                    'Importance': forecaster.best_model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(importance_df, x='Importance', y='Feature', 
                                      orientation='h', title="Feature Importance")
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.info("Feature importance is only available for Random Forest model.")
        
        # Statistical summary
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab4:
        st.header("Data Management")
        
        # Data source info
        if use_database:
            st.info("📊 Using SQLite database with realistic sales patterns")
            
            # Database statistics
            generator = SalesDataGenerator()
            if os.path.exists(generator.db_path):
                conn = sqlite3.connect(generator.db_path)
                
                # Get record counts
                sales_count = pd.read_sql_query("SELECT COUNT(*) as count FROM sales", conn).iloc[0]['count']
                products_count = pd.read_sql_query("SELECT COUNT(*) as count FROM products", conn).iloc[0]['count']
                regions_count = pd.read_sql_query("SELECT COUNT(*) as count FROM regions", conn).iloc[0]['count']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sales Records", sales_count)
                with col2:
                    st.metric("Products", products_count)
                with col3:
                    st.metric("Regions", regions_count)
                
                conn.close()
        else:
            st.info("📈 Using generated synthetic sales data")
        
        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Data download
        st.subheader("Export Data")
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Sales Data (CSV)",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Regenerate data
        if st.button("🔄 Regenerate Database", type="secondary"):
            if use_database:
                with st.spinner("Regenerating database..."):
                    generator = SalesDataGenerator()
                    if os.path.exists(generator.db_path):
                        os.remove(generator.db_path)
                    generator.setup_database()
                    generator.populate_database(2000)
                st.success("✅ Database regenerated successfully!")
                st.experimental_rerun()

if __name__ == "__main__":
    main()
