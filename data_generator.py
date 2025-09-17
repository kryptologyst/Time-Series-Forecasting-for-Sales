"""
Mock Database and Data Generator for Sales Forecasting
Generates realistic sales data with seasonal patterns and trends.
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta

class SalesDataGenerator:
    """
    Generates realistic sales data with various patterns and stores in SQLite database.
    """
    
    def __init__(self, db_path='sales_data.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """
        Create SQLite database and tables for sales data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sales table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                sales_amount REAL NOT NULL,
                product_category TEXT,
                region TEXT,
                promotion BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                base_price REAL NOT NULL,
                seasonal_factor REAL DEFAULT 1.0
            )
        ''')
        
        # Create regions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                population INTEGER,
                economic_factor REAL DEFAULT 1.0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_base_sales_data(self, start_date='2020-01-01', periods=48):
        """
        Generate base sales data with realistic patterns.
        
        Args:
            start_date (str): Start date for data generation
            periods (int): Number of months to generate
            
        Returns:
            pd.DataFrame: Generated sales data
        """
        np.random.seed(42)
        
        # Create date range
        dates = pd.date_range(start=start_date, periods=periods, freq='ME')
        
        # Base sales trend (growing over time)
        base_trend = np.linspace(1000, 1500, periods)
        
        # Seasonal pattern (higher sales in Q4, lower in Q1)
        seasonal_pattern = []
        for date in dates:
            month = date.month
            if month in [11, 12]:  # Holiday season
                seasonal_factor = 1.4
            elif month in [1, 2]:  # Post-holiday slump
                seasonal_factor = 0.8
            elif month in [6, 7, 8]:  # Summer boost
                seasonal_factor = 1.2
            else:
                seasonal_factor = 1.0
            seasonal_pattern.append(seasonal_factor)
        
        # Economic cycles (simulate business cycles)
        economic_cycle = 1 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, periods))
        
        # Random noise
        noise = np.random.normal(0, 50, periods)
        
        # Combine all factors
        sales = base_trend * seasonal_pattern * economic_cycle + noise
        
        # Ensure no negative sales
        sales = np.maximum(sales, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Date': dates,
            'Sales': sales,
            'Seasonal_Factor': seasonal_pattern,
            'Economic_Cycle': economic_cycle,
            'Base_Trend': base_trend
        })
        
        return df
    
    def add_promotional_effects(self, df, promo_probability=0.15):
        """
        Add promotional effects to sales data.
        
        Args:
            df (pd.DataFrame): Base sales data
            promo_probability (float): Probability of promotion in any given month
            
        Returns:
            pd.DataFrame: Sales data with promotional effects
        """
        np.random.seed(42)
        
        # Randomly assign promotions
        promotions = np.random.random(len(df)) < promo_probability
        
        # Promotional boost (20-50% increase)
        promo_boost = np.random.uniform(1.2, 1.5, len(df))
        
        # Apply promotional effects
        df['Promotion'] = promotions
        df['Promo_Boost'] = np.where(promotions, promo_boost, 1.0)
        df['Sales'] = df['Sales'] * df['Promo_Boost']
        
        return df
    
    def populate_database(self, num_records=1000):
        """
        Populate the database with sample data.
        
        Args:
            num_records (int): Number of records to generate
        """
        conn = sqlite3.connect(self.db_path)
        
        # Sample products
        products = [
            ('Electronics', 'Electronics', 500, 1.2),
            ('Clothing', 'Fashion', 100, 1.5),
            ('Home & Garden', 'Home', 200, 1.1),
            ('Sports Equipment', 'Sports', 150, 0.9),
            ('Books', 'Education', 25, 1.0)
        ]
        
        # Sample regions
        regions = [
            ('North', 1000000, 1.2),
            ('South', 800000, 1.0),
            ('East', 1200000, 1.3),
            ('West', 900000, 1.1),
            ('Central', 700000, 0.9)
        ]
        
        # Insert products
        conn.executemany('''
            INSERT OR REPLACE INTO products (name, category, base_price, seasonal_factor)
            VALUES (?, ?, ?, ?)
        ''', products)
        
        # Insert regions
        conn.executemany('''
            INSERT OR REPLACE INTO regions (name, population, economic_factor)
            VALUES (?, ?, ?)
        ''', regions)
        
        # Generate sales records
        np.random.seed(42)
        sales_records = []
        
        start_date = datetime(2020, 1, 1)
        
        for i in range(num_records):
            # Random date within the last 4 years
            random_days = np.random.randint(0, 1460)  # 4 years
            record_date = start_date + timedelta(days=random_days)
            
            # Random product and region
            product = np.random.choice(products)
            region = np.random.choice(regions)
            
            # Calculate sales amount with various factors
            base_amount = product[2] * region[2]  # base_price * economic_factor
            
            # Seasonal adjustment
            month = record_date.month
            if month in [11, 12]:
                seasonal_adj = product[3] * 1.4  # seasonal_factor * holiday boost
            elif month in [1, 2]:
                seasonal_adj = product[3] * 0.8
            else:
                seasonal_adj = product[3]
            
            # Random variation
            variation = np.random.uniform(0.7, 1.3)
            
            # Promotion (15% chance)
            promotion = np.random.random() < 0.15
            promo_boost = 1.3 if promotion else 1.0
            
            final_amount = base_amount * seasonal_adj * variation * promo_boost
            
            sales_records.append((
                record_date.strftime('%Y-%m-%d'),
                round(final_amount, 2),
                product[1],  # category
                region[0],   # region name
                promotion
            ))
        
        # Insert sales records
        conn.executemany('''
            INSERT INTO sales (date, sales_amount, product_category, region, promotion)
            VALUES (?, ?, ?, ?, ?)
        ''', sales_records)
        
        conn.commit()
        conn.close()
        
        print(f"✅ Database populated with {num_records} sales records")
    
    def get_aggregated_monthly_data(self):
        """
        Retrieve aggregated monthly sales data from database.
        
        Returns:
            pd.DataFrame: Monthly aggregated sales data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                DATE(date, 'start of month') as month,
                SUM(sales_amount) as total_sales,
                COUNT(*) as transaction_count,
                AVG(sales_amount) as avg_transaction,
                SUM(CASE WHEN promotion = 1 THEN 1 ELSE 0 END) as promo_count
            FROM sales
            GROUP BY DATE(date, 'start of month')
            ORDER BY month
        '''
        
        df = pd.read_sql_query(query, conn)
        
        # Check if we have data
        if len(df) == 0:
            conn.close()
            return pd.DataFrame(columns=['total_sales'])
        
        df['month'] = pd.to_datetime(df['month'])
        df.set_index('month', inplace=True)
        
        conn.close()
        
        return df

def generate_sales_data(use_database=False):
    """
    Generate or retrieve sales data for forecasting.
    
    Args:
        use_database (bool): Whether to use database or generate simple data
        
    Returns:
        pd.DataFrame: Sales data ready for forecasting
    """
    if use_database:
        # Use database approach
        generator = SalesDataGenerator()
        
        # Check if database exists and has data
        if not os.path.exists(generator.db_path):
            print("📊 Creating new sales database...")
            generator.populate_database(2000)
        
        # Get monthly aggregated data
        df = generator.get_aggregated_monthly_data()
        
        # Check if we have data
        if len(df) == 0:
            print("⚠️ No data in database, generating fresh data...")
            generator.populate_database(2000)
            df = generator.get_aggregated_monthly_data()
        
        df.rename(columns={'total_sales': 'Sales'}, inplace=True)
        
        return df[['Sales']]
    
    else:
        # Use simple generation approach (original method enhanced)
        generator = SalesDataGenerator()
        df = generator.generate_base_sales_data(periods=48)
        df = generator.add_promotional_effects(df)
        df.set_index('Date', inplace=True)
        
        return df[['Sales']]

if __name__ == "__main__":
    # Test the data generator
    print("🧪 Testing Sales Data Generator")
    print("=" * 40)
    
    # Generate simple data
    print("📈 Generating simple sales data...")
    simple_data = generate_sales_data(use_database=False)
    print(f"Generated {len(simple_data)} months of data")
    print(simple_data.head())
    
    # Generate database data
    print("\n📊 Generating database sales data...")
    db_data = generate_sales_data(use_database=True)
    print(f"Generated {len(db_data)} months of data")
    print(db_data.head())
    
    print("\n✅ Data generation complete!")
