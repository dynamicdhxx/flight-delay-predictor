"""
Flight Delay Prediction - Flask Web Application
================================================
A modern web interface for flight delay prediction and route analysis.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'flight-delay-secret-key'
app.config['UPLOAD_FOLDER'] = 'data/raw'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Global variables for model and data
model = None
scaler = None
df = None
feature_columns = None

# Airlines and airports for the form
AIRLINES = [
    'Qantas', 'Virgin Australia', 'Jetstar', 'American Airlines',
    'Delta Air Lines', 'United Airlines', 'Southwest Airlines',
    'British Airways', 'Emirates', 'Singapore Airlines'
]

AIRPORTS = [
    'SYD', 'MEL', 'BNE', 'PER', 'ADL', 'LAX', 'JFK', 'ORD', 
    'DFW', 'DEN', 'SFO', 'SEA', 'ATL', 'LHR', 'CDG', 'DXB', 'SIN'
]

BOOKING_CLASSES = ['Economy', 'Premium Economy', 'Business', 'First']


def load_model():
    """Load trained model if available."""
    global model, scaler
    model_path = 'models/delay_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)


def load_data():
    """Load processed flight data."""
    global df
    
    # Try different data paths
    data_paths = [
        'data/processed/flights_cleaned.csv',
        'data/raw/flights.csv',
        'data/raw/Flight_data.csv'
    ]
    
    for data_path in data_paths:
        if os.path.exists(data_path):
            try:
                df = pd.read_csv(data_path)
                print(f"✓ Loaded data from {data_path}: {len(df)} records")
                
                # Create arrival_delay if not exists
                if 'arrival_delay' not in df.columns:
                    delay_cols = [c for c in df.columns if 'delay' in c.lower()]
                    if delay_cols:
                        df['arrival_delay'] = pd.to_numeric(df[delay_cols[0]], errors='coerce').fillna(0)
                    else:
                        # Create synthetic delay for demo
                        import numpy as np
                        np.random.seed(42)
                        df['arrival_delay'] = np.random.normal(5, 20, len(df)).clip(-30, 120)
                
                # Create is_delayed if not exists
                if 'is_delayed' not in df.columns:
                    df['is_delayed'] = (df['arrival_delay'] >= 15).astype(int)
                
                return True
            except Exception as e:
                print(f"Error loading {data_path}: {e}")
                continue
    
    return False


def get_dashboard_stats():
    """Calculate dashboard statistics."""
    default_stats = {
        'total_flights': 0,
        'delay_rate': 0,
        'avg_delay': 0,
        'on_time_rate': 100,
        'total_routes': 0,
        'total_airlines': 0
    }
    
    if df is None or len(df) == 0:
        return default_stats
    
    try:
        # Find delay column
        delay_col = 'arrival_delay'
        if delay_col not in df.columns:
            for col in df.columns:
                if 'delay' in col.lower():
                    delay_col = col
                    break
        
        # Calculate delay stats
        if delay_col in df.columns:
            delays = pd.to_numeric(df[delay_col], errors='coerce').dropna()
            if len(delays) > 0:
                avg_delay = float(delays.mean())
                delay_rate = float((delays >= 15).mean() * 100)
            else:
                avg_delay = 0
                delay_rate = 0
        else:
            avg_delay = 0
            delay_rate = 0
        
        # Count routes
        total_routes = 0
        if 'Route' in df.columns:
            total_routes = df['Route'].nunique()
        elif 'origin' in df.columns and 'destination' in df.columns:
            total_routes = len(df.groupby(['origin', 'destination']))
        
        # Count airlines
        total_airlines = 0
        for col in df.columns:
            if 'airline' in col.lower() or 'carrier' in col.lower():
                total_airlines = df[col].nunique()
                break
        
        return {
            'total_flights': int(len(df)),
            'delay_rate': round(delay_rate, 1),
            'avg_delay': round(avg_delay, 1),
            'on_time_rate': round(100 - delay_rate, 1),
            'total_routes': int(total_routes),
            'total_airlines': int(total_airlines)
        }
    except Exception as e:
        print(f"Error calculating stats: {e}")
        return default_stats


def get_chart_data():
    """Get data for charts."""
    # Default empty chart data
    default_charts = {
        'hourly': {'labels': [], 'values': []},
        'airline': {'labels': [], 'values': []},
        'distribution': {'labels': ['Early', 'On-Time', 'Minor Delay', 'Moderate', 'Severe'], 'values': [0, 0, 0, 0, 0]}
    }
    
    if df is None or len(df) == 0:
        return default_charts
    
    # Find delay column
    delay_col = None
    for col in ['arrival_delay', 'Arrival_Delay_Minutes', 'Departure Delay in Minutes']:
        if col in df.columns:
            delay_col = col
            break
    
    if delay_col is None:
        return default_charts
    
    try:
        # Hourly delays
        hourly_data = {'labels': [], 'values': []}
        hour_col = None
        for col in ['hour', 'Hour', 'Scheduled_Departure']:
            if col in df.columns:
                hour_col = col
                break
        
        if hour_col:
            if df[hour_col].dtype == 'object' or 'datetime' in str(df[hour_col].dtype):
                try:
                    df['_hour'] = pd.to_datetime(df[hour_col], errors='coerce').dt.hour
                    hour_col = '_hour'
                except:
                    pass
            
            if hour_col in df.columns:
                hourly = df.groupby(hour_col)[delay_col].mean().dropna()
                if len(hourly) > 0:
                    hourly_data = {
                        'labels': [f'{int(h)}:00' for h in hourly.index.tolist()],
                        'values': [round(float(v), 1) for v in hourly.values.tolist()]
                    }
        
        # Airline delays
        airline_data = {'labels': [], 'values': []}
        airline_col = None
        for col in df.columns:
            if 'airline' in col.lower() or 'carrier' in col.lower():
                airline_col = col
                break
        
        if airline_col:
            airline_delays = df.groupby(airline_col)[delay_col].mean().sort_values(ascending=False).head(10).dropna()
            if len(airline_delays) > 0:
                airline_data = {
                    'labels': [str(l) for l in airline_delays.index.tolist()],
                    'values': [round(float(v), 1) for v in airline_delays.values.tolist()]
                }
        
        # Delay distribution
        delays = pd.to_numeric(df[delay_col], errors='coerce').dropna()
        if len(delays) > 0:
            default_charts['distribution']['values'] = [
                int((delays < 0).sum()),
                int(((delays >= 0) & (delays < 15)).sum()),
                int(((delays >= 15) & (delays < 30)).sum()),
                int(((delays >= 30) & (delays < 60)).sum()),
                int((delays >= 60).sum())
            ]
        
        default_charts['hourly'] = hourly_data
        default_charts['airline'] = airline_data
        
    except Exception as e:
        print(f"Error generating chart data: {e}")
    
    return default_charts


@app.route('/')
def index():
    """Home page / Dashboard."""
    load_data()
    stats = get_dashboard_stats()
    charts = get_chart_data()
    return render_template('index.html', stats=stats, charts=charts)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page."""
    prediction = None
    probability = None
    
    if request.method == 'POST':
        # Get form data
        airline = request.form.get('airline', '')
        origin = request.form.get('origin', '')
        destination = request.form.get('destination', '')
        hour = int(request.form.get('hour', 12))
        day_of_week = int(request.form.get('day_of_week', 0))
        month = int(request.form.get('month', 1))
        booking_class = request.form.get('booking_class', 'Economy')
        
        # Simple rule-based prediction if no model
        # Higher delay probability for:
        # - Afternoon/evening flights (14-20)
        # - Fridays and Sundays
        # - Summer months (6-8) and winter (12-2)
        # - Budget airlines
        
        base_prob = 0.25
        
        # Time of day effect
        if 14 <= hour <= 20:
            base_prob += 0.15
        elif hour >= 21 or hour <= 5:
            base_prob += 0.05
        
        # Day of week effect
        if day_of_week in [4, 6]:  # Friday, Sunday
            base_prob += 0.08
        
        # Seasonal effect
        if month in [6, 7, 8]:
            base_prob += 0.1
        elif month in [12, 1, 2]:
            base_prob += 0.12
        
        # Airline effect
        budget_airlines = ['Jetstar', 'Southwest Airlines']
        premium_airlines = ['Emirates', 'Singapore Airlines', 'Qantas']
        if airline in budget_airlines:
            base_prob += 0.1
        elif airline in premium_airlines:
            base_prob -= 0.08
        
        # Class effect
        if booking_class in ['Business', 'First']:
            base_prob -= 0.05
        
        probability = min(max(base_prob, 0.05), 0.85)
        prediction = 'Delayed' if probability > 0.4 else 'On-Time'
    
    return render_template('predict.html', 
                          airlines=AIRLINES, 
                          airports=AIRPORTS,
                          booking_classes=BOOKING_CLASSES,
                          prediction=prediction,
                          probability=probability)


@app.route('/analytics')
def analytics():
    """Analytics page."""
    load_data()
    stats = get_dashboard_stats()
    charts = get_chart_data()
    return render_template('analytics.html', stats=stats, charts=charts)


@app.route('/routes')
def routes():
    """Route analysis page."""
    load_data()
    
    route_data = []
    if df is not None:
        delay_col = 'arrival_delay' if 'arrival_delay' in df.columns else 'Arrival_Delay_Minutes'
        
        if 'Route' in df.columns and delay_col in df.columns:
            route_stats = df.groupby('Route').agg({
                delay_col: ['mean', 'count']
            }).reset_index()
            route_stats.columns = ['route', 'avg_delay', 'flights']
            route_stats['delay_rate'] = df.groupby('Route').apply(
                lambda x: (x[delay_col] >= 15).mean() * 100
            ).values
            route_data = route_stats.sort_values('flights', ascending=False).head(20).to_dict('records')
    
    return render_template('routes.html', routes=route_data)


@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard stats."""
    load_data()
    return jsonify(get_dashboard_stats())


@app.route('/api/charts')
def api_charts():
    """API endpoint for chart data."""
    load_data()
    return jsonify(get_chart_data())


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions."""
    data = request.json
    
    # Simple prediction logic
    hour = data.get('hour', 12)
    day_of_week = data.get('day_of_week', 0)
    month = data.get('month', 1)
    
    base_prob = 0.25
    if 14 <= hour <= 20:
        base_prob += 0.15
    if day_of_week in [4, 6]:
        base_prob += 0.08
    if month in [6, 7, 8, 12, 1, 2]:
        base_prob += 0.1
    
    probability = min(max(base_prob, 0.05), 0.85)
    
    return jsonify({
        'prediction': 'Delayed' if probability > 0.4 else 'On-Time',
        'probability': round(probability, 3),
        'confidence': round((1 - abs(probability - 0.5) * 2) * 100, 1)
    })


# Create necessary directories
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)


if __name__ == '__main__':
    load_model()
    load_data()
    app.run(debug=True, port=5000)

