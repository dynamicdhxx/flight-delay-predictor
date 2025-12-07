"""
Sample Data Generator & Kaggle Dataset Loader
==============================================

Generate synthetic flight delay data or load the Kaggle Flight Analytics Dataset.
Dataset: https://www.kaggle.com/datasets/goyaladi/flight-dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import random
import os


def load_kaggle_dataset(filepath: str = None) -> pd.DataFrame:
    """
    Load the Kaggle Flight Analytics Dataset.
    
    Download from: https://www.kaggle.com/datasets/goyaladi/flight-dataset
    Place the Flight_data.csv file in data/raw/
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file. Default: data/raw/Flight_data.csv
        
    Returns
    -------
    pd.DataFrame
        Loaded flight data
    """
    if filepath is None:
        filepath = 'data/raw/Flight_data.csv'
    
    if not os.path.exists(filepath):
        print("="*70)
        print("⚠️  KAGGLE DATASET NOT FOUND")
        print("="*70)
        print(f"\nPlease download the dataset from:")
        print("https://www.kaggle.com/datasets/goyaladi/flight-dataset")
        print(f"\nThen place 'Flight_data.csv' in: {filepath}")
        print("\nAlternatively, use generate_sample_dataset() to create synthetic data.")
        print("="*70)
        return None
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded Kaggle dataset: {len(df):,} records, {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return df


def preprocess_kaggle_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Kaggle Flight Analytics Dataset for delay prediction.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw Kaggle dataset
        
    Returns
    -------
    pd.DataFrame
        Preprocessed dataset with derived features
    """
    df = df.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Parse date columns if present
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create delay-related features based on available columns
    # The Kaggle dataset may have columns like:
    # - Departure Delay in Minutes
    # - Arrival Delay in Minutes
    # - Flight Status
    
    # Find delay-related columns
    delay_cols = [col for col in df.columns if 'delay' in col.lower()]
    
    if delay_cols:
        # Use first delay column found as primary delay
        primary_delay_col = delay_cols[0]
        df['arrival_delay'] = pd.to_numeric(df[primary_delay_col], errors='coerce').fillna(0)
    else:
        # Create synthetic delay if not available
        print("⚠️  No delay columns found. Creating synthetic delay based on patterns.")
        np.random.seed(42)
        df['arrival_delay'] = np.random.normal(5, 25, len(df))
        df['arrival_delay'] = df['arrival_delay'].clip(-30, 180)
    
    # Create binary delay target (>15 minutes is delayed)
    df['is_delayed'] = (df['arrival_delay'] >= 15).astype(int)
    
    # Extract route components if route column exists
    route_col = [col for col in df.columns if 'route' in col.lower()]
    if route_col:
        route_col = route_col[0]
        # Split route into origin and destination
        route_split = df[route_col].str.split('-', expand=True)
        if route_split.shape[1] >= 2:
            df['origin'] = route_split[0].str.strip()
            df['destination'] = route_split[1].str.strip()
    
    # Generate flight_id if not present
    if 'flight_id' not in df.columns:
        df['flight_id'] = [f'FL{i:06d}' for i in range(len(df))]
    
    print(f"✓ Preprocessed dataset: {len(df.columns)} columns")
    print(f"  Delay rate: {df['is_delayed'].mean()*100:.1f}%")
    
    return df


# Major US/Australian airports for sample data
AIRPORTS = [
    'SYD', 'MEL', 'BNE', 'PER', 'ADL', 'CBR', 'OOL', 'HBA',
    'LAX', 'JFK', 'ORD', 'DFW', 'DEN', 'SFO', 'SEA', 'ATL',
    'LHR', 'CDG', 'FRA', 'AMS', 'DXB', 'SIN', 'HKG', 'NRT'
]

# Carriers
CARRIERS = {
    'QF': 'Qantas',
    'VA': 'Virgin Australia',
    'JQ': 'Jetstar',
    'TT': 'Tigerair',
    'AA': 'American Airlines',
    'DL': 'Delta Air Lines',
    'UA': 'United Airlines',
    'WN': 'Southwest Airlines',
    'BA': 'British Airways',
    'EK': 'Emirates'
}

# Carrier delay factors
CARRIER_DELAY_FACTOR = {
    'QF': 0.8, 'VA': 0.9, 'JQ': 1.2, 'TT': 1.3,
    'AA': 1.0, 'DL': 0.8, 'UA': 1.1, 'WN': 0.9,
    'BA': 0.85, 'EK': 0.7
}

# Booking classes
BOOKING_CLASSES = ['Economy', 'Premium Economy', 'Business', 'First']

# Frequent flyer statuses
FF_STATUSES = ['Bronze', 'Silver', 'Gold', 'Platinum', 'None']


def get_distance(origin: str, destination: str) -> int:
    """Get approximate distance between airports."""
    idx1 = AIRPORTS.index(origin) if origin in AIRPORTS else 0
    idx2 = AIRPORTS.index(destination) if destination in AIRPORTS else 0
    np.random.seed(idx1 * 100 + idx2)
    base_distance = abs(idx1 - idx2) * 200 + np.random.randint(300, 1000)
    return min(max(base_distance, 300), 5000)


def generate_sample_dataset(n_flights: int = 50000,
                           start_date: str = '2023-01-01',
                           end_date: str = '2023-12-31',
                           random_seed: int = 42,
                           save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic flight delay dataset matching Kaggle dataset structure.
    
    Parameters
    ----------
    n_flights : int
        Number of flights to generate
    start_date : str
        Start date for data
    end_date : str
        End date for data
    random_seed : int
        Random seed for reproducibility
    save_path : str
        Path to save CSV (optional)
        
    Returns
    -------
    pd.DataFrame
        Generated flight data
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"Generating {n_flights:,} synthetic flight records...")
    
    # Generate date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    date_range = pd.date_range(start, end, freq='H')
    
    # First names and last names for passenger generation
    first_names = ['James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard', 'Joseph',
                   'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Sarah',
                   'Emma', 'Oliver', 'Charlotte', 'Amelia', 'Lucas', 'Noah', 'Sophia', 'Liam']
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                  'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'White', 'Harris']
    
    data = []
    
    for i in range(n_flights):
        # Generate passenger info
        passenger_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        
        # Select origin and destination
        origin = random.choice(AIRPORTS)
        destination = random.choice([a for a in AIRPORTS if a != origin])
        route = f"{origin}-{destination}"
        
        # Select carrier
        carrier = random.choice(list(CARRIERS.keys()))
        
        # Generate flight details
        flight_number = f"{carrier}{random.randint(100, 9999)}"
        scheduled_departure = random.choice(date_range)
        
        # Distance and flight time
        distance = get_distance(origin, destination)
        scheduled_flight_time = distance / 8 + 30
        scheduled_arrival = scheduled_departure + timedelta(minutes=scheduled_flight_time)
        
        # Booking class and frequent flyer status
        booking_class = random.choices(BOOKING_CLASSES, weights=[50, 20, 20, 10])[0]
        ff_status = random.choices(FF_STATUSES, weights=[20, 25, 25, 15, 15])[0]
        
        # Calculate delay probability
        hour = scheduled_departure.hour
        month = scheduled_departure.month
        day_of_week = scheduled_departure.dayofweek
        
        delay_prob = 0.2
        
        # Time effects
        if 14 <= hour <= 20:
            delay_prob += 0.12
        elif hour >= 21 or hour <= 5:
            delay_prob += 0.05
        
        # Seasonal effects
        if month in [6, 7, 8]:
            delay_prob += 0.1
        elif month in [12, 1, 2]:
            delay_prob += 0.15
        
        # Weekend effect
        if day_of_week in [4, 5, 6]:
            delay_prob += 0.05
        
        # Carrier effect
        delay_prob *= CARRIER_DELAY_FACTOR.get(carrier, 1.0)
        
        # Premium passengers less likely to be on delayed flights (selection bias)
        if booking_class in ['Business', 'First']:
            delay_prob *= 0.85
        if ff_status in ['Gold', 'Platinum']:
            delay_prob *= 0.9
        
        delay_prob = min(delay_prob, 0.55)
        
        # Generate delay
        is_delayed = np.random.random() < delay_prob
        
        if is_delayed:
            delay_minutes = np.random.lognormal(mean=3, sigma=0.8)
            delay_minutes = max(15, min(delay_minutes, 300))
        else:
            delay_minutes = np.random.normal(loc=-5, scale=10)
            delay_minutes = max(-30, min(delay_minutes, 14))
        
        departure_delay = delay_minutes * (0.6 + 0.4 * np.random.random())
        arrival_delay = delay_minutes
        
        # Actual times
        actual_departure = scheduled_departure + timedelta(minutes=departure_delay)
        actual_arrival = scheduled_arrival + timedelta(minutes=arrival_delay)
        
        # Cancellation (rare)
        is_cancelled = np.random.random() < 0.015
        
        if is_cancelled:
            actual_departure = pd.NaT
            actual_arrival = pd.NaT
            flight_status = 'Cancelled'
        elif arrival_delay >= 15:
            flight_status = 'Delayed'
        elif arrival_delay <= -5:
            flight_status = 'Early'
        else:
            flight_status = 'On-Time'
        
        # Generate ticket price based on class and distance
        base_price = distance * 0.15
        class_multiplier = {'Economy': 1, 'Premium Economy': 1.5, 'Business': 3, 'First': 5}
        ticket_price = base_price * class_multiplier[booking_class] * (0.8 + 0.4 * np.random.random())
        
        # Customer satisfaction (influenced by delay and class)
        base_satisfaction = 4.0
        if arrival_delay > 60:
            base_satisfaction -= 1.5
        elif arrival_delay > 30:
            base_satisfaction -= 0.8
        elif arrival_delay > 15:
            base_satisfaction -= 0.4
        elif arrival_delay < -10:
            base_satisfaction += 0.2
        
        if booking_class in ['Business', 'First']:
            base_satisfaction += 0.3
        
        satisfaction = max(1, min(5, base_satisfaction + np.random.normal(0, 0.5)))
        
        data.append({
            'Flight_ID': f'FL{i+1:06d}',
            'Passenger_Name': passenger_name,
            'Flight_Number': flight_number,
            'Airline': CARRIERS[carrier],
            'Origin': origin,
            'Destination': destination,
            'Route': route,
            'Distance_km': distance,
            'Scheduled_Departure': scheduled_departure,
            'Scheduled_Arrival': scheduled_arrival,
            'Actual_Departure': actual_departure if not is_cancelled else None,
            'Actual_Arrival': actual_arrival if not is_cancelled else None,
            'Departure_Delay_Minutes': round(departure_delay, 1) if not is_cancelled else None,
            'Arrival_Delay_Minutes': round(arrival_delay, 1) if not is_cancelled else None,
            'Flight_Status': flight_status,
            'Booking_Class': booking_class,
            'Frequent_Flyer_Status': ff_status,
            'Ticket_Price': round(ticket_price, 2),
            'Customer_Satisfaction': round(satisfaction, 1),
            'Day_of_Week': scheduled_departure.strftime('%A'),
            'Month': scheduled_departure.strftime('%B'),
            'Hour': scheduled_departure.hour
        })
        
        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1:,} flights...")
    
    df = pd.DataFrame(data)
    df = df.sort_values('Scheduled_Departure').reset_index(drop=True)
    
    # Print summary
    print(f"\n✓ Dataset generated successfully!")
    print(f"  - Total flights: {len(df):,}")
    print(f"  - Date range: {df['Scheduled_Departure'].min()} to {df['Scheduled_Departure'].max()}")
    print(f"  - Unique airlines: {df['Airline'].nunique()}")
    print(f"  - Unique routes: {df['Route'].nunique()}")
    print(f"  - Cancelled: {(df['Flight_Status'] == 'Cancelled').sum():,} ({(df['Flight_Status'] == 'Cancelled').mean()*100:.1f}%)")
    
    non_cancelled = df[df['Flight_Status'] != 'Cancelled']
    delayed = (non_cancelled['Arrival_Delay_Minutes'] >= 15).sum()
    print(f"  - Delayed (>15 min): {delayed:,} ({delayed/len(non_cancelled)*100:.1f}%)")
    print(f"  - Average delay: {non_cancelled['Arrival_Delay_Minutes'].mean():.1f} minutes")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"\n✓ Data saved to: {save_path}")
    
    return df


def generate_airport_metadata() -> pd.DataFrame:
    """Generate airport metadata."""
    airport_info = {
        'SYD': ('Sydney', 'Australia', -33.9399, 151.1753, 3),
        'MEL': ('Melbourne', 'Australia', -37.6690, 144.8410, 4),
        'BNE': ('Brisbane', 'Australia', -27.3842, 153.1175, 2),
        'PER': ('Perth', 'Australia', -31.9403, 115.9672, 2),
        'ADL': ('Adelaide', 'Australia', -34.9450, 138.5306, 1),
        'LAX': ('Los Angeles', 'USA', 33.9425, -118.4081, 4),
        'JFK': ('New York', 'USA', 40.6413, -73.7781, 4),
        'ORD': ('Chicago', 'USA', 41.9742, -87.9073, 8),
        'LHR': ('London', 'UK', 51.4700, -0.4543, 2),
        'CDG': ('Paris', 'France', 49.0097, 2.5479, 4),
        'DXB': ('Dubai', 'UAE', 25.2532, 55.3657, 2),
        'SIN': ('Singapore', 'Singapore', 1.3644, 103.9915, 3),
    }
    
    data = []
    for code, (city, country, lat, lon, runways) in airport_info.items():
        data.append({
            'Airport_Code': code,
            'City': city,
            'Country': country,
            'Latitude': lat,
            'Longitude': lon,
            'Runways': runways,
            'Is_Hub': code in ['SYD', 'MEL', 'LAX', 'JFK', 'LHR', 'DXB', 'SIN']
        })
    
    return pd.DataFrame(data)


def generate_carrier_metadata() -> pd.DataFrame:
    """Generate carrier metadata."""
    data = []
    for code, name in CARRIERS.items():
        data.append({
            'Carrier_Code': code,
            'Carrier_Name': name,
            'Delay_Factor': CARRIER_DELAY_FACTOR.get(code, 1.0)
        })
    return pd.DataFrame(data)


if __name__ == '__main__':
    # Try to load Kaggle dataset first
    kaggle_df = load_kaggle_dataset()
    
    if kaggle_df is None:
        print("\nGenerating sample dataset instead...")
        df = generate_sample_dataset(
            n_flights=50000,
            start_date='2023-01-01',
            end_date='2023-12-31',
            save_path='data/raw/flights.csv'
        )
    else:
        df = preprocess_kaggle_dataset(kaggle_df)
        df.to_csv('data/processed/flights_processed.csv', index=False)
    
    # Generate metadata
    airports_df = generate_airport_metadata()
    airports_df.to_csv('data/external/airports.csv', index=False)
    print(f"\n✓ Airport metadata saved: data/external/airports.csv")
    
    carriers_df = generate_carrier_metadata()
    carriers_df.to_csv('data/external/carriers.csv', index=False)
    print(f"✓ Carrier metadata saved: data/external/carriers.csv")
