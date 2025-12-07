"""
Feature Engineering Module
===========================

Functions for creating and transforming features for flight delay prediction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from departure datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data with scheduled_departure column
        
    Returns
    -------
    pd.DataFrame
        Data with temporal features
    """
    df = df.copy()
    
    if 'scheduled_departure' not in df.columns:
        print("⚠ scheduled_departure column not found")
        return df
    
    # Time of day categories
    hour = df['scheduled_departure'].dt.hour
    conditions = [
        (hour >= 5) & (hour < 12),   # Morning
        (hour >= 12) & (hour < 17),  # Afternoon
        (hour >= 17) & (hour < 21),  # Evening
        (hour >= 21) | (hour < 5)    # Night
    ]
    categories = ['Morning', 'Afternoon', 'Evening', 'Night']
    df['time_of_day'] = np.select(conditions, categories, default='Unknown')
    
    # Peak hours (6-9 AM and 5-8 PM)
    df['is_peak_hour'] = ((hour >= 6) & (hour <= 9) | (hour >= 17) & (hour <= 20)).astype(int)
    
    # Day of month categories
    day = df['scheduled_departure'].dt.day
    df['is_month_start'] = (day <= 7).astype(int)
    df['is_month_end'] = (day >= 24).astype(int)
    
    # Season
    month = df['scheduled_departure'].dt.month
    season_conditions = [
        month.isin([12, 1, 2]),   # Winter
        month.isin([3, 4, 5]),    # Spring
        month.isin([6, 7, 8]),    # Summer
        month.isin([9, 10, 11])   # Fall
    ]
    seasons = ['Winter', 'Spring', 'Summer', 'Fall']
    df['season'] = np.select(season_conditions, seasons, default='Unknown')
    
    print("✓ Created temporal features")
    return df


def create_holiday_features(df: pd.DataFrame,
                           holidays: List[str] = None) -> pd.DataFrame:
    """
    Create holiday-related features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    holidays : List[str]
        List of holiday dates in 'YYYY-MM-DD' format
        
    Returns
    -------
    pd.DataFrame
        Data with holiday features
    """
    df = df.copy()
    
    # Default US holidays for 2023
    if holidays is None:
        holidays = [
            '2023-01-01',  # New Year's Day
            '2023-01-16',  # MLK Day
            '2023-02-20',  # Presidents Day
            '2023-05-29',  # Memorial Day
            '2023-07-04',  # Independence Day
            '2023-09-04',  # Labor Day
            '2023-10-09',  # Columbus Day
            '2023-11-11',  # Veterans Day
            '2023-11-23',  # Thanksgiving
            '2023-12-25',  # Christmas
        ]
    
    holiday_dates = pd.to_datetime(holidays)
    
    if 'scheduled_departure' in df.columns:
        df['date'] = df['scheduled_departure'].dt.date
        df['is_holiday'] = df['date'].isin(holiday_dates.date).astype(int)
        
        # Days near holiday (travel surge)
        df['days_to_holiday'] = df['date'].apply(
            lambda x: min((abs((pd.Timestamp(x) - h).days) for h in holiday_dates), default=365)
        )
        df['is_holiday_period'] = (df['days_to_holiday'] <= 3).astype(int)
        
        df = df.drop('date', axis=1)
    
    print("✓ Created holiday features")
    return df


def create_route_features(df: pd.DataFrame,
                         airport_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Create route-based features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    airport_data : pd.DataFrame
        Airport metadata (optional)
        
    Returns
    -------
    pd.DataFrame
        Data with route features
    """
    df = df.copy()
    
    # Create route identifier
    if 'origin' in df.columns and 'destination' in df.columns:
        df['route'] = df['origin'] + '-' + df['destination']
        
        # Route frequency (popularity)
        route_counts = df['route'].value_counts()
        df['route_frequency'] = df['route'].map(route_counts)
        
        # Is same region (simplified - same first letter for demonstration)
        df['is_same_region'] = (df['origin'].str[0] == df['destination'].str[0]).astype(int)
    
    # Distance category
    if 'distance' in df.columns:
        conditions = [
            df['distance'] < 500,
            df['distance'] < 1000,
            df['distance'] < 2000,
            df['distance'] >= 2000
        ]
        categories = ['Short', 'Medium', 'Long', 'Ultra-Long']
        df['distance_category'] = np.select(conditions, categories, default='Unknown')
    
    print("✓ Created route features")
    return df


def create_carrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create carrier-based features using historical performance.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    pd.DataFrame
        Data with carrier features
    """
    df = df.copy()
    
    if 'carrier' not in df.columns:
        print("⚠ carrier column not found")
        return df
    
    # Carrier statistics
    if 'arrival_delay' in df.columns:
        carrier_stats = df.groupby('carrier').agg({
            'arrival_delay': ['mean', 'std', 'median'],
            'flight_id': 'count'
        }).reset_index()
        
        carrier_stats.columns = ['carrier', 'carrier_mean_delay', 'carrier_std_delay',
                                  'carrier_median_delay', 'carrier_flight_count']
        
        df = df.merge(carrier_stats, on='carrier', how='left')
    
    # Carrier size category
    if 'carrier_flight_count' in df.columns:
        df['carrier_size'] = pd.qcut(df['carrier_flight_count'], q=3, 
                                      labels=['Small', 'Medium', 'Large'], duplicates='drop')
    
    print("✓ Created carrier features")
    return df


def create_airport_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create airport-based features using historical data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    pd.DataFrame
        Data with airport features
    """
    df = df.copy()
    
    # Origin airport statistics
    if 'origin' in df.columns and 'departure_delay' in df.columns:
        origin_stats = df.groupby('origin').agg({
            'departure_delay': ['mean', 'std'],
            'flight_id': 'count'
        }).reset_index()
        
        origin_stats.columns = ['origin', 'origin_mean_delay', 'origin_std_delay', 
                                'origin_flight_count']
        
        df = df.merge(origin_stats, on='origin', how='left')
    
    # Destination airport statistics  
    if 'destination' in df.columns and 'arrival_delay' in df.columns:
        dest_stats = df.groupby('destination').agg({
            'arrival_delay': ['mean', 'std'],
            'flight_id': 'count'
        }).reset_index()
        
        dest_stats.columns = ['destination', 'dest_mean_delay', 'dest_std_delay',
                              'dest_flight_count']
        
        df = df.merge(dest_stats, on='destination', how='left')
    
    # Hub indicator (high traffic airports)
    if 'origin_flight_count' in df.columns:
        threshold = df['origin_flight_count'].quantile(0.9)
        df['origin_is_hub'] = (df['origin_flight_count'] >= threshold).astype(int)
    
    if 'dest_flight_count' in df.columns:
        threshold = df['dest_flight_count'].quantile(0.9)
        df['dest_is_hub'] = (df['dest_flight_count'] >= threshold).astype(int)
    
    print("✓ Created airport features")
    return df


def create_congestion_features(df: pd.DataFrame,
                               time_window: int = 60) -> pd.DataFrame:
    """
    Create airport congestion features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    time_window : int
        Time window in minutes for congestion calculation
        
    Returns
    -------
    pd.DataFrame
        Data with congestion features
    """
    df = df.copy()
    
    if 'scheduled_departure' not in df.columns or 'origin' not in df.columns:
        print("⚠ Required columns not found for congestion features")
        return df
    
    # Sort by departure time
    df = df.sort_values('scheduled_departure').reset_index(drop=True)
    
    # Calculate hourly departures at each airport
    df['departure_date_hour'] = df['scheduled_departure'].dt.floor('H')
    
    hourly_departures = df.groupby(['origin', 'departure_date_hour']).size().reset_index(name='hourly_departures')
    df = df.merge(hourly_departures, on=['origin', 'departure_date_hour'], how='left')
    
    # Congestion level
    df['congestion_level'] = pd.cut(df['hourly_departures'], 
                                    bins=[0, 5, 10, 20, 50, float('inf')],
                                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    df = df.drop('departure_date_hour', axis=1)
    
    print("✓ Created congestion features")
    return df


def create_weather_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weather-related proxy features (when actual weather data unavailable).
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    pd.DataFrame
        Data with weather proxy features
    """
    df = df.copy()
    
    # Seasonal weather risk (simplified model)
    if 'departure_month' in df.columns:
        # Higher risk in winter months
        winter_months = [12, 1, 2]
        storm_months = [6, 7, 8]  # Summer thunderstorms
        
        df['winter_weather_risk'] = df['departure_month'].isin(winter_months).astype(int)
        df['summer_storm_risk'] = df['departure_month'].isin(storm_months).astype(int)
    
    # Time-based weather risk (afternoon thunderstorms more common)
    if 'departure_hour' in df.columns:
        df['afternoon_storm_risk'] = ((df['departure_hour'] >= 14) & 
                                       (df['departure_hour'] <= 18) &
                                       df.get('summer_storm_risk', 0) == 1).astype(int)
    
    print("✓ Created weather proxy features")
    return df


def create_historical_delay_features(df: pd.DataFrame,
                                     lookback_days: int = 30) -> pd.DataFrame:
    """
    Create features based on historical delay patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    lookback_days : int
        Number of days to look back for historical patterns
        
    Returns
    -------
    pd.DataFrame
        Data with historical features
    """
    df = df.copy()
    
    if 'route' not in df.columns:
        if 'origin' in df.columns and 'destination' in df.columns:
            df['route'] = df['origin'] + '-' + df['destination']
    
    # Historical route delay rate
    if 'route' in df.columns and 'is_delayed' in df.columns:
        route_delay_rate = df.groupby('route')['is_delayed'].mean().reset_index()
        route_delay_rate.columns = ['route', 'route_delay_rate']
        df = df.merge(route_delay_rate, on='route', how='left')
    
    # Historical carrier-route performance
    if all(col in df.columns for col in ['carrier', 'route', 'arrival_delay']):
        carrier_route_stats = df.groupby(['carrier', 'route'])['arrival_delay'].agg(['mean', 'count']).reset_index()
        carrier_route_stats.columns = ['carrier', 'route', 'carrier_route_avg_delay', 'carrier_route_flights']
        
        df = df.merge(carrier_route_stats, on=['carrier', 'route'], how='left')
    
    # Fill NaN with overall averages
    for col in ['route_delay_rate', 'carrier_route_avg_delay']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    print("✓ Created historical delay features")
    return df


def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw flight data
        
    Returns
    -------
    pd.DataFrame
        Data with all engineered features
    """
    print("\n" + "="*50)
    print("Starting Feature Engineering Pipeline")
    print("="*50 + "\n")
    
    df = create_temporal_features(df)
    df = create_holiday_features(df)
    df = create_route_features(df)
    df = create_carrier_features(df)
    df = create_airport_features(df)
    df = create_congestion_features(df)
    df = create_weather_proxy_features(df)
    df = create_historical_delay_features(df)
    
    print("\n" + "="*50)
    print(f"Feature Engineering Complete")
    print(f"Total features: {len(df.columns)}")
    print("="*50 + "\n")
    
    return df


def get_feature_importance_df(model, feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Parameters
    ----------
    model : trained model
        Model with feature_importances_ attribute
    feature_names : List[str]
        Names of features
    top_n : int
        Number of top features to return
        
    Returns
    -------
    pd.DataFrame
        Feature importance dataframe
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model doesn't have feature importance attribute")
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Normalize to percentages
    importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
    
    return importance_df.head(top_n)


def select_features(X: pd.DataFrame, y: pd.Series,
                   method: str = 'mutual_info',
                   n_features: int = 20) -> List[str]:
    """
    Select top features based on importance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    method : str
        Selection method: 'mutual_info', 'chi2', 'f_classif'
    n_features : int
        Number of features to select
        
    Returns
    -------
    List[str]
        Selected feature names
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, f_classif
    
    methods = {
        'mutual_info': mutual_info_classif,
        'chi2': chi2,
        'f_classif': f_classif
    }
    
    # Handle non-negative constraint for chi2
    if method == 'chi2':
        X_temp = X - X.min()
    else:
        X_temp = X
    
    selector = SelectKBest(methods[method], k=min(n_features, len(X.columns)))
    selector.fit(X_temp, y)
    
    selected_mask = selector.get_support()
    selected_features = X.columns[selected_mask].tolist()
    
    print(f"✓ Selected {len(selected_features)} features using {method}")
    
    return selected_features

