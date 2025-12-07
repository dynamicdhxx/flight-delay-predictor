"""
Data Processing Module
======================

Utilities for loading, cleaning, and transforming flight delay data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def load_flight_data(filepath: str) -> pd.DataFrame:
    """
    Load flight data from CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file
        
    Returns
    -------
    pd.DataFrame
        Loaded flight data
    """
    df = pd.read_csv(filepath, parse_dates=['scheduled_departure', 'scheduled_arrival',
                                            'actual_departure', 'actual_arrival'])
    print(f"✓ Loaded {len(df):,} flight records")
    return df


def clean_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time columns to proper datetime format and extract components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data with time columns
        
    Returns
    -------
    pd.DataFrame
        Data with cleaned time columns
    """
    df = df.copy()
    
    # Ensure datetime format
    time_cols = ['scheduled_departure', 'scheduled_arrival', 
                 'actual_departure', 'actual_arrival']
    
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract time components
    if 'scheduled_departure' in df.columns:
        df['departure_hour'] = df['scheduled_departure'].dt.hour
        df['departure_day'] = df['scheduled_departure'].dt.day
        df['departure_month'] = df['scheduled_departure'].dt.month
        df['departure_year'] = df['scheduled_departure'].dt.year
        df['day_of_week'] = df['scheduled_departure'].dt.dayofweek
        df['day_name'] = df['scheduled_departure'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['quarter'] = df['scheduled_departure'].dt.quarter
        
    print("✓ Time columns cleaned and components extracted")
    return df


def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'smart') -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    strategy : str
        Strategy for handling missing values: 'smart', 'drop', 'fill'
        
    Returns
    -------
    pd.DataFrame
        Data with handled missing values
    """
    df = df.copy()
    initial_rows = len(df)
    
    if strategy == 'smart':
        # For cancelled flights, set actual times to NaT and mark as cancelled
        if 'is_cancelled' in df.columns:
            cancelled_mask = df['is_cancelled'] == 1
            df.loc[cancelled_mask, 'actual_departure'] = pd.NaT
            df.loc[cancelled_mask, 'actual_arrival'] = pd.NaT
        
        # Fill missing delay values with 0 for non-cancelled flights
        delay_cols = ['departure_delay', 'arrival_delay', 'carrier_delay',
                      'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay']
        
        for col in delay_cols:
            if col in df.columns:
                # Only fill where flight wasn't cancelled
                if 'is_cancelled' in df.columns:
                    mask = (df[col].isna()) & (df['is_cancelled'] == 0)
                    df.loc[mask, col] = 0
                else:
                    df[col] = df[col].fillna(0)
        
        # Fill missing categorical with mode
        cat_cols = ['carrier', 'origin', 'destination']
        for col in cat_cols:
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])
                
    elif strategy == 'drop':
        df = df.dropna()
        
    elif strategy == 'fill':
        # Fill numeric with median, categorical with mode
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    removed = initial_rows - len(df)
    if removed > 0:
        print(f"✓ Removed {removed:,} rows with missing values")
    else:
        print("✓ Missing values handled")
    
    return df


def remove_cancelled_flights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelled flights from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    pd.DataFrame
        Data without cancelled flights
    """
    df = df.copy()
    initial_rows = len(df)
    
    if 'is_cancelled' in df.columns:
        df = df[df['is_cancelled'] == 0]
        
    removed = initial_rows - len(df)
    print(f"✓ Removed {removed:,} cancelled flights ({removed/initial_rows*100:.1f}%)")
    
    return df


def encode_categorical(df: pd.DataFrame, 
                       columns: List[str] = None,
                       method: str = 'label') -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    columns : List[str]
        Columns to encode. If None, auto-detect categorical columns.
    method : str
        Encoding method: 'label', 'onehot', 'target'
        
    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Encoded data and mapping dictionary
    """
    df = df.copy()
    mappings = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Exclude date-like string columns
        columns = [c for c in columns if c not in ['day_name']]
    
    if method == 'label':
        from sklearn.preprocessing import LabelEncoder
        
        for col in columns:
            if col in df.columns:
                le = LabelEncoder()
                # Handle missing values
                df[col] = df[col].fillna('Unknown')
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
                
    elif method == 'onehot':
        for col in columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                mappings[col] = list(dummies.columns)
                
    print(f"✓ Encoded {len(columns)} categorical columns using {method} encoding")
    return df, mappings


def create_delay_target(df: pd.DataFrame, 
                        delay_col: str = 'arrival_delay',
                        threshold: int = 15) -> pd.DataFrame:
    """
    Create binary delay target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    delay_col : str
        Column containing delay in minutes
    threshold : int
        Delay threshold in minutes to classify as delayed
        
    Returns
    -------
    pd.DataFrame
        Data with delay target column
    """
    df = df.copy()
    
    df['is_delayed'] = (df[delay_col] >= threshold).astype(int)
    
    delayed_pct = df['is_delayed'].mean() * 100
    print(f"✓ Created delay target (threshold={threshold} min)")
    print(f"  - Delayed flights: {df['is_delayed'].sum():,} ({delayed_pct:.1f}%)")
    print(f"  - On-time flights: {(~df['is_delayed'].astype(bool)).sum():,} ({100-delayed_pct:.1f}%)")
    
    return df


def create_delay_categories(df: pd.DataFrame,
                            delay_col: str = 'arrival_delay') -> pd.DataFrame:
    """
    Create categorical delay severity levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    delay_col : str
        Column containing delay in minutes
        
    Returns
    -------
    pd.DataFrame
        Data with delay category column
    """
    df = df.copy()
    
    conditions = [
        df[delay_col] < 0,          # Early
        df[delay_col] < 15,         # On-time
        df[delay_col] < 30,         # Minor delay
        df[delay_col] < 60,         # Moderate delay
        df[delay_col] >= 60         # Severe delay
    ]
    
    categories = ['Early', 'On-Time', 'Minor Delay', 'Moderate Delay', 'Severe Delay']
    
    df['delay_category'] = np.select(conditions, categories, default='Unknown')
    
    print("✓ Created delay categories")
    for cat in categories:
        count = (df['delay_category'] == cat).sum()
        pct = count / len(df) * 100
        print(f"  - {cat}: {count:,} ({pct:.1f}%)")
    
    return df


def calculate_delay_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate delay-related metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    pd.DataFrame
        Data with delay metrics
    """
    df = df.copy()
    
    # Calculate departure delay if not present
    if 'departure_delay' not in df.columns:
        if 'actual_departure' in df.columns and 'scheduled_departure' in df.columns:
            df['departure_delay'] = (df['actual_departure'] - df['scheduled_departure']).dt.total_seconds() / 60
    
    # Calculate arrival delay if not present
    if 'arrival_delay' not in df.columns:
        if 'actual_arrival' in df.columns and 'scheduled_arrival' in df.columns:
            df['arrival_delay'] = (df['actual_arrival'] - df['scheduled_arrival']).dt.total_seconds() / 60
    
    # Calculate block time (actual flight duration)
    if 'actual_departure' in df.columns and 'actual_arrival' in df.columns:
        df['actual_flight_time'] = (df['actual_arrival'] - df['actual_departure']).dt.total_seconds() / 60
    
    if 'scheduled_departure' in df.columns and 'scheduled_arrival' in df.columns:
        df['scheduled_flight_time'] = (df['scheduled_arrival'] - df['scheduled_departure']).dt.total_seconds() / 60
    
    print("✓ Calculated delay metrics")
    return df


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
        
    Returns
    -------
    Dict
        Summary statistics
    """
    summary = {
        'total_flights': len(df),
        'date_range': {
            'start': df['scheduled_departure'].min() if 'scheduled_departure' in df.columns else None,
            'end': df['scheduled_departure'].max() if 'scheduled_departure' in df.columns else None
        },
        'unique_carriers': df['carrier'].nunique() if 'carrier' in df.columns else 0,
        'unique_origins': df['origin'].nunique() if 'origin' in df.columns else 0,
        'unique_destinations': df['destination'].nunique() if 'destination' in df.columns else 0,
        'missing_values': df.isnull().sum().to_dict(),
        'delay_stats': {}
    }
    
    if 'arrival_delay' in df.columns:
        summary['delay_stats'] = {
            'mean_delay': df['arrival_delay'].mean(),
            'median_delay': df['arrival_delay'].median(),
            'max_delay': df['arrival_delay'].max(),
            'min_delay': df['arrival_delay'].min(),
            'std_delay': df['arrival_delay'].std(),
            'delayed_pct': (df['arrival_delay'] >= 15).mean() * 100
        }
    
    return summary


def prepare_for_modeling(df: pd.DataFrame,
                         target_col: str = 'is_delayed',
                         exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning modeling.
    
    Parameters
    ----------
    df : pd.DataFrame
        Processed flight data
    target_col : str
        Name of target column
    exclude_cols : List[str]
        Columns to exclude from features
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix X and target vector y
    """
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = []
    
    # Default columns to exclude
    default_exclude = [
        target_col, 'flight_id', 'flight_number', 'tail_number',
        'scheduled_departure', 'scheduled_arrival', 'actual_departure', 'actual_arrival',
        'arrival_delay', 'departure_delay', 'delay_category', 'day_name',
        'carrier', 'origin', 'destination',  # Keep encoded versions
        'carrier_delay', 'weather_delay', 'nas_delay', 'security_delay', 'late_aircraft_delay',
        'is_cancelled', 'cancellation_code'
    ]
    
    exclude_cols = list(set(exclude_cols + default_exclude))
    
    # Select features
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Keep only numeric columns
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col]
    
    print(f"✓ Prepared data for modeling")
    print(f"  - Features: {len(X.columns)}")
    print(f"  - Samples: {len(X):,}")
    print(f"  - Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple:
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
    stratify : bool
        Whether to stratify split by target
        
    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    stratify_by = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_by
    )
    
    print(f"✓ Data split complete")
    print(f"  - Training: {len(X_train):,} samples")
    print(f"  - Testing: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test

