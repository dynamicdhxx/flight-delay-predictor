"""
Visualization Module
====================

Plotting utilities for flight delay analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#18A558',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'on_time': '#18A558',
    'delayed': '#C73E1D',
    'neutral': '#6C757D'
}

PALETTE = ['#2E86AB', '#A23B72', '#18A558', '#F18F01', '#C73E1D', '#6C757D', '#4ECDC4', '#45B7D1']


def set_plot_style():
    """Set consistent plot styling."""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False


def plot_delay_distribution(df: pd.DataFrame, delay_col: str = 'arrival_delay',
                           figsize: Tuple = (14, 5)) -> plt.Figure:
    """
    Plot delay distribution with histogram and box plot.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    delay_col : str
        Column containing delay values
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    ax1 = axes[0]
    data = df[delay_col].clip(-60, 180)  # Clip for better visualization
    
    colors = [COLORS['success'] if x < 15 else COLORS['danger'] for x in data]
    ax1.hist(data, bins=50, color=COLORS['primary'], edgecolor='white', alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label='On-Time')
    ax1.axvline(x=15, color=COLORS['danger'], linestyle='--', linewidth=1.5, label='Delay Threshold (15 min)')
    
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Flight Delays')
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    box_data = df[delay_col].clip(-60, 120)
    bp = ax2.boxplot(box_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor(COLORS['primary'])
    bp['boxes'][0].set_alpha(0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=15, color=COLORS['danger'], linestyle='--', linewidth=1.5)
    
    ax2.set_ylabel('Delay (minutes)')
    ax2.set_title('Delay Distribution Box Plot')
    
    plt.tight_layout()
    return fig


def plot_delay_by_category(df: pd.DataFrame, category_col: str,
                           delay_col: str = 'arrival_delay',
                           top_n: int = 10,
                           figsize: Tuple = (12, 6)) -> plt.Figure:
    """
    Plot average delay by category (carrier, airport, etc.).
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data
    category_col : str
        Categorical column to group by
    delay_col : str
        Delay column
    top_n : int
        Number of top categories to show
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate average delay by category
    avg_delay = df.groupby(category_col)[delay_col].agg(['mean', 'count']).reset_index()
    avg_delay.columns = [category_col, 'avg_delay', 'count']
    avg_delay = avg_delay.nlargest(top_n, 'avg_delay')
    
    # Color based on delay
    colors = [COLORS['danger'] if x > 15 else COLORS['warning'] if x > 0 else COLORS['success'] 
              for x in avg_delay['avg_delay']]
    
    bars = ax.barh(avg_delay[category_col], avg_delay['avg_delay'], color=colors, edgecolor='white')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=15, color=COLORS['danger'], linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('Average Delay (minutes)')
    ax.set_title(f'Average Delay by {category_col.replace("_", " ").title()}')
    
    # Add value labels
    for bar, val in zip(bars, avg_delay['avg_delay']):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}', va='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_temporal_patterns(df: pd.DataFrame, 
                          delay_col: str = 'arrival_delay',
                          figsize: Tuple = (14, 10)) -> plt.Figure:
    """
    Plot temporal patterns in delays.
    
    Parameters
    ----------
    df : pd.DataFrame
        Flight data with temporal columns
    delay_col : str
        Delay column
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # By hour
    ax1 = axes[0, 0]
    if 'departure_hour' in df.columns:
        hourly = df.groupby('departure_hour')[delay_col].mean()
        ax1.plot(hourly.index, hourly.values, marker='o', linewidth=2, 
                 color=COLORS['primary'], markersize=6)
        ax1.fill_between(hourly.index, hourly.values, alpha=0.3, color=COLORS['primary'])
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Delay (min)')
        ax1.set_title('Delays by Hour of Day')
        ax1.set_xticks(range(0, 24, 2))
    
    # By day of week
    ax2 = axes[0, 1]
    if 'day_of_week' in df.columns:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily = df.groupby('day_of_week')[delay_col].mean()
        colors = [COLORS['danger'] if x > daily.mean() else COLORS['success'] for x in daily.values]
        bars = ax2.bar(range(7), daily.values, color=colors, edgecolor='white')
        ax2.set_xticks(range(7))
        ax2.set_xticklabels(day_names)
        ax2.axhline(y=daily.mean(), color='black', linestyle='--', alpha=0.5, label='Average')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Average Delay (min)')
        ax2.set_title('Delays by Day of Week')
        ax2.legend()
    
    # By month
    ax3 = axes[1, 0]
    if 'departure_month' in df.columns:
        monthly = df.groupby('departure_month')[delay_col].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax3.bar(range(1, 13), monthly.values, color=PALETTE[:12], edgecolor='white')
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(month_names, rotation=45)
        ax3.axhline(y=monthly.mean(), color='black', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Average Delay (min)')
        ax3.set_title('Delays by Month')
    
    # Delay rate by time of day
    ax4 = axes[1, 1]
    if 'time_of_day' in df.columns and 'is_delayed' in df.columns:
        time_order = ['Morning', 'Afternoon', 'Evening', 'Night']
        delay_rate = df.groupby('time_of_day')['is_delayed'].mean() * 100
        delay_rate = delay_rate.reindex(time_order)
        colors = [COLORS['success'], COLORS['warning'], COLORS['danger'], COLORS['primary']]
        ax4.bar(time_order, delay_rate.values, color=colors, edgecolor='white')
        ax4.set_xlabel('Time of Day')
        ax4.set_ylabel('Delay Rate (%)')
        ax4.set_title('Delay Rate by Time of Day')
        
        for i, v in enumerate(delay_rate.values):
            ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels: List[str] = None,
                         figsize: Tuple = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    labels : List[str]
        Class labels
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = ['On-Time', 'Delayed']
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize for percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count'})
    
    # Add percentage annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j + 0.5, i + 0.7, f'({cm_pct[i, j]:.1f}%)',
                   ha='center', va='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray,
                      figsize: Tuple = (14, 5)) -> plt.Figure:
    """
    Plot ROC and Precision-Recall curves.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ROC Curve
    ax1 = axes[0]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = np.trapz(tpr, fpr)
    
    ax1.plot(fpr, tpr, color=COLORS['primary'], linewidth=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.2, color=COLORS['primary'])
    
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Precision-Recall Curve
    ax2 = axes[1]
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = np.trapz(precision, recall)
    
    ax2.plot(recall, precision, color=COLORS['secondary'], linewidth=2,
             label=f'PR Curve (AUC = {pr_auc:.3f})')
    baseline = y_true.mean()
    ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1, alpha=0.5, 
                label=f'Baseline ({baseline:.2f})')
    ax2.fill_between(recall, precision, alpha=0.2, color=COLORS['secondary'])
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='lower left')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame,
                           top_n: int = 15,
                           figsize: Tuple = (10, 8)) -> plt.Figure:
    """
    Plot feature importance.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to show
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    data = importance_df.head(top_n).sort_values('importance')
    
    colors = plt.cm.Blues(np.linspace(0.3, 1, len(data)))
    
    bars = ax.barh(data['feature'], data['importance'], color=colors, edgecolor='white')
    
    ax.set_xlabel('Importance')
    ax.set_title(f'Top {top_n} Feature Importance')
    
    # Add value labels
    max_val = data['importance'].max()
    for bar, val in zip(bars, data['importance']):
        ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame,
                         figsize: Tuple = (12, 6)) -> plt.Figure:
    """
    Plot model comparison chart.
    
    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with model metrics
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, comparison_df[metric], width, 
                      label=metric, color=PALETTE[i], edgecolor='white')
    
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_cluster_analysis(df: pd.DataFrame, cluster_col: str,
                         features: List[str],
                         figsize: Tuple = (14, 10)) -> plt.Figure:
    """
    Plot cluster analysis results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Data with cluster assignments
    cluster_col : str
        Column containing cluster labels
    features : List[str]
        Features used for clustering
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    
    n_features = min(len(features), 4)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    n_clusters = df[cluster_col].nunique()
    colors = PALETTE[:n_clusters]
    
    for i, feature in enumerate(features[:n_features]):
        ax = axes[i]
        for j, cluster in enumerate(sorted(df[cluster_col].unique())):
            cluster_data = df[df[cluster_col] == cluster][feature]
            ax.hist(cluster_data, bins=30, alpha=0.6, label=f'Cluster {cluster}',
                   color=colors[j], edgecolor='white')
        
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Frequency')
        ax.set_title(f'{feature.replace("_", " ").title()} by Cluster')
        ax.legend()
    
    # Hide unused subplots
    for i in range(n_features, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_time_series_forecast(actual: pd.Series, predicted: pd.Series,
                             dates: pd.DatetimeIndex = None,
                             confidence_interval: Tuple = None,
                             figsize: Tuple = (14, 6)) -> plt.Figure:
    """
    Plot time series forecast vs actual.
    
    Parameters
    ----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
    dates : pd.DatetimeIndex
        Date index
    confidence_interval : Tuple
        Lower and upper bounds
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    x = dates if dates is not None else range(len(actual))
    
    ax.plot(x, actual, label='Actual', color=COLORS['primary'], linewidth=2)
    ax.plot(x, predicted, label='Predicted', color=COLORS['secondary'], 
            linewidth=2, linestyle='--')
    
    if confidence_interval is not None:
        ax.fill_between(x, confidence_interval[0], confidence_interval[1],
                       alpha=0.2, color=COLORS['secondary'], label='95% CI')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Forecast')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_association_rules(rules_df: pd.DataFrame, top_n: int = 15,
                          figsize: Tuple = (12, 8)) -> plt.Figure:
    """
    Plot association rules visualization.
    
    Parameters
    ----------
    rules_df : pd.DataFrame
        DataFrame with association rules
    top_n : int
        Number of top rules to show
    figsize : Tuple
        Figure size
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Top rules by lift
    ax1 = axes[0]
    top_rules = rules_df.nlargest(top_n, 'lift')
    top_rules['rule'] = top_rules['antecedents'].astype(str) + ' → ' + top_rules['consequents'].astype(str)
    top_rules['rule'] = top_rules['rule'].str.replace("frozenset\\(\\{|\\}\\)", "", regex=True)
    top_rules['rule'] = top_rules['rule'].str.replace("'", "")
    
    colors = plt.cm.Oranges(np.linspace(0.4, 1, len(top_rules)))
    ax1.barh(top_rules['rule'], top_rules['lift'], color=colors)
    ax1.set_xlabel('Lift')
    ax1.set_title(f'Top {top_n} Rules by Lift')
    
    # Scatter plot: Support vs Confidence
    ax2 = axes[1]
    scatter = ax2.scatter(rules_df['support'], rules_df['confidence'], 
                         c=rules_df['lift'], cmap='viridis', 
                         alpha=0.6, s=50, edgecolors='white')
    plt.colorbar(scatter, ax=ax2, label='Lift')
    ax2.set_xlabel('Support')
    ax2.set_ylabel('Confidence')
    ax2.set_title('Support vs Confidence (colored by Lift)')
    
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filename: str, 
               directory: str = 'reports/figures',
               dpi: int = 150):
    """
    Save figure to file.
    
    Parameters
    ----------
    fig : plt.Figure
        Figure to save
    filename : str
        Filename (without extension)
    directory : str
        Directory to save to
    dpi : int
        Resolution
    """
    import os
    os.makedirs(directory, exist_ok=True)
    
    filepath = os.path.join(directory, f'{filename}.png')
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"✓ Figure saved: {filepath}")

