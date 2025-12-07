# ✈️ Airline Flight Delay Prediction & Route Analysis

A comprehensive machine learning project for predicting flight delays and analyzing route disruption patterns. Built for travel aggregators and airlines to improve customer experience and optimize scheduling.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

## 🎯 Project Overview

This project addresses the challenge of predicting whether a flight will be delayed at booking time and identifying routes prone to disruptions. It combines multiple analytical approaches:

- **Classification**: Predict "Delayed" vs "On-Time" flights
- **Clustering**: Segment airports based on delay patterns
- **Frequent Itemset Mining**: Discover cascade delay patterns
- **Forecasting**: Predict seasonal delay trends

## 📁 Project Structure

```
flight-delay/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned and transformed data
│   └── external/               # Weather, airport metadata
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_classification.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_frequent_itemsets.ipynb
│   └── 06_forecasting.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py      # Data loading and cleaning utilities
│   ├── features.py             # Feature engineering functions
│   ├── models.py               # Model training and evaluation
│   ├── visualization.py        # Plotting utilities
│   └── sample_data.py          # Sample data generator
├── reports/
│   └── figures/                # Generated visualizations
├── config/
│   └── config.yaml             # Project configuration
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
cd flight-delay
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Generate sample data (if not using real data):
```bash
python -c "from src.sample_data import generate_sample_dataset; generate_sample_dataset()"
```

5. Launch Jupyter:
```bash
jupyter lab
```

## 📊 Data Sources

### Recommended Real Data Sources

| Source | Description | Link |
|--------|-------------|------|
| BTS On-Time Performance | US domestic flight data | [transtats.bts.gov](https://www.transtats.bts.gov/) |
| OpenSky Network | Global flight tracking | [opensky-network.org](https://opensky-network.org/) |
| NOAA Weather | Weather conditions | [noaa.gov](https://www.noaa.gov/) |

### Sample Data

The project includes a synthetic data generator (`src/sample_data.py`) that creates realistic flight delay data for development and testing.

## 📓 Notebooks Guide

### 1. Data Exploration (`01_data_exploration.ipynb`)
- Dataset overview and statistics
- Delay distribution analysis
- Temporal patterns visualization
- Correlation analysis

### 2. Data Cleaning (`02_data_cleaning.ipynb`)
- Missing value handling
- Time format standardization
- Outlier detection and treatment
- Feature encoding

### 3. Classification (`03_classification.ipynb`)
- Baseline models (Logistic Regression)
- Advanced models (XGBoost, LightGBM)
- Handling imbalanced data with SMOTE
- Model evaluation and comparison
- Feature importance analysis

### 4. Clustering (`04_clustering.ipynb`)
- Airport segmentation with K-Means
- Optimal cluster selection
- Cluster profiling and visualization
- Business insights extraction

### 5. Frequent Itemset Mining (`05_frequent_itemsets.ipynb`)
- Route delay pattern discovery
- Cascade delay detection
- Association rule mining
- Pattern interpretation

### 6. Forecasting (`06_forecasting.ipynb`)
- Time series decomposition
- SARIMA modeling
- Prophet forecasting
- Seasonal trend prediction

## 🔬 Key Features

### Classification Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- LightGBM
- Ensemble methods

### Handling Imbalanced Data
- SMOTE oversampling
- Class weight adjustment
- Threshold optimization
- Cost-sensitive learning

### Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix
- Business-specific metrics

## 📈 Results Summary

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.72 | 0.68 | 0.70 | 0.76 |
| Random Forest | 0.78 | 0.74 | 0.76 | 0.82 |
| XGBoost | 0.81 | 0.77 | 0.79 | 0.85 |
| LightGBM | 0.80 | 0.78 | 0.79 | 0.84 |

*Results based on sample data; actual performance may vary.*

## 🎨 Visualizations

The project generates various visualizations:
- Delay distribution plots
- Airport clustering maps
- Feature importance charts
- Time series forecasts
- Association rule networks

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Visualization settings
- Threshold values

## 📚 Learning Objectives

This project demonstrates:
- ✅ Classification on imbalanced datasets
- ✅ Unsupervised learning for segmentation
- ✅ Pattern mining for discovery
- ✅ Time series forecasting
- ✅ Feature engineering best practices
- ✅ Model interpretation with SHAP

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Bureau of Transportation Statistics for flight data
- Scikit-learn, XGBoost, and Prophet communities
- Open source contributors

---

**Built with ❤️ for aviation analytics**

# flight-delay-predictor
