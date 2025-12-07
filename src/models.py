"""
Models Module
=============

Machine learning model training, evaluation, and utilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DelayClassifier:
    """
    Wrapper class for flight delay classification models.
    """
    
    def __init__(self, model_type: str = 'logistic_regression', **kwargs):
        """
        Initialize the classifier.
        
        Parameters
        ----------
        model_type : str
            Type of model: 'logistic_regression', 'random_forest', 'xgboost', 'lightgbm'
        **kwargs : dict
            Additional parameters for the model
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def _create_model(self, model_type: str, **kwargs):
        """Create the underlying model."""
        
        if model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                class_weight=kwargs.get('class_weight', 'balanced'),
                random_state=kwargs.get('random_state', 42)
            )
            
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                class_weight=kwargs.get('class_weight', 'balanced'),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
            
        elif model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    scale_pos_weight=kwargs.get('scale_pos_weight', 3),
                    random_state=kwargs.get('random_state', 42),
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                print("⚠ XGBoost not installed, using Gradient Boosting instead")
                return GradientBoostingClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42)
                )
                
        elif model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    class_weight=kwargs.get('class_weight', 'balanced'),
                    random_state=kwargs.get('random_state', 42),
                    verbose=-1
                )
            except ImportError:
                print("⚠ LightGBM not installed, using Gradient Boosting instead")
                return GradientBoostingClassifier(
                    n_estimators=kwargs.get('n_estimators', 200),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42)
                )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, scale: bool = True):
        """
        Fit the model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y : pd.Series
            Training target
        scale : bool
            Whether to scale features
        """
        self.feature_names = X.columns.tolist()
        
        if scale:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
            
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"✓ {self.model_type} model trained on {len(X):,} samples")
        
    def predict(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        if scale:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
            
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, scale: bool = True) -> Dict:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X : pd.DataFrame
            Test features
        y : pd.Series
            Test target
        scale : bool
            Whether to scale features
            
        Returns
        -------
        Dict
            Evaluation metrics
        """
        y_pred = self.predict(X, scale)
        y_prob = self.predict_proba(X, scale)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_prob),
            'pr_auc': average_precision_score(y, y_prob),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
            
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        importance_df['importance_pct'] = (
            importance_df['importance'] / importance_df['importance'].sum() * 100
        )
        
        return importance_df.head(top_n)


def train_multiple_models(X_train: pd.DataFrame, y_train: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         models: List[str] = None) -> Dict:
    """
    Train and evaluate multiple models.
    
    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train and test features
    y_train, y_test : pd.Series
        Train and test targets
    models : List[str]
        List of model types to train
        
    Returns
    -------
    Dict
        Dictionary of trained models and their metrics
    """
    if models is None:
        models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    results = {}
    
    print("\n" + "="*60)
    print("Training Multiple Models")
    print("="*60 + "\n")
    
    for model_type in models:
        print(f"\n{'─'*40}")
        print(f"Training: {model_type}")
        print('─'*40)
        
        try:
            clf = DelayClassifier(model_type=model_type)
            clf.fit(X_train, y_train)
            metrics = clf.evaluate(X_test, y_test)
            
            results[model_type] = {
                'model': clf,
                'metrics': metrics
            }
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            
        except Exception as e:
            print(f"  ⚠ Error: {str(e)}")
            results[model_type] = {'error': str(e)}
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60 + "\n")
    
    return results


def apply_smote(X: pd.DataFrame, y: pd.Series,
                sampling_strategy: float = 0.8,
                random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE oversampling to handle class imbalance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    sampling_strategy : float
        Ratio of minority to majority class after resampling
    random_state : int
        Random seed
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Resampled X and y
    """
    try:
        from imblearn.over_sampling import SMOTE
        
        print(f"Before SMOTE: {y.value_counts().to_dict()}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        print(f"After SMOTE:  {y_resampled.value_counts().to_dict()}")
        
        return X_resampled, y_resampled
        
    except ImportError:
        print("⚠ imbalanced-learn not installed, returning original data")
        return X, y


def cross_validate_model(model: DelayClassifier, X: pd.DataFrame, y: pd.Series,
                        cv: int = 5, scoring: str = 'f1') -> Dict:
    """
    Perform cross-validation.
    
    Parameters
    ----------
    model : DelayClassifier
        Model to validate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of folds
    scoring : str
        Scoring metric
        
    Returns
    -------
    Dict
        Cross-validation results
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Scale features
    X_scaled = model.scaler.fit_transform(X)
    
    scores = cross_val_score(model.model, X_scaled, y, cv=skf, scoring=scoring)
    
    results = {
        'scores': scores.tolist(),
        'mean': scores.mean(),
        'std': scores.std(),
        'cv_folds': cv,
        'scoring': scoring
    }
    
    print(f"✓ {cv}-Fold Cross-Validation ({scoring})")
    print(f"  Mean: {results['mean']:.4f} (+/- {results['std']*2:.4f})")
    
    return results


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                      metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1', 'precision', 'recall'
        
    Returns
    -------
    Tuple[float, float]
        Optimal threshold and corresponding metric value
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    print(f"✓ Optimal threshold for {metric}: {best_threshold:.2f} (score: {best_score:.4f})")
    
    return best_threshold, best_score


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                             target_names: List[str] = None) -> str:
    """
    Generate detailed classification report.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    target_names : List[str]
        Names for classes
        
    Returns
    -------
    str
        Classification report
    """
    if target_names is None:
        target_names = ['On-Time', 'Delayed']
        
    return classification_report(y_true, y_pred, target_names=target_names)


def compare_models(results: Dict) -> pd.DataFrame:
    """
    Create comparison table for multiple models.
    
    Parameters
    ----------
    results : Dict
        Results from train_multiple_models
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = []
    
    for model_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'ROC-AUC': metrics['roc_auc'],
                'PR-AUC': metrics['pr_auc']
            })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    
    return df


def explain_predictions(model: DelayClassifier, X: pd.DataFrame,
                       n_samples: int = 100) -> Dict:
    """
    Generate SHAP explanations for model predictions.
    
    Parameters
    ----------
    model : DelayClassifier
        Trained model
    X : pd.DataFrame
        Features to explain
    n_samples : int
        Number of samples to explain
        
    Returns
    -------
    Dict
        SHAP values and explanations
    """
    try:
        import shap
        
        # Use a sample for efficiency
        X_sample = X.sample(min(n_samples, len(X)), random_state=42)
        X_scaled = model.scaler.transform(X_sample)
        
        # Create explainer based on model type
        if model.model_type in ['xgboost', 'lightgbm', 'random_forest']:
            explainer = shap.TreeExplainer(model.model)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.LinearExplainer(model.model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        
        return {
            'explainer': explainer,
            'shap_values': shap_values,
            'X_sample': X_sample,
            'feature_names': model.feature_names
        }
        
    except ImportError:
        print("⚠ SHAP not installed")
        return {}
    except Exception as e:
        print(f"⚠ Error generating SHAP values: {e}")
        return {}

