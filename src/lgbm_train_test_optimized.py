# src/lgbm_train_test_optimized.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Tuple, Union, List
from dataclasses import dataclass
import logging
from functools import lru_cache
import json
from pathlib import Path
import pickle

from src.config_optimized import GANANCIA, ESTIMULO

logger = logging.getLogger(__name__)

@dataclass
class ModelParameters:
    """LightGBM model parameters"""
    objective: str = 'binary'
    boosting_type: str = 'gbdt'
    first_metric_only: bool = True
    boost_from_average: bool = True
    feature_pre_filter: bool = False
    max_bin: int = 31
    num_leaves: int = None
    learning_rate: float = None
    min_data_in_leaf: int = None
    feature_fraction: float = None
    bagging_fraction: float = None
    seed: int = None
    verbose: int = 0

class ChurnPredictor:
    """Main class for churn prediction using LightGBM"""
    
    def __init__(self, batch_size: int = 10000):
        self.batch_size = batch_size
        self._feature_importance_cache = {}
    
    def train_model(self, 
                    X_train: pd.DataFrame,
                    y_train: pd.Series,
                    w_train: pd.Series,
                    best_iter: int,
                    best_params: dict,
                    name: str,
                    output_path: str,
                    seed: int) -> lgb.Booster:
        """Train LightGBM model with optimized parameters"""
        
        logger.info(f"Training LightGBM model: {name} for months: {X_train['foto_mes'].unique()}")
        
        params = ModelParameters(
            num_leaves=best_params['num_leaves'],
            learning_rate=best_params['learning_rate'],
            min_data_in_leaf=best_params['min_data_in_leaf'],
            feature_fraction=best_params['feature_fraction'],
            bagging_fraction=best_params['bagging_fraction'],
            seed=seed
        )
        
        # Process data in batches for large datasets
        train_dataset = self._create_dataset_in_batches(
            X_train, y_train, w_train
        )
        
        model = lgb.train(params.__dict__,
                         train_dataset,
                         num_boost_round=best_iter)
        
        self._save_model(model, name, output_path)
        return model
    
    @staticmethod
    def _create_dataset_in_batches(X: pd.DataFrame, 
                                 y: pd.Series, 
                                 w: pd.Series,
                                 batch_size: int = 10000) -> lgb.Dataset:
        """Create LightGBM dataset in batches to handle large datasets"""
        datasets = []
        for i in range(0, len(X), batch_size):
            batch_end = min(i + batch_size, len(X))
            batch_dataset = lgb.Dataset(
                X.iloc[i:batch_end],
                label=y.iloc[i:batch_end],
                weight=w.iloc[i:batch_end]
            )
            datasets.append(batch_dataset)
        
        # Merge all batches
        final_dataset = datasets[0]
        for dataset in datasets[1:]:
            final_dataset.concat(dataset)
        
        return final_dataset
    
    @staticmethod
    def _save_model(model: lgb.Booster, name: str, output_path: str):
        """Save model with error handling"""
        try:
            model_path = Path(output_path) / f'{name}_model_lgbm.txt'
            model.save_model(str(model_path))
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model {name}: {e}")
            raise
    
    @lru_cache(maxsize=32)
    def calculate_feature_importance(self, 
                                  model: lgb.Booster,
                                  X_train: pd.DataFrame,
                                  name: str,
                                  output_path: str):
        """Calculate and cache feature importance"""
        logger.info("Calculating feature importance")
        
        try:
            # Generate and save plot
            plt.figure(figsize=(10, 20))
            lgb.plot_importance(model)
            plt.savefig(Path(output_path) / f"{name}_feature_importance_grafico.png",
                       bbox_inches='tight')
            plt.close()
            
            # Calculate importance metrics
            importances = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importance()
            })
            importances['importance_%'] = (
                importances['importance'] / importances['importance'].sum() * 100
            )
            importances = importances.sort_values('importance', ascending=False)
            
            # Save to Excel
            importances.to_excel(
                Path(output_path) / f"{name}_feature_importance_data_frame.xlsx",
                index=False
            )
            
            # Cache results
            self._feature_importance_cache[name] = importances
            
        except Exception as e:
            logger.error(f"Error in feature importance calculation: {e}")
            raise
            
    def predict(self, X: pd.DataFrame, model: lgb.Booster) -> np.ndarray:
        """Make predictions in batches"""
        logger.info(f"Predicting for month: {X['foto_mes'].unique()}")
        
        predictions = []
        for i in range(0, len(X), self.batch_size):
            batch = X.iloc[i:i + self.batch_size]
            batch_pred = model.predict(batch)
            predictions.append(batch_pred)
            
        return np.concatenate(predictions)

class ProfitOptimizer:
    """Optimize profit thresholds and evaluate model performance"""
    
    def __init__(self, profit_correct: float = GANANCIA, 
                 cost_stimulus: float = ESTIMULO):
        self.profit_correct = profit_correct
        self.cost_stimulus = cost_stimulus
        
    def calculate_optimal_threshold(self,
                                 y_true: pd.Series,
                                 y_pred: pd.Series,
                                 name: str,
                                 output_path: str,
                                 seed: int,
                                 save: bool = True) -> dict:
        """Calculate optimal probability threshold for maximum profit"""
        logger.info("Calculating optimal threshold")
        
        try:
            # Calculate profit vector
            profits = np.where(y_true == "BAJA+2", 
                             self.profit_correct, 
                             -self.cost_stimulus)
            
            # Sort by predictions
            sorted_indices = np.argsort(y_pred)[::-1]
            sorted_profits = profits[sorted_indices]
            sorted_preds = y_pred[sorted_indices]
            
            # Find optimal point
            cumulative_profit = np.cumsum(sorted_profits)
            max_profit_idx = np.argmax(cumulative_profit)
            optimal_threshold = sorted_preds[max_profit_idx]
            
            results = {
                "umbral_optimo": float(optimal_threshold),
                "cliente": int(max_profit_idx),
                "ganancia_max": float(cumulative_profit[max_profit_idx]),
                "SEMILLA": seed
            }
            
            if save:
                self._save_threshold_results(results, name, output_path)
                
            return {
                "umbrales": results,
                "y_pred_sorted": sorted_preds,
                "ganancia_acumulada": cumulative_profit
            }
            
        except Exception as e:
            logger.error(f"Error in threshold optimization: {e}")
            raise
            
    @staticmethod
    def _save_threshold_results(results: dict, name: str, output_path: str):
        """Save threshold optimization results"""
        try:
            output_file = Path(output_path) / f"{name}_umbral_optimo.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save threshold results: {e}")
            
    def evaluate_performance(self,
                           X_test: pd.DataFrame,
                           y_test: pd.Series,
                           y_pred: pd.Series,
                           threshold_mode: str,
                           threshold: Union[float, int],
                           seed: int,
                           n_splits: int = 5) -> pd.DataFrame:
        """Evaluate model performance with public/private split"""
        logger.info(f"Evaluating performance with {n_splits} splits")
        
        splitter = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=0.3,
            random_state=seed
        )
        
        results = []
        for private_idx, public_idx in splitter.split(X_test, y_test):
            result = self._calculate_split_metrics(
                y_test, y_pred, private_idx, public_idx,
                threshold_mode, threshold
            )
            results.append(result)
            
        return self._format_evaluation_results(results)
        
    def _calculate_split_metrics(self,
                               y_true: pd.Series,
                               y_pred: pd.Series,
                               private_idx: np.ndarray,
                               public_idx: np.ndarray,
                               threshold_mode: str,
                               threshold: Union[float, int]) -> dict:
        """Calculate metrics for a single split"""
        metrics = {}
        
        for split_name, idx, prop in [("public", public_idx, 0.3),
                                    ("private", private_idx, 0.7)]:
            if threshold_mode == "prob":
                gain = self._calculate_probability_gain(
                    y_pred[idx], y_true.iloc[idx], prop, threshold
                )
            else:
                gain = self._calculate_customer_gain(
                    y_pred[idx], y_true.iloc[idx], prop, threshold
                )
            metrics[f"lgbm_{split_name}"] = gain
            
        return metrics
        
    def _calculate_probability_gain(self,
                                 y_pred: np.ndarray,
                                 y_true: pd.Series,
                                 prop: float,
                                 threshold: float) -> float:
        """Calculate gain using probability threshold"""
        profits = np.where(y_true == "BAJA+2",
                         self.profit_correct,
                         -self.cost_stimulus)
        return profits[y_pred >= threshold].sum() / prop
        
    def _calculate_customer_gain(self,
                              y_pred: np.ndarray,
                              y_true: pd.Series,
                              prop: float,
                              n_customers: int) -> float:
        """Calculate gain using top N customers"""
        threshold = int(n_customers * prop)
        profits = np.where(y_true == "BAJA+2",
                         self.profit_correct,
                         -self.cost_stimulus)
        sorted_profits = profits[np.argsort(y_pred)[::-1]]
        return sorted_profits[:threshold].sum() / prop
        
    @staticmethod
    def _format_evaluation_results(results: List[dict]) -> pd.DataFrame:
        """Format evaluation results into a DataFrame"""
        df = pd.DataFrame(results)
        df_long = df.reset_index().melt(
            id_vars=['index'],
            var_name='model_type',
            value_name='ganancia'
        )
        df_long[['modelo', 'tipo']] = df_long['model_type'].str.split('_', expand=True)
        return df_long[['ganancia', 'tipo', 'modelo']]