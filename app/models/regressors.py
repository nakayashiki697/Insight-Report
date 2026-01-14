"""
回帰モデル候補モジュール
FR-031: 回帰モデル候補
"""

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any
import numpy as np


def get_regression_models() -> Dict[str, Any]:
    """
    回帰モデルの候補とハイパーパラメータ探索範囲を取得
    
    Returns:
        dict: モデル名とモデル・パラメータの辞書
    """
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'param_distributions': {}
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'param_distributions': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'param_distributions': {
                'n_estimators': [50, 100],  # 高速化: 200を削除
                'max_depth': [5, 10, 15],   # 高速化: Noneを削除
                'min_samples_split': [2, 5]
            }
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'param_distributions': {
                'n_estimators': [50, 100],  # 高速化: 200を削除
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
    }
    
    return models


def train_regression_model(model_name: str, X_train, y_train, cv: int = 5, n_iter: int = 20) -> Dict[str, Any]:
    """
    回帰モデルを学習
    
    Args:
        model_name: モデル名
        X_train: 訓練データ
        y_train: 訓練ターゲット
        cv: クロスバリデーションの分割数
        n_iter: RandomizedSearchCVの試行回数
        
    Returns:
        dict: 学習結果（モデル、スコア、パラメータなど）
    """
    models = get_regression_models()
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = models[model_name]
    base_model = model_config['model']
    param_dist = model_config['param_distributions']
    
    # パラメータがない場合はそのまま学習
    if not param_dist:
        base_model.fit(X_train, y_train)
        return {
            'model': base_model,
            'best_score': None,  # 後で評価する
            'best_params': {},
            'cv_scores': []
        }
    
    # RandomizedSearchCVでハイパーパラメータ探索
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    return {
        'model': search.best_estimator_,
        'best_score': float(-search.best_score_),  # RMSEに変換（後で計算）
        'best_params': search.best_params_,
        'cv_scores': [-score for score in search.cv_results_['mean_test_score']]
    }

