"""
分類モデル候補モジュール
FR-030: 分類モデル候補
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from typing import Dict, Any, List
import numpy as np


def get_classification_models() -> Dict[str, Any]:
    """
    分類モデルの候補とハイパーパラメータ探索範囲を取得
    
    Returns:
        dict: モデル名とモデル・パラメータの辞書
    """
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'param_distributions': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']  # l1ペナルティはliblinearのみ対応
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_distributions': {
                'n_estimators': [50, 100],  # 高速化: 200を削除
                'max_depth': [5, 10, 15],   # 高速化: Noneを削除（深さ制限）
                'min_samples_split': [2, 5]
            }
        },
        'SVC': {
            'model': SVC(random_state=42, probability=True),
            'param_distributions': {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf']
            }
        }
    }
    
    return models


def train_classification_model(model_name: str, X_train, y_train, cv: int = 5, n_iter: int = 20) -> Dict[str, Any]:
    """
    分類モデルを学習
    
    Args:
        model_name: モデル名
        X_train: 訓練データ
        y_train: 訓練ターゲット
        cv: クロスバリデーションの分割数
        n_iter: RandomizedSearchCVの試行回数
        
    Returns:
        dict: 学習結果（モデル、スコア、パラメータなど）
    """
    models = get_classification_models()
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_config = models[model_name]
    base_model = model_config['model']
    param_dist = model_config['param_distributions']
    
    # RandomizedSearchCVでハイパーパラメータ探索
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    search.fit(X_train, y_train)
    
    return {
        'model': search.best_estimator_,
        'best_score': float(search.best_score_),
        'best_params': search.best_params_,
        'cv_scores': search.cv_results_['mean_test_score'].tolist()
    }

