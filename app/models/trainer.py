"""
モデル学習モジュール
FR-030~034: モデル学習・比較
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import train_test_split
from app.models.classifiers import get_classification_models, train_classification_model
from app.models.regressors import get_regression_models, train_regression_model
from app.models.selector import evaluate_classification_model, evaluate_regression_model, select_best_model
from app.config import Config


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    cv: int = None,
    n_iter: int = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    すべてのモデル候補を学習・比較
    
    Args:
        X: 特徴量
        y: ターゲット
        problem_type: 問題種別（"classification" or "regression"）
        cv: クロスバリデーションの分割数（Noneの場合はConfigから取得）
        n_iter: RandomizedSearchCVの試行回数（Noneの場合はConfigから取得）
        test_size: テストデータの割合
        
    Returns:
        dict: 学習結果（各モデルの結果、ベストモデル、比較表など）
    """
    if cv is None:
        cv = Config.CV_FOLDS
    if n_iter is None:
        n_iter = Config.N_ITER_SEARCH
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=Config.RANDOM_STATE, stratify=y if problem_type == 'classification' else None
    )
    
    model_results = {}
    model_comparison = []
    
    if problem_type == 'classification':
        # 分類モデルを学習
        models = get_classification_models()
        
        for model_name in models.keys():
            try:
                print(f"Training {model_name}...")
                result = train_classification_model(model_name, X_train, y_train, cv=cv, n_iter=n_iter)
                
                # 評価を実行
                evaluation = evaluate_classification_model(result['model'], X_train, y_train, cv=cv)
                
                model_results[model_name] = {
                    **result,
                    **evaluation,
                    'training_time': 0.0  # 簡易版では記録しない
                }
                
                # 比較表用の情報
                model_comparison.append({
                    'model_name': model_name,
                    'auc': evaluation.get('auc'),
                    'accuracy': evaluation.get('accuracy'),
                    'primary_score': evaluation.get('primary_score'),
                    'best_params': result.get('best_params', {})
                })
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
    
    else:  # regression
        # 回帰モデルを学習
        models = get_regression_models()
        
        for model_name in models.keys():
            try:
                print(f"Training {model_name}...")
                result = train_regression_model(model_name, X_train, y_train, cv=cv, n_iter=n_iter)
                
                # 評価を実行
                evaluation = evaluate_regression_model(result['model'], X_train, y_train, cv=cv)
                
                model_results[model_name] = {
                    **result,
                    **evaluation,
                    'training_time': 0.0  # 簡易版では記録しない
                }
                
                # 比較表用の情報
                model_comparison.append({
                    'model_name': model_name,
                    'rmse': evaluation.get('rmse'),
                    'primary_score': evaluation.get('primary_score'),
                    'best_params': result.get('best_params', {})
                })
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
    
    # ベストモデルを選択
    best_model_name, best_model_result = select_best_model(model_results, problem_type)
    
    return {
        'model_results': model_results,
        'best_model_name': best_model_name,
        'best_model': best_model_result['model'],
        'model_comparison': model_comparison,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

