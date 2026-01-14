"""
ベストモデル選択モジュール
FR-033: ベストモデルの選択
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import cross_val_score


def evaluate_classification_model(model, X, y, cv: int = 5) -> Dict[str, float]:
    """
    分類モデルを評価
    
    Args:
        model: 学習済みモデル
        X: 特徴量
        y: ターゲット
        cv: クロスバリデーションの分割数
        
    Returns:
        dict: 評価指標（AUC、Accuracy）
    """
    try:
        # AUCを計算（2クラス分類の場合）
        if len(np.unique(y)) == 2:
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
            auc = float(np.mean(cv_scores))
        else:
            auc = None
        
        # Accuracyを計算
        cv_scores_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        accuracy = float(np.mean(cv_scores_acc))
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'primary_score': auc if auc is not None else accuracy  # プライマリスコア
        }
    except Exception as e:
        # エラーが発生した場合、Accuracyのみを使用
        cv_scores_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        accuracy = float(np.mean(cv_scores_acc))
        return {
            'auc': None,
            'accuracy': accuracy,
            'primary_score': accuracy
        }


def evaluate_regression_model(model, X, y, cv: int = 5) -> Dict[str, float]:
    """
    回帰モデルを評価
    
    Args:
        model: 学習済みモデル
        X: 特徴量
        y: ターゲット
        cv: クロスバリデーションの分割数
        
    Returns:
        dict: 評価指標（RMSE）
    """
    # RMSEを計算（neg_mean_squared_errorの平方根）
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse = float(np.sqrt(-np.mean(cv_scores)))
    
    return {
        'rmse': rmse,
        'primary_score': rmse  # プライマリスコア
    }


def select_best_model(model_results: Dict[str, Dict[str, Any]], problem_type: str) -> Tuple[str, Dict[str, Any]]:
    """
    ベストモデルを選択
    
    Args:
        model_results: 各モデルの学習結果
        problem_type: 問題種別（"classification" or "regression"）
        
    Returns:
        tuple: (ベストモデル名, ベストモデルの結果)
    """
    if not model_results:
        raise ValueError("No models to select from")
    
    if problem_type == 'classification':
        # 分類: AUC（なければAccuracy）が最も高いモデル
        best_model_name = None
        best_score = -np.inf
        
        for model_name, result in model_results.items():
            score = result.get('primary_score', -np.inf)
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
    else:  # regression
        # 回帰: RMSEが最も低いモデル
        best_model_name = None
        best_score = np.inf
        
        for model_name, result in model_results.items():
            score = result.get('primary_score', np.inf)
            if score < best_score:
                best_score = score
                best_model_name = model_name
    
    return best_model_name, model_results[best_model_name]

