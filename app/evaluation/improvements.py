"""
改善提案モジュール
FR-043: 改善提案
"""

from typing import Dict, Any, List


def generate_improvement_suggestions(
    evaluation_results: Dict[str, Any],
    problem_type: str,
    preprocessing_log: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    改善提案を生成
    
    Args:
        evaluation_results: 評価結果
        problem_type: 問題種別（"classification" or "regression"）
        preprocessing_log: 前処理ログ（オプション）
        
    Returns:
        list: 改善提案のリスト
    """
    suggestions = []
    
    if problem_type == 'classification':
        suggestions.extend(generate_classification_suggestions(evaluation_results, preprocessing_log))
    else:
        suggestions.extend(generate_regression_suggestions(evaluation_results, preprocessing_log))
    
    return suggestions


def generate_classification_suggestions(
    evaluation_results: Dict[str, Any],
    preprocessing_log: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """分類問題の改善提案を生成"""
    suggestions = []
    
    accuracy = evaluation_results.get('accuracy', 0)
    precision = evaluation_results.get('precision', 0)
    recall = evaluation_results.get('recall', 0)
    f1 = evaluation_results.get('f1_score', 0)
    roc_auc = evaluation_results.get('roc_auc')
    
    # 精度が低い場合
    if accuracy < 0.7:
        suggestions.append({
            'category': 'モデル性能',
            'title': 'モデル精度の改善',
            'description': '精度が70%未満です。特徴量エンジニアリングやモデルの調整を検討してください。',
            'priority': 'high'
        })
    
    # AUCが低い場合
    if roc_auc is not None and roc_auc < 0.7:
        suggestions.append({
            'category': 'モデル性能',
            'title': 'AUCの改善',
            'description': 'AUCが0.7未満です。クラス不均衡の対処や特徴量の追加を検討してください。',
            'priority': 'high'
        })
    
    # 適合率と再現率のバランス
    if precision < 0.7 and recall < 0.7:
        suggestions.append({
            'category': 'モデル性能',
            'title': '適合率・再現率の改善',
            'description': '適合率と再現率が低いです。特徴量の質を向上させるか、モデルのハイパーパラメータを調整してください。',
            'priority': 'medium'
        })
    elif precision < recall:
        suggestions.append({
            'category': 'モデル性能',
            'title': '適合率の改善',
            'description': '適合率が再現率より低いです。False Positiveを減らすため、モデルの閾値を調整するか、特徴量を追加してください。',
            'priority': 'medium'
        })
    elif recall < precision:
        suggestions.append({
            'category': 'モデル性能',
            'title': '再現率の改善',
            'description': '再現率が適合率より低いです。False Negativeを減らすため、モデルの閾値を調整するか、データを増やしてください。',
            'priority': 'medium'
        })
    
    # 前処理に関する提案
    if preprocessing_log:
        numeric_count = preprocessing_log.get('numeric_count', 0)
        categorical_count = preprocessing_log.get('categorical_count', 0)
        
        if numeric_count == 0 and categorical_count > 0:
            suggestions.append({
                'category': '特徴量',
                'title': '数値特徴量の追加',
                'description': '数値特徴量がありません。数値特徴量を追加することで予測性能が向上する可能性があります。',
                'priority': 'low'
            })
        
        if categorical_count == 0 and numeric_count > 0:
            suggestions.append({
                'category': '特徴量',
                'title': 'カテゴリ特徴量の追加',
                'description': 'カテゴリ特徴量がありません。カテゴリ特徴量を追加することで予測性能が向上する可能性があります。',
                'priority': 'low'
            })
    
    return suggestions


def generate_regression_suggestions(
    evaluation_results: Dict[str, Any],
    preprocessing_log: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """回帰問題の改善提案を生成"""
    suggestions = []
    
    r2 = evaluation_results.get('r2_score', 0)
    rmse = evaluation_results.get('rmse', 0)
    
    # R²が低い場合
    if r2 < 0.7:
        suggestions.append({
            'category': 'モデル性能',
            'title': 'R²の改善',
            'description': 'R²が0.7未満です。特徴量エンジニアリングやモデルの調整を検討してください。',
            'priority': 'high'
        })
    
    if r2 < 0.5:
        suggestions.append({
            'category': 'モデル性能',
            'title': '予測精度の大幅な改善',
            'description': 'R²が0.5未満です。データの質を確認し、より適切な特徴量を選択してください。',
            'priority': 'high'
        })
    
    # 前処理に関する提案
    if preprocessing_log:
        numeric_count = preprocessing_log.get('numeric_count', 0)
        categorical_count = preprocessing_log.get('categorical_count', 0)
        
        if numeric_count < 3:
            suggestions.append({
                'category': '特徴量',
                'title': '特徴量の追加',
                'description': '数値特徴量が少ないです。より多くの特徴量を追加することで予測性能が向上する可能性があります。',
                'priority': 'medium'
            })
    
    return suggestions

