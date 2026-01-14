"""
評価サマリ生成モジュール
FR-042: 評価サマリ
"""

from typing import Dict, Any


def generate_evaluation_summary(
    evaluation_results: Dict[str, Any],
    problem_type: str
) -> Dict[str, Any]:
    """
    評価結果のサマリを生成
    
    Args:
        evaluation_results: 評価結果
        problem_type: 問題種別（"classification" or "regression"）
        
    Returns:
        dict: 評価サマリ
    """
    if problem_type == 'classification':
        return generate_classification_summary(evaluation_results)
    else:
        return generate_regression_summary(evaluation_results)


def generate_classification_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """分類問題の評価サマリを生成"""
    accuracy = evaluation_results.get('accuracy', 0)
    precision = evaluation_results.get('precision', 0)
    recall = evaluation_results.get('recall', 0)
    f1 = evaluation_results.get('f1_score', 0)
    roc_auc = evaluation_results.get('roc_auc')
    
    # サマリテキストを生成
    summary_parts = []
    
    summary_parts.append(f"モデルの精度（Accuracy）は {accuracy:.3f} です。")
    
    if roc_auc is not None:
        summary_parts.append(f"ROC曲線下面積（AUC）は {roc_auc:.3f} です。")
        if roc_auc > 0.9:
            summary_parts.append("AUCが0.9を超えており、優れた分類性能を示しています。")
        elif roc_auc > 0.7:
            summary_parts.append("AUCが0.7を超えており、良好な分類性能を示しています。")
        else:
            summary_parts.append("AUCが0.7未満であり、改善の余地があります。")
    
    summary_parts.append(f"適合率（Precision）は {precision:.3f}、再現率（Recall）は {recall:.3f}、F1スコアは {f1:.3f} です。")
    
    if precision < 0.7 or recall < 0.7:
        summary_parts.append("適合率または再現率が低い場合、クラス不均衡や特徴量の不足が原因の可能性があります。")
    
    summary = " ".join(summary_parts)
    
    return {
        'summary': summary,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    }


def generate_regression_summary(evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
    """回帰問題の評価サマリを生成"""
    rmse = evaluation_results.get('rmse', 0)
    mae = evaluation_results.get('mae', 0)
    r2 = evaluation_results.get('r2_score', 0)
    
    # サマリテキストを生成
    summary_parts = []
    
    summary_parts.append(f"モデルの決定係数（R²）は {r2:.3f} です。")
    
    if r2 > 0.9:
        summary_parts.append("R²が0.9を超えており、優れた予測性能を示しています。")
    elif r2 > 0.7:
        summary_parts.append("R²が0.7を超えており、良好な予測性能を示しています。")
    elif r2 > 0.5:
        summary_parts.append("R²が0.5を超えており、中程度の予測性能を示しています。")
    else:
        summary_parts.append("R²が0.5未満であり、改善の余地があります。")
    
    summary_parts.append(f"平均二乗誤差の平方根（RMSE）は {rmse:.3f}、平均絶対誤差（MAE）は {mae:.3f} です。")
    
    if r2 < 0.7:
        summary_parts.append("予測精度を向上させるため、特徴量エンジニアリングやモデルの調整を検討してください。")
    
    summary = " ".join(summary_parts)
    
    return {
        'summary': summary,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
    }

