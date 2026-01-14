"""
自動インサイト生成モジュール
FR-014: 自動インサイト
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from app.eda.correlation import calculate_correlation_matrix, get_top_correlations


def detect_strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """
    強い相関を検出
    
    Args:
        df: DataFrame
        threshold: 相関の閾値（絶対値）
        
    Returns:
        list: 強い相関のペアリスト
    """
    corr_matrix = calculate_correlation_matrix(df)
    
    if corr_matrix.empty:
        return []
    
    strong_corrs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if not pd.isna(corr_value) and abs(corr_value) >= threshold:
                strong_corrs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': float(corr_value),
                    'type': 'positive' if corr_value > 0 else 'negative'
                })
    
    return sorted(strong_corrs, key=lambda x: abs(x['correlation']), reverse=True)


def detect_class_imbalance(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    クラス不均衡を検出（分類問題用）
    
    Args:
        df: DataFrame
        target_column: ターゲット列名
        
    Returns:
        dict: クラス不均衡の情報
    """
    if target_column not in df.columns:
        return {'is_imbalanced': bool(False), 'message': 'Target column not found'}
    
    target_series = df[target_column].dropna()
    
    if len(target_series) == 0:
        return {'is_imbalanced': bool(False), 'message': 'No valid target values'}
    
    value_counts = target_series.value_counts()
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = min_count / max_count if max_count > 0 else 0
    
    # 不均衡の閾値: 最小クラスが最大クラスの20%未満
    is_imbalanced = bool(imbalance_ratio < 0.2)
    
    return {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': float(imbalance_ratio),
        'class_distribution': {str(k): int(v) for k, v in value_counts.items()},
        'message': f'クラス不均衡が検出されました（比率: {imbalance_ratio:.2%}）' if is_imbalanced else f'クラス分布は比較的均等です（比率: {imbalance_ratio:.2%}）'
    }


def detect_outliers_iqr(df: pd.DataFrame, threshold: float = 1.5) -> Dict[str, Any]:
    """
    IQR法で外れ値を検出
    
    Args:
        df: DataFrame
        threshold: IQRの倍数（デフォルト1.5）
        
    Returns:
        dict: 外れ値の情報
    """
    outlier_info = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        if len(data) < 4:  # IQR計算には最低4つの値が必要
            continue
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_ratio = len(outliers) / len(data) if len(data) > 0 else 0
        
        if outlier_ratio > 0.05:  # 5%以上の外れ値がある列を報告
            outlier_info[col] = {
                'outlier_count': int(len(outliers)),
                'outlier_ratio': float(outlier_ratio),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
    
    # 外れ値の多い列をトップ5取得
    top_outlier_columns = sorted(
        outlier_info.items(),
        key=lambda x: x[1]['outlier_ratio'],
        reverse=True
    )[:5]
    
    return {
        'columns_with_outliers': {k: v for k, v in outlier_info.items()},
        'top_outlier_columns': [
            {'column': col, 'outlier_ratio': info['outlier_ratio'], 'outlier_count': info['outlier_count']}
            for col, info in top_outlier_columns
        ],
        'total_columns_with_outliers': len(outlier_info)
    }


def generate_insights(df: pd.DataFrame, target_column: str = None, problem_type: str = None) -> Dict[str, Any]:
    """
    自動インサイトを生成
    
    Args:
        df: DataFrame
        target_column: ターゲット列名
        problem_type: 問題種別（classification/regression）
        
    Returns:
        dict: インサイト情報
    """
    insights = {
        'strong_correlations': [],
        'class_imbalance': None,
        'outliers': {},
        'summary_text': []
    }
    
    # 強い相関の検出
    strong_corrs = detect_strong_correlations(df, threshold=0.7)
    insights['strong_correlations'] = strong_corrs
    
    if strong_corrs:
        insights['summary_text'].append(
            f"強い相関が{len(strong_corrs)}組検出されました。"
        )
    
    # クラス不均衡の検出（分類問題の場合）
    if problem_type == 'classification' and target_column:
        class_imbalance = detect_class_imbalance(df, target_column)
        insights['class_imbalance'] = class_imbalance
        insights['summary_text'].append(class_imbalance['message'])
    
    # 外れ値の検出
    outliers = detect_outliers_iqr(df)
    insights['outliers'] = outliers
    
    if outliers['total_columns_with_outliers'] > 0:
        insights['summary_text'].append(
            f"外れ値が多い列が{outliers['total_columns_with_outliers']}列検出されました。"
        )
    
    # 自然文サマリ
    insights['summary'] = ' '.join(insights['summary_text']) if insights['summary_text'] else '特に問題は検出されませんでした。'
    
    return insights

