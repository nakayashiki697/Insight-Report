"""
基本統計量計算モジュール
FR-010: 基本統計量
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def calculate_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    基本統計量を計算
    
    Args:
        df: DataFrame
        
    Returns:
        dict: 各列の基本統計量（平均、中央値、標準偏差、最小値、最大値など）
    """
    stats = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            stats[col] = {
                'count': 0,
                'null_count': len(df[col]) - len(col_data),
                'null_ratio': 1.0
            }
            continue
        
        col_stats = {
            'count': int(len(col_data)),
            'null_count': int(df[col].isna().sum()),
            'null_ratio': float(df[col].isna().sum() / len(df))
        }
        
        # 数値列の場合
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'is_numeric': bool(True)
            })
        else:
            # カテゴリ列の場合
            col_stats.update({
                'unique_count': int(col_data.nunique()),
                'most_frequent': str(col_data.mode().iloc[0]) if len(col_data.mode()) > 0 else None,
                'most_frequent_count': int(col_data.value_counts().iloc[0]) if len(col_data.value_counts()) > 0 else 0,
                'is_numeric': bool(False)
            })
        
        stats[col] = col_stats
    
    return stats


def get_summary_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    データ全体のサマリ統計量を取得
    
    Args:
        df: DataFrame
        
    Returns:
        dict: データ全体のサマリ
    """
    return {
        'total_rows': int(len(df)),
        'total_columns': int(len(df.columns)),
        'numeric_columns': int(sum(1 for col in df.columns if pd.api.types.is_numeric_dtype(df[col]))),
        'categorical_columns': int(sum(1 for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]))),
        'total_missing_values': int(df.isna().sum().sum()),
        'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
    }

