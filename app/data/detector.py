"""
問題種別判定モジュール
FR-003: 問題種別の自動判定
"""

import pandas as pd
import numpy as np
from typing import Literal


def detect_problem_type(df: pd.DataFrame, target_column: str) -> Literal["classification", "regression"]:
    """
    問題種別を自動判定
    
    判定ロジック:
    - ターゲット列が数値型で、ユニーク値数が多ければ回帰
    - ターゲット列がカテゴリ型、またはユニーク値数が少なければ分類
    
    Args:
        df: DataFrame
        target_column: ターゲット列名
        
    Returns:
        "classification" or "regression"
        
    Raises:
        ValueError: ターゲット列が存在しない場合
    """
    if target_column not in df.columns:
        raise ValueError(f"ターゲット列 '{target_column}' が存在しません")
    
    target_series = df[target_column]
    
    # 欠損値を除外
    target_series = target_series.dropna()
    
    if len(target_series) == 0:
        raise ValueError("ターゲット列に有効なデータがありません")
    
    # データ型を確認
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    # ユニーク値の数を取得
    unique_count = target_series.nunique()
    total_count = len(target_series)
    unique_ratio = unique_count / total_count
    
    # 判定ロジック
    if is_numeric:
        # 数値型の場合
        # ユニーク値の割合が高い（>0.5）またはユニーク値数が多い（>20）場合は回帰
        if unique_ratio > 0.5 or unique_count > 20:
            return "regression"
        else:
            # ユニーク値が少ない場合は分類（例: 0/1, 1-5の評価など）
            return "classification"
    else:
        # カテゴリ型の場合は分類
        return "classification"


def get_column_info(df: pd.DataFrame) -> dict:
    """
    列の情報を取得
    
    Args:
        df: DataFrame
        
    Returns:
        dict: 列情報（列名、データ型、ユニーク値数など）
        すべての値はJSONシリアライズ可能な型（int, float, str, bool, None）に変換される
    """
    column_info = {}
    
    for col in df.columns:
        col_series = df[col].dropna()
        
        # numpy/pandas型をPython標準型に変換
        unique_count = int(df[col].nunique())
        null_count = int(df[col].isna().sum())
        null_ratio = float(df[col].isna().sum() / len(df))
        
        column_info[col] = {
            'dtype': str(df[col].dtype),
            'is_numeric': bool(pd.api.types.is_numeric_dtype(df[col])),
            'unique_count': unique_count,
            'null_count': null_count,
            'null_ratio': null_ratio,
        }
        
        # 数値列の場合、統計情報を追加
        if column_info[col]['is_numeric']:
            if len(col_series) > 0:
                column_info[col]['min'] = float(col_series.min())
                column_info[col]['max'] = float(col_series.max())
                column_info[col]['mean'] = float(col_series.mean())
            else:
                column_info[col]['min'] = None
                column_info[col]['max'] = None
                column_info[col]['mean'] = None
    
    return column_info

