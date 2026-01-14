"""
相関分析モジュール
FR-013: 相関分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_correlation_matrix(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    """
    相関行列を計算
    
    Args:
        df: DataFrame
        numeric_only: 数値列のみを対象にするか
        
    Returns:
        DataFrame: 相関行列
    """
    if numeric_only:
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) < 2:
            return pd.DataFrame()
        return numeric_df.corr()
    else:
        # カテゴリ列も含める場合は、エンコーディングが必要
        # 簡易版では数値列のみ
        return calculate_correlation_matrix(df, numeric_only=True)


def get_top_correlations(corr_matrix: pd.DataFrame, n: int = 5, exclude_diagonal: bool = True) -> List[Dict[str, Any]]:
    """
    相関の高いペアをトップN取得
    
    Args:
        corr_matrix: 相関行列
        n: 取得するペア数
        exclude_diagonal: 対角成分（自己相関）を除外するか
        
    Returns:
        list: 相関の高いペアのリスト
    """
    if corr_matrix.empty:
        return []
    
    # 上三角行列を取得（重複を避ける）
    corr_matrix_upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1 if exclude_diagonal else 0).astype(bool)
    )
    
    # 相関値をフラット化
    corr_pairs = []
    for i in range(len(corr_matrix_upper.columns)):
        for j in range(i + (1 if exclude_diagonal else 0), len(corr_matrix_upper.columns)):
            col1 = corr_matrix_upper.columns[i]
            col2 = corr_matrix_upper.columns[j]
            corr_value = corr_matrix_upper.iloc[i, j]
            
            if not pd.isna(corr_value):
                corr_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': float(corr_value),
                    'abs_correlation': float(abs(corr_value))
                })
    
    # 絶対値でソートしてトップNを取得
    corr_pairs_sorted = sorted(corr_pairs, key=lambda x: x['abs_correlation'], reverse=True)
    
    return corr_pairs_sorted[:n]


def create_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: Path) -> Path:
    """
    相関ヒートマップを作成
    
    Args:
        corr_matrix: 相関行列
        output_path: 出力ファイルパス
        
    Returns:
        Path: 保存されたファイルのパス
    """
    if corr_matrix.empty or len(corr_matrix.columns) < 2:
        # 相関行列が空または列が少ない場合
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, 'Not enough numeric columns for correlation analysis', 
                ha='center', va='center', fontsize=14)
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.axis('off')
    else:
        plt.figure(figsize=(12, 10))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def analyze_correlations(df: pd.DataFrame, output_dir: Path) -> Dict[str, Any]:
    """
    相関分析を実行
    
    Args:
        df: DataFrame
        output_dir: 出力ディレクトリ
        
    Returns:
        dict: 相関分析結果
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 相関行列を計算
    corr_matrix = calculate_correlation_matrix(df)
    
    if corr_matrix.empty:
        return {
            'correlation_matrix': {},
            'top_correlations': [],
            'heatmap_path': None,
            'numeric_columns_count': 0
        }
    
    # トップ5の相関ペアを取得
    top_correlations = get_top_correlations(corr_matrix, n=5)
    
    # ヒートマップを作成
    heatmap_path = output_dir / 'correlation_heatmap.png'
    create_correlation_heatmap(corr_matrix, heatmap_path)
    
    # 相関行列をJSONシリアライズ可能な形式に変換
    corr_dict = {}
    for col1 in corr_matrix.columns:
        corr_dict[col1] = {}
        for col2 in corr_matrix.columns:
            corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2]) if not pd.isna(corr_matrix.loc[col1, col2]) else None
    
    return {
        'correlation_matrix': corr_dict,
        'top_correlations': top_correlations,
        'heatmap_path': str(heatmap_path),
        'numeric_columns_count': len(corr_matrix.columns)
    }

