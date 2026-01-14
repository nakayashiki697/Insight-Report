"""
特徴量分布可視化モジュール
FR-012: 特徴量分布
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_numeric_histogram(df: pd.DataFrame, column: str, output_path: Path, bins: int = 30) -> Path:
    """
    数値列のヒストグラムを作成
    
    Args:
        df: DataFrame
        column: 列名
        output_path: 出力ファイルパス
        bins: ビンの数
        
    Returns:
        Path: 保存されたファイルのパス
    """
    plt.figure(figsize=(10, 6))
    
    data = df[column].dropna()
    
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_categorical_barplot(df: pd.DataFrame, column: str, output_path: Path, top_n: int = 20) -> Path:
    """
    カテゴリ列の棒グラフを作成
    
    Args:
        df: DataFrame
        column: 列名
        output_path: 出力ファイルパス
        top_n: 表示する上位N件
        
    Returns:
        Path: 保存されたファイルのパス
    """
    plt.figure(figsize=(12, max(6, len(df[column].value_counts().head(top_n)) * 0.5)))
    
    value_counts = df[column].value_counts().head(top_n)
    
    value_counts.plot(kind='barh')
    plt.title(f'Distribution of {column}', fontsize=14, fontweight='bold')
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_all_distributions(df: pd.DataFrame, output_dir: Path, target_column: str = None) -> Dict[str, List[str]]:
    """
    すべての特徴量の分布を可視化
    
    Args:
        df: DataFrame
        output_dir: 出力ディレクトリ
        target_column: ターゲット列（除外する場合）
        
    Returns:
        dict: 作成されたグラフファイルのパスのリスト
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    numeric_plots = []
    categorical_plots = []
    
    for col in df.columns:
        if col == target_column:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            plot_path = output_dir / f'dist_{col}.png'
            create_numeric_histogram(df, col, plot_path)
            numeric_plots.append(str(plot_path))
        else:
            plot_path = output_dir / f'dist_{col}.png'
            create_categorical_barplot(df, col, plot_path)
            categorical_plots.append(str(plot_path))
    
    return {
        'numeric': numeric_plots,
        'categorical': categorical_plots
    }

