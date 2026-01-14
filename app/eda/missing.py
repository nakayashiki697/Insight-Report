"""
欠損値分析モジュール
FR-011: 欠損値可視化
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    欠損値の分析
    
    Args:
        df: DataFrame
        
    Returns:
        dict: 欠損値の分析結果
    """
    missing_info = {}
    
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_ratio = float(null_count / len(df))
        
        missing_info[col] = {
            'null_count': null_count,
            'null_ratio': null_ratio,
            'has_missing': bool(null_count > 0)
        }
    
    # 欠損値の多い列をトップ5取得
    missing_sorted = sorted(
        missing_info.items(),
        key=lambda x: x[1]['null_ratio'],
        reverse=True
    )[:5]
    
    return {
        'column_info': missing_info,
        'top_missing_columns': [
            {'column': col, 'null_count': info['null_count'], 'null_ratio': info['null_ratio']}
            for col, info in missing_sorted if info['has_missing']
        ],
        'total_missing': int(df.isna().sum().sum()),
        'columns_with_missing': int(sum(1 for info in missing_info.values() if info['has_missing']))
    }


def create_missing_heatmap(df: pd.DataFrame, output_path: Path) -> Path:
    """
    欠損値ヒートマップを作成
    
    Args:
        df: DataFrame
        output_path: 出力ファイルパス
        
    Returns:
        Path: 保存されたファイルのパス
    """
    plt.figure(figsize=(12, max(6, len(df.columns) * 0.3)))
    
    # 欠損値の有無をTrue/Falseで表現
    missing_matrix = df.isnull()
    
    sns.heatmap(
        missing_matrix,
        yticklabels=False,
        cbar=True,
        cmap='viridis',
        cbar_kws={'label': 'Missing Values'}
    )
    
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.xlabel('Columns', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_missing_barplot(df: pd.DataFrame, output_path: Path) -> Path:
    """
    欠損値の割合を棒グラフで可視化
    
    Args:
        df: DataFrame
        output_path: 出力ファイルパス
        
    Returns:
        Path: 保存されたファイルのパス
    """
    missing_ratios = df.isnull().sum() / len(df)
    missing_ratios = missing_ratios[missing_ratios > 0].sort_values(ascending=False)
    
    if len(missing_ratios) == 0:
        # 欠損値がない場合、空のグラフを作成
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No Missing Values', 
                ha='center', va='center', fontsize=16)
        plt.title('Missing Values Ratio', fontsize=14, fontweight='bold')
        plt.axis('off')
    else:
        plt.figure(figsize=(12, max(6, len(missing_ratios) * 0.5)))
        missing_ratios.plot(kind='barh')
        plt.title('Missing Values Ratio by Column', fontsize=14, fontweight='bold')
        plt.xlabel('Missing Ratio', fontsize=12)
        plt.ylabel('Columns', fontsize=12)
        plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

