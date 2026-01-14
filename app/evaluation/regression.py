"""
回帰モデル評価モジュール
FR-041: 回帰評価
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_regression(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """
    回帰モデルを評価
    
    Args:
        model: 学習済みモデル
        X_test: テストデータ
        y_test: テストターゲット
        output_dir: 出力ディレクトリ
        
    Returns:
        dict: 評価結果
    """
    # 予測
    y_pred = model.predict(X_test)
    
    # 指標計算
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))
    
    # 残差計算
    residuals = y_test - y_pred
    
    # 散布図（予測値 vs 実際値）
    scatter_plot_path = output_dir / 'scatter_plot.png'
    create_scatter_plot(y_test, y_pred, r2, scatter_plot_path)
    
    # 残差プロット
    residual_plot_path = output_dir / 'residual_plot.png'
    create_residual_plot(y_pred, residuals, residual_plot_path)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'residuals': residuals.tolist(),
        'scatter_plot_path': f'{output_dir.name}/scatter_plot.png',
        'residual_plot_path': f'{output_dir.name}/residual_plot.png'
    }


def create_scatter_plot(y_true: np.ndarray, y_pred: np.ndarray, r2: float, output_path: Path):
    """予測値 vs 実際値の散布図を作成"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # 理想的な予測線（y=x）
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Predicted vs Actual (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_residual_plot(y_pred: np.ndarray, residuals: np.ndarray, output_path: Path):
    """残差プロットを作成"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, s=50)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

