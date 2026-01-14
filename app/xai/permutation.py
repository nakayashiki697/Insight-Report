"""
Permutation Importance計算モジュール
FR-050: Permutation Importance
"""

import numpy as np
from sklearn.inspection import permutation_importance
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def calculate_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    problem_type: str,
    output_dir: Path,
    n_repeats: int = 10,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Permutation Importanceを計算し、可視化する
    
    Args:
        model: 学習済みモデル
        X_test: テストデータ
        y_test: テストターゲット
        feature_names: 特徴量名のリスト
        problem_type: 問題種別（"classification" or "regression"）
        output_dir: 出力ディレクトリ
        n_repeats: シャッフルの繰り返し回数
        random_state: 乱数シード
        
    Returns:
        dict: Permutation Importance結果
    """
    # スコアリング関数を決定
    if problem_type == 'classification':
        scoring = 'roc_auc' if len(np.unique(y_test)) == 2 else 'accuracy'
    else:
        scoring = 'neg_mean_squared_error'
    
    # Permutation Importanceの計算
    print(f"[DEBUG] Calculating Permutation Importance (n_repeats={n_repeats}, scoring={scoring})...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )
    
    # 重要度の取得
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    
    # 上位5特徴量を抽出
    top_indices = np.argsort(importances_mean)[::-1][:5]
    
    # 特徴量名の範囲チェック
    if len(feature_names) != len(importances_mean):
        # 特徴量名の数が一致しない場合、インデックスベースの名前を使用
        print(f"[WARNING] Feature names count ({len(feature_names)}) doesn't match importance count ({len(importances_mean)})")
        top_features = [f'feature_{i}' for i in top_indices]
    else:
        top_features = [feature_names[i] if i < len(feature_names) else f'feature_{i}' for i in top_indices]
    
    top_importances = importances_mean[top_indices]
    top_std = importances_std[top_indices]
    
    # 可視化
    plot_path = output_dir / 'permutation_importance.png'
    _plot_permutation_importance(
        top_features,
        top_importances,
        top_std,
        plot_path
    )
    
    print(f"[DEBUG] Permutation Importance plot saved: {plot_path}")
    
    return {
        'importances_mean': importances_mean.tolist(),
        'importances_std': importances_std.tolist(),
        'top_features': top_features,
        'top_importances': top_importances.tolist(),
        'top_std': top_std.tolist(),
        'top_indices': top_indices.tolist(),
        'plot_path': str(plot_path.relative_to(output_dir.parent))
    }


def _plot_permutation_importance(
    feature_names: List[str],
    importances: np.ndarray,
    std: np.ndarray,
    output_path: Path
):
    """
    Permutation Importanceを棒グラフで可視化
    
    Args:
        feature_names: 特徴量名のリスト
        importances: 重要度の配列
        std: 標準偏差の配列
        output_path: 出力パス
    """
    plt.figure(figsize=(10, 6))
    
    # 特徴量名を短縮（長すぎる場合）
    display_names = [name[:30] + '...' if len(name) > 30 else name for name in feature_names]
    
    # 棒グラフを描画（重要度の降順）
    y_pos = np.arange(len(feature_names))
    bars = plt.barh(y_pos, importances, xerr=std, capsize=5, alpha=0.7)
    
    # カラーバー（重要度に応じて色を変える）
    colors = plt.cm.viridis(importances / importances.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(y_pos, display_names)
    plt.xlabel('Permutation Importance', fontsize=12)
    plt.title('Top 5 Feature Importance (Permutation Importance)', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # レイアウト調整
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

