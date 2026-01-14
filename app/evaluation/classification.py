"""
分類モデル評価モジュール
FR-040: 分類評価
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, 
    precision_score, recall_score, f1_score, 
    accuracy_score, classification_report
)
from typing import Dict, Any, Tuple, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_classification(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, Any]:
    """
    分類モデルを評価
    
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
    y_pred_proba = None
    
    # 確率予測（可能な場合）
    if hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            pass
    
    # 基本指標
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC曲線とAUC（2クラス分類の場合）
    roc_auc = None
    roc_curve_data = None
    roc_plot_path = None
    
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        # 2クラス分類の場合
        # pos_labelを明示的に指定（文字列ラベルの場合）
        unique_labels = np.unique(y_test)
        if len(unique_labels) == 2:
            # アルファベット順で2番目のクラスをpos_labelとして使用
            pos_label = sorted(unique_labels)[1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1], pos_label=pos_label)
        else:
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = float(auc(fpr, tpr))
        roc_curve_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
        
        # ROC曲線をプロット
        roc_plot_path = output_dir / 'roc_curve.png'
        create_roc_curve(fpr, tpr, roc_auc, roc_plot_path)
    
    # 混同行列をプロット
    cm_plot_path = output_dir / 'confusion_matrix.png'
    create_confusion_matrix_plot(cm, output_dir, cm_plot_path)
    
    # クラス別の指標
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'roc_curve': roc_curve_data,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report,
        'roc_plot_path': f'{output_dir.name}/roc_curve.png' if roc_plot_path else None,
        'cm_plot_path': f'{output_dir.name}/confusion_matrix.png' if cm_plot_path else None
    }


def create_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_score: float, output_path: Path):
    """ROC曲線を作成"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_confusion_matrix_plot(cm: np.ndarray, output_dir: Path, output_path: Path):
    """混同行列をプロット"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

