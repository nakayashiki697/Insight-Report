"""
XAI解説文生成モジュール
FR-052: XAI解説文生成
"""

from typing import Dict, Any, List


def generate_xai_summary(
    permutation_results: Dict[str, Any],
    pdp_results: Dict[str, Any],
    problem_type: str
) -> Dict[str, str]:
    """
    XAI結果から自然文の解説を生成
    
    Args:
        permutation_results: Permutation Importance結果
        pdp_results: PDP結果
        problem_type: 問題種別（"classification" or "regression"）
        
    Returns:
        dict: 解説文の辞書
    """
    top_features = permutation_results.get('top_features', [])
    top_importances = permutation_results.get('top_importances', [])
    pdp_plots = pdp_results.get('pdp_plots', [])
    plot_paths = pdp_results.get('plot_paths', [])
    
    # Permutation Importanceの解説
    permutation_summary = _generate_permutation_summary(top_features, top_importances, problem_type)
    
    # PDPの解説（プロットパスも渡す）
    pdp_summary = _generate_pdp_summary(pdp_plots, plot_paths)
    
    # 全体のサマリ
    overall_summary = _generate_overall_summary(permutation_summary, pdp_summary, problem_type)
    
    return {
        'permutation_summary': permutation_summary,
        'pdp_summary': pdp_summary,
        'overall_summary': overall_summary
    }


def _generate_permutation_summary(
    top_features: List[str],
    top_importances: List[float],
    problem_type: str
) -> str:
    """Permutation Importanceの解説を生成"""
    if not top_features:
        return "特徴量重要度の計算に失敗しました。"
    
    summary_parts = []
    
    if problem_type == 'classification':
        summary_parts.append("この分類モデルでは、以下の特徴量が予測に最も影響を与えています：")
    else:
        summary_parts.append("この回帰モデルでは、以下の特徴量が予測に最も影響を与えています：")
    
    # 上位3つを詳しく説明
    for i, (feature, importance) in enumerate(zip(top_features[:3], top_importances[:3]), 1):
        summary_parts.append(f"{i}. **{feature}**（重要度: {importance:.4f}）")
    
    # 最も重要な特徴量について追加説明
    if len(top_features) > 0:
        top_feature = top_features[0]
        top_importance = top_importances[0]
        
        if top_importance > 0.1:
            summary_parts.append(f"\n特に**{top_feature}**は最も重要な特徴量であり、"
                               f"この特徴量をランダムにシャッフルするとモデルの性能が"
                               f"大きく低下します（重要度: {top_importance:.4f}）。")
        elif top_importance > 0.05:
            summary_parts.append(f"\n**{top_feature}**は最も重要な特徴量ですが、"
                               f"重要度は中程度です（{top_importance:.4f}）。")
        else:
            summary_parts.append(f"\n**{top_feature}**は最も重要な特徴量ですが、"
                               f"重要度は低めです（{top_importance:.4f}）。"
                               f"他の特徴量との組み合わせが重要かもしれません。")
    
    return "\n".join(summary_parts)


def _generate_pdp_summary(pdp_plots: List[Dict[str, Any]], plot_paths: List[str] = None) -> str:
    """PDPの解説を生成"""
    # プロットパスがある場合は成功とみなす
    if plot_paths and len(plot_paths) > 0:
        summary_parts = []
        summary_parts.append(f"特徴量と予測値の関係を分析した結果、{len(plot_paths)}個のPDPプロットが生成されました。")
        summary_parts.append("各プロットから、特徴量の値が変化した際の予測値への影響を確認できます。")
        
        # pdp_plotsに詳細情報がある場合は追加説明
        if pdp_plots:
            summary_parts.append("\n詳細な分析結果：")
            for pdp_info in pdp_plots:
                feature_name = pdp_info.get('feature_name', '特徴量')
                trend = pdp_info.get('trend', 'complex')
                min_val = pdp_info.get('min_value', 0)
                max_val = pdp_info.get('max_value', 0)
                value_range = pdp_info.get('range', 0)
                
                if trend == "increasing":
                    summary_parts.append(f"- **{feature_name}**: 値が増加するにつれて、予測値も上昇する傾向があります。")
                elif trend == "decreasing":
                    summary_parts.append(f"- **{feature_name}**: 値が増加するにつれて、予測値は下降する傾向があります。")
                else:
                    summary_parts.append(f"- **{feature_name}**: 値の変化に対して予測値は複雑な関係を示しています。")
                
                if value_range > 0.1:
                    summary_parts.append(f"  この特徴量は予測値に大きな影響を与えています（変動幅: {value_range:.3f}）。")
        
        return "\n".join(summary_parts)
    
    return "PDP（Partial Dependence Plot）の生成に失敗しました。"


def _generate_overall_summary(
    permutation_summary: str,
    pdp_summary: str,
    problem_type: str
) -> str:
    """全体のサマリを生成"""
    summary_parts = []
    
    if problem_type == 'classification':
        summary_parts.append("## モデルの説明可能性分析結果")
        summary_parts.append("\nこの分類モデルの予測メカニズムを分析した結果、以下のことが明らかになりました：")
    else:
        summary_parts.append("## モデルの説明可能性分析結果")
        summary_parts.append("\nこの回帰モデルの予測メカニズムを分析した結果、以下のことが明らかになりました：")
    
    summary_parts.append("\n### 特徴量重要度")
    summary_parts.append(permutation_summary)
    
    summary_parts.append("\n### 特徴量と予測値の関係")
    summary_parts.append(pdp_summary)
    
    summary_parts.append("\nこれらの分析結果から、モデルがどの特徴量を重視して予測を行っているかが理解できます。")
    
    return "\n".join(summary_parts)

