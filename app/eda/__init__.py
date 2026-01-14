"""
自動EDAモジュール
"""

from app.eda.statistics import calculate_basic_statistics, get_summary_statistics
from app.eda.missing import analyze_missing_values, create_missing_heatmap, create_missing_barplot
from app.eda.distribution import create_all_distributions
from app.eda.correlation import analyze_correlations
from app.eda.insights import generate_insights

__all__ = [
    'calculate_basic_statistics',
    'get_summary_statistics',
    'analyze_missing_values',
    'create_missing_heatmap',
    'create_missing_barplot',
    'create_all_distributions',
    'analyze_correlations',
    'generate_insights'
]
