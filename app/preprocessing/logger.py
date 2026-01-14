"""
前処理ログモジュール
FR-022: 前処理ログ
FR-080, FR-081: 前処理カスタマイズのログ（Phase 7.2）
"""

from typing import Dict, Any


def create_preprocessing_log(preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    前処理ログを作成
    
    Args:
        preprocessing_info: 前処理情報（build_preprocessing_pipelineから返される）
        
    Returns:
        dict: 前処理ログ
    """
    # 設定情報を取得
    config = preprocessing_info.get('config', {})
    numerical_config = config.get('numerical', {})
    categorical_config = config.get('categorical', {})
    
    # 数値処理の説明マップ
    imputer_desc_map = {
        'median': ('median', '中央値で補完'),
        'mean': ('mean', '平均値で補完'),
        'most_frequent': ('most_frequent', '最頻値で補完'),
    }
    scaler_desc_map = {
        'standard': ('StandardScaler', '標準化（平均0、標準偏差1）'),
        'minmax': ('MinMaxScaler', 'Min-Max正規化（0〜1）'),
        'robust': ('RobustScaler', 'ロバスト標準化（外れ値に強い）'),
        'none': ('なし', 'スケーリングなし'),
    }
    
    # カテゴリ処理の説明マップ
    cat_imputer_desc_map = {
        'missing': ('constant (missing)', '"missing"で補完'),
        'most_frequent': ('most_frequent', '最頻値で補完'),
    }
    encoder_desc_map = {
        'onehot': ('OneHotEncoder', 'ワンホットエンコーディング'),
        'ordinal': ('OrdinalEncoder', '順序エンコーディング'),
        'frequency': ('FrequencyEncoder', '頻度エンコーディング'),
    }
    
    # 数値処理の設定を取得
    num_imputer = numerical_config.get('imputer', 'median')
    num_scaler = numerical_config.get('scaler', 'standard')
    num_imputer_info = imputer_desc_map.get(num_imputer, ('median', '中央値で補完'))
    num_scaler_info = scaler_desc_map.get(num_scaler, ('StandardScaler', '標準化'))
    
    # カテゴリ処理の設定を取得
    cat_imputer = categorical_config.get('imputer', 'missing')
    cat_encoder = categorical_config.get('encoder', 'onehot')
    cat_max = categorical_config.get('max_categories', 20)
    cat_imputer_info = cat_imputer_desc_map.get(cat_imputer, ('constant (missing)', '"missing"で補完'))
    cat_encoder_info = encoder_desc_map.get(cat_encoder, ('OneHotEncoder', 'ワンホットエンコーディング'))
    
    # 数値特徴量の説明文を生成
    num_description = f'数値特徴量: 欠損値を{num_imputer_info[1]}'
    if num_scaler != 'none':
        num_description += f'し、{num_scaler_info[1]}を適用'
    else:
        num_description += '（スケーリングなし）'
    
    # カテゴリ特徴量の説明文を生成
    cat_description = f'カテゴリ特徴量: 欠損値を{cat_imputer_info[1]}し、{cat_encoder_info[1]}を適用'
    if cat_max < 100:
        cat_description += f'（カテゴリ上限: {cat_max}）'
    
    log = {
        'numeric_columns': preprocessing_info.get('numeric_columns', []),
        'categorical_columns': preprocessing_info.get('categorical_columns', []),
        'exclude_columns': preprocessing_info.get('exclude_columns', []),
        'numeric_count': int(preprocessing_info.get('numeric_count', 0)),
        'categorical_count': int(preprocessing_info.get('categorical_count', 0)),
        'total_features': int(preprocessing_info.get('total_features', 0)),
        'preprocessing_steps': {
            'numerical': {
                'missing_value_strategy': num_imputer_info[0],
                'scaling': num_scaler_info[0],
                'description': num_description
            },
            'categorical': {
                'missing_value_strategy': cat_imputer_info[0],
                'encoding': cat_encoder_info[0],
                'max_categories': cat_max,
                'handle_unknown': 'ignore' if cat_encoder == 'onehot' else 'use_encoded_value',
                'description': cat_description
            }
        },
        # 設定値（PDF・UI表示用）
        'config_summary': {
            'numerical': {
                'imputer': num_imputer,
                'imputer_label': num_imputer_info[1],
                'scaler': num_scaler,
                'scaler_label': num_scaler_info[1],
            },
            'categorical': {
                'imputer': cat_imputer,
                'imputer_label': cat_imputer_info[1],
                'encoder': cat_encoder,
                'encoder_label': cat_encoder_info[1],
                'max_categories': cat_max,
            },
        }
    }
    
    return log


def format_preprocessing_log_for_display(log: Dict[str, Any]) -> Dict[str, Any]:
    """
    前処理ログを表示用にフォーマット
    
    Args:
        log: 前処理ログ
        
    Returns:
        dict: 表示用フォーマット
    """
    config_summary = log.get('config_summary', {})
    numerical = config_summary.get('numerical', {})
    categorical = config_summary.get('categorical', {})
    
    return {
        'overview': {
            'numeric_count': log.get('numeric_count', 0),
            'categorical_count': log.get('categorical_count', 0),
            'total_features': log.get('total_features', 0),
            'exclude_count': len(log.get('exclude_columns', [])),
        },
        'numerical_settings': {
            'imputer': numerical.get('imputer_label', '中央値で補完'),
            'scaler': numerical.get('scaler_label', '標準化'),
            'columns': log.get('numeric_columns', []),
        },
        'categorical_settings': {
            'imputer': categorical.get('imputer_label', '"missing"で補完'),
            'encoder': categorical.get('encoder_label', 'ワンホットエンコーディング'),
            'max_categories': categorical.get('max_categories', 20),
            'columns': log.get('categorical_columns', []),
        },
        'exclude_columns': log.get('exclude_columns', []),
    }
