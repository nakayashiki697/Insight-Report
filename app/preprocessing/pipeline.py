"""
前処理パイプライン構築モジュール
FR-020, FR-021: 前処理パイプライン
FR-080, FR-081, FR-082: 前処理カスタマイズ（Phase 7.2）
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from app.preprocessing.numerical import NumericalTransformer
from app.preprocessing.categorical import CategoricalTransformer
from app.config import Config


def get_default_preprocessing_config():
    """
    デフォルトの前処理設定を取得
    
    Returns:
        dict: 前処理設定
    """
    return Config.DEFAULT_PREPROCESSING_CONFIG.copy()


def validate_preprocessing_config(config):
    """
    前処理設定のバリデーション
    
    Args:
        config: 前処理設定
        
    Returns:
        dict: バリデーション済みの設定
        
    Raises:
        ValueError: 無効な設定値がある場合
    """
    default_config = get_default_preprocessing_config()
    
    # 設定がNoneまたは空の場合はデフォルトを使用
    if config is None:
        return default_config
    
    validated = {}
    
    # 数値データの設定
    numerical = config.get('numerical', {})
    validated['numerical'] = {
        'imputer': numerical.get('imputer', default_config['numerical']['imputer']),
        'scaler': numerical.get('scaler', default_config['numerical']['scaler']),
    }
    
    # カテゴリデータの設定
    categorical = config.get('categorical', {})
    max_cat = categorical.get('max_categories', default_config['categorical']['max_categories'])
    # max_categoriesの範囲チェック（10〜100）
    max_cat = max(10, min(100, int(max_cat)))
    
    validated['categorical'] = {
        'imputer': categorical.get('imputer', default_config['categorical']['imputer']),
        'encoder': categorical.get('encoder', default_config['categorical']['encoder']),
        'max_categories': max_cat,
    }
    
    # 除外列
    validated['exclude_columns'] = config.get('exclude_columns', [])
    
    return validated


def build_preprocessing_pipeline(df: pd.DataFrame, target_column: str, config=None) -> tuple:
    """
    前処理パイプラインを構築
    
    Args:
        df: DataFrame
        target_column: ターゲット列名
        config: 前処理設定（Noneの場合はデフォルト設定を使用）
        
    Returns:
        tuple: (前処理パイプライン, 特徴量名リスト, 前処理情報)
    """
    # 設定のバリデーション
    config = validate_preprocessing_config(config)
    
    # ターゲット列を除外
    feature_df = df.drop(columns=[target_column])
    
    # 除外列を処理
    exclude_columns = config.get('exclude_columns', [])
    # 存在する列のみ除外
    exclude_columns = [col for col in exclude_columns if col in feature_df.columns]
    if exclude_columns:
        feature_df = feature_df.drop(columns=exclude_columns)
        print(f"[INFO] Excluded columns: {exclude_columns}")
    
    # 数値列とカテゴリ列を分離
    numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # トランスフォーマーを作成（設定を適用）
    numerical_config = config.get('numerical', {})
    categorical_config = config.get('categorical', {})
    
    numerical_transformer = NumericalTransformer(
        imputer_strategy=numerical_config.get('imputer', 'median'),
        scaler_type=numerical_config.get('scaler', 'standard')
    )
    
    categorical_transformer = CategoricalTransformer(
        imputer_strategy=categorical_config.get('imputer', 'missing'),
        encoder_type=categorical_config.get('encoder', 'onehot'),
        max_categories=categorical_config.get('max_categories', Config.MAX_CATEGORIES)
    )
    
    # 前処理ステップを定義
    transformers = []
    if numeric_columns:
        transformers.append(('num', numerical_transformer, numeric_columns))
    if categorical_columns:
        transformers.append(('cat', categorical_transformer, categorical_columns))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # パイプライン情報を記録（設定情報を含む）
    preprocessing_info = {
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'exclude_columns': exclude_columns,
        'total_features': len(numeric_columns) + len(categorical_columns),
        'numeric_count': len(numeric_columns),
        'categorical_count': len(categorical_columns),
        # 適用された設定
        'config': {
            'numerical': {
                'imputer': numerical_config.get('imputer', 'median'),
                'scaler': numerical_config.get('scaler', 'standard'),
            },
            'categorical': {
                'imputer': categorical_config.get('imputer', 'missing'),
                'encoder': categorical_config.get('encoder', 'onehot'),
                'max_categories': categorical_config.get('max_categories', Config.MAX_CATEGORIES),
            },
        },
    }
    
    return preprocessor, feature_df.columns.tolist(), preprocessing_info


def apply_preprocessing(preprocessor, X_train, X_test=None):
    """
    前処理を適用
    
    Args:
        preprocessor: 前処理パイプライン
        X_train: 訓練データ
        X_test: テストデータ（オプション）
        
    Returns:
        tuple: (変換後の訓練データ, 変換後のテストデータ or None)
    """
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    if X_test is not None:
        X_test_transformed = preprocessor.transform(X_test)
        return X_train_transformed, X_test_transformed
    
    return X_train_transformed, None


def get_preprocessing_summary(preprocessing_info):
    """
    前処理の設定内容をわかりやすいテキストで返す（レポート・ログ用）
    
    Args:
        preprocessing_info: build_preprocessing_pipelineから返される情報
        
    Returns:
        dict: 表示用の前処理サマリ
    """
    config = preprocessing_info.get('config', {})
    numerical = config.get('numerical', {})
    categorical = config.get('categorical', {})
    
    # 数値処理の説明
    imputer_desc = {
        'median': '中央値で補完',
        'mean': '平均値で補完',
        'most_frequent': '最頻値で補完',
    }
    scaler_desc = {
        'standard': '標準化（平均0、標準偏差1）',
        'minmax': 'Min-Max正規化（0〜1）',
        'robust': 'ロバスト標準化（外れ値に強い）',
        'none': 'スケーリングなし',
    }
    
    # カテゴリ処理の説明
    cat_imputer_desc = {
        'missing': '"missing"として扱う',
        'most_frequent': '最頻値で補完',
    }
    encoder_desc = {
        'onehot': 'ワンホットエンコーディング',
        'ordinal': '順序エンコーディング',
        'frequency': '頻度エンコーディング',
    }
    
    return {
        'numerical': {
            'count': preprocessing_info.get('numeric_count', 0),
            'columns': preprocessing_info.get('numeric_columns', []),
            'imputer': imputer_desc.get(numerical.get('imputer', 'median'), '中央値で補完'),
            'scaler': scaler_desc.get(numerical.get('scaler', 'standard'), '標準化'),
        },
        'categorical': {
            'count': preprocessing_info.get('categorical_count', 0),
            'columns': preprocessing_info.get('categorical_columns', []),
            'imputer': cat_imputer_desc.get(categorical.get('imputer', 'missing'), '"missing"として扱う'),
            'encoder': encoder_desc.get(categorical.get('encoder', 'onehot'), 'ワンホットエンコーディング'),
            'max_categories': categorical.get('max_categories', 20),
        },
        'exclude_columns': preprocessing_info.get('exclude_columns', []),
        'total_features': preprocessing_info.get('total_features', 0),
    }


def config_to_session_format(config):
    """
    前処理設定をセッション保存用の軽量フォーマットに変換
    
    Args:
        config: 前処理設定
        
    Returns:
        dict: 軽量化された設定
    """
    if config is None:
        config = get_default_preprocessing_config()
    
    return {
        'n_imp': config.get('numerical', {}).get('imputer', 'median'),
        'n_scl': config.get('numerical', {}).get('scaler', 'standard'),
        'c_imp': config.get('categorical', {}).get('imputer', 'missing'),
        'c_enc': config.get('categorical', {}).get('encoder', 'onehot'),
        'c_max': config.get('categorical', {}).get('max_categories', 20),
        'excl': config.get('exclude_columns', []),
    }


def session_format_to_config(session_config):
    """
    セッション保存フォーマットから前処理設定に変換
    
    Args:
        session_config: 軽量化された設定
        
    Returns:
        dict: 前処理設定
    """
    if session_config is None:
        return get_default_preprocessing_config()
    
    return {
        'numerical': {
            'imputer': session_config.get('n_imp', 'median'),
            'scaler': session_config.get('n_scl', 'standard'),
        },
        'categorical': {
            'imputer': session_config.get('c_imp', 'missing'),
            'encoder': session_config.get('c_enc', 'onehot'),
            'max_categories': session_config.get('c_max', 20),
        },
        'exclude_columns': session_config.get('excl', []),
    }
