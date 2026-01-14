"""
設定管理モジュール
"""

import os
from pathlib import Path


class Config:
    """アプリケーション設定"""
    
    # 基本設定
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # ファイル制限
    MAX_ROWS = 100000
    MAX_COLUMNS = 100
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'csv'}
    
    # パス設定
    BASE_DIR = Path(__file__).parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    OUTPUT_FOLDER = BASE_DIR / 'outputs'
    TEMP_FOLDER = BASE_DIR / 'temp'
    
    # ディレクトリの作成（存在しない場合）
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    TEMP_FOLDER.mkdir(exist_ok=True)
    
    # モデル設定
    RANDOM_STATE = 42
    CV_FOLDS = 3  # 交差検証の分割数（高速化のため5→3）
    N_ITER_SEARCH = 10  # RandomizedSearchCVの試行回数（高速化のため20→10）
    
    # 前処理設定
    MAX_CATEGORIES = 20  # カテゴリ数の上限（超過時は頻度上位N個のみ使用）
    
    # 前処理カスタマイズのデフォルト設定（Phase 7.2）
    DEFAULT_PREPROCESSING_CONFIG = {
        'numerical': {
            'imputer': 'median',      # median / mean / most_frequent
            'scaler': 'standard',     # standard / minmax / robust / none
        },
        'categorical': {
            'imputer': 'missing',     # missing / most_frequent
            'encoder': 'onehot',      # onehot / ordinal / frequency
            'max_categories': 20,     # 10〜100
        },
        'exclude_columns': [],        # 除外する列名のリスト
    }
    
    # 前処理オプションの定義（UI表示用）
    PREPROCESSING_OPTIONS = {
        'numerical_imputer': [
            {'value': 'median', 'label': '中央値（おすすめ）', 'description': '外れ値の影響を受けにくい'},
            {'value': 'mean', 'label': '平均値', 'description': 'きれいなデータ向け'},
            {'value': 'most_frequent', 'label': '最頻値', 'description': '特定の値に集中している場合'},
        ],
        'numerical_scaler': [
            {'value': 'standard', 'label': '標準化（おすすめ）', 'description': '平均0、標準偏差1に変換'},
            {'value': 'minmax', 'label': 'Min-Max正規化', 'description': '0〜1の範囲に変換'},
            {'value': 'robust', 'label': 'ロバスト標準化', 'description': '外れ値の影響を抑える'},
            {'value': 'none', 'label': 'なし', 'description': '決定木系モデルのみ使う場合'},
        ],
        'categorical_imputer': [
            {'value': 'missing', 'label': '"missing"として扱う（おすすめ）', 'description': '欠損自体に意味がある場合'},
            {'value': 'most_frequent', 'label': '最頻値で補完', 'description': '欠損が少ない場合'},
        ],
        'categorical_encoder': [
            {'value': 'onehot', 'label': 'ワンホット（おすすめ）', 'description': 'カテゴリごとに0/1の列を作成'},
            {'value': 'ordinal', 'label': '順序エンコード', 'description': '順序があるカテゴリ向け（低/中/高など）'},
            {'value': 'frequency', 'label': '頻度エンコード', 'description': 'カテゴリ数が多い場合（50以上）'},
        ],
    }
    
    # PDF設定
    PDF_TEMPLATE = BASE_DIR / 'templates' / 'report.html'
    PDF_STYLE = BASE_DIR / 'static' / 'css' / 'report.css'
    
    # セッション設定
    SESSION_PERMANENT = False
    PERMANENT_SESSION_LIFETIME = 3600  # 1時間

