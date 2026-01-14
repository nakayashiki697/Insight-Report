"""
カテゴリ特徴量処理モジュール
FR-021: カテゴリ特徴量処理
FR-081: カテゴリデータの前処理カスタマイズ（Phase 7.2）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from app.config import Config


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    頻度エンコーダー: カテゴリを出現頻度（割合）に変換
    
    カテゴリ数が多い場合に有効。
    各カテゴリを訓練データ内での出現頻度（0〜1）に置き換える。
    """
    
    def __init__(self):
        self.frequency_map_ = {}
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        頻度マップを学習
        
        Args:
            X: カテゴリ特徴量の配列
            y: ターゲット（使用しない）
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
            self.feature_names_ = [f'col_{i}' for i in range(X_values.shape[1])]
        
        # 各列ごとに頻度マップを作成
        for col_idx in range(X_values.shape[1]):
            col = X_values[:, col_idx]
            value_counts = pd.Series(col).value_counts(normalize=True)
            self.frequency_map_[col_idx] = value_counts.to_dict()
        
        return self
    
    def transform(self, X):
        """
        カテゴリを頻度に変換
        
        Args:
            X: カテゴリ特徴量の配列
            
        Returns:
            numpy.ndarray: 頻度に変換されたデータ
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        X_transformed = np.zeros(X_values.shape, dtype=float)
        
        for col_idx in range(X_values.shape[1]):
            freq_map = self.frequency_map_.get(col_idx, {})
            for row_idx, val in enumerate(X_values[:, col_idx]):
                # 未知のカテゴリは0.0（非常に稀）として扱う
                X_transformed[row_idx, col_idx] = freq_map.get(val, 0.0)
        
        return X_transformed
    
    def get_feature_names_out(self, input_features=None):
        """変換後の特徴量名を取得（元の列名をそのまま返す）"""
        if input_features is not None:
            return np.array(input_features)
        if self.feature_names_ is not None:
            return np.array(self.feature_names_)
        return None


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    カテゴリ特徴量の前処理トランスフォーマー
    
    処理内容:
    - 欠損値補完: "missing" / most_frequent（カスタマイズ可能）
    - 高カーディナリティ対策: カテゴリ数が多い場合、頻度上位N個のみ使用
    - エンコーディング: onehot / ordinal / frequency（カスタマイズ可能）
    
    Args:
        imputer_strategy: 欠損値補完方法 ('missing', 'most_frequent')
        encoder_type: エンコーディング方法 ('onehot', 'ordinal', 'frequency')
        max_categories: カテゴリ数の上限（超過時は頻度上位N個のみ使用）
    """
    
    # 有効なオプション値
    VALID_IMPUTER_STRATEGIES = ['missing', 'most_frequent']
    VALID_ENCODER_TYPES = ['onehot', 'ordinal', 'frequency']
    
    def __init__(self, imputer_strategy='missing', encoder_type='onehot', max_categories=None):
        # バリデーション
        if imputer_strategy not in self.VALID_IMPUTER_STRATEGIES:
            raise ValueError(
                f"無効なimputer_strategy: '{imputer_strategy}'. "
                f"有効な値: {self.VALID_IMPUTER_STRATEGIES}"
            )
        if encoder_type not in self.VALID_ENCODER_TYPES:
            raise ValueError(
                f"無効なencoder_type: '{encoder_type}'. "
                f"有効な値: {self.VALID_ENCODER_TYPES}"
            )
        
        self.imputer_strategy = imputer_strategy
        self.encoder_type = encoder_type
        self.max_categories = max_categories or Config.MAX_CATEGORIES
        
        # インピュータの設定
        if imputer_strategy == 'missing':
            self.imputer = SimpleImputer(strategy='constant', fill_value='missing')
        else:  # 'most_frequent'
            self.imputer = SimpleImputer(strategy='most_frequent')
        
        # エンコーダーの設定
        if encoder_type == 'onehot':
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        elif encoder_type == 'ordinal':
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:  # 'frequency'
            self.encoder = FrequencyEncoder()
        
        self.feature_names_ = None
        self.top_categories_ = {}  # 各列の頻度上位カテゴリを保存
    
    def fit(self, X, y=None):
        """
        トランスフォーマーを学習
        
        Args:
            X: カテゴリ特徴量のDataFrame
            y: ターゲット（使用しない）
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_df = X.copy()
        else:
            # numpy配列の場合はDataFrameに変換
            X_df = pd.DataFrame(X, columns=[f'col_{i}' for i in range(X.shape[1])])
            self.feature_names_ = X_df.columns.tolist()
        
        # 高カーディナリティ対策: 各列のカテゴリ数を確認し、上限を超える場合は頻度上位N個のみ使用
        X_processed = X_df.copy()
        
        for col in X_df.columns:
            # 欠損値を一時的に除外してカテゴリ数をカウント
            value_counts = X_df[col].value_counts()
            unique_count = len(value_counts)
            
            if unique_count > self.max_categories:
                # 頻度上位N個のカテゴリを取得
                top_categories = value_counts.head(self.max_categories).index.tolist()
                self.top_categories_[col] = set(top_categories)
                
                # 残りのカテゴリを「_other」に置き換え
                X_processed[col] = X_df[col].apply(
                    lambda x: x if pd.isna(x) or x in top_categories else '_other'
                )
                
                print(f"[INFO] Column '{col}' has {unique_count} categories, "
                      f"using top {self.max_categories} categories (others → '_other')")
            else:
                self.top_categories_[col] = None
        
        # 欠損値補完を学習
        X_values = X_processed.values
        X_imputed = self.imputer.fit_transform(X_values)
        
        # エンコーダーを学習
        self.encoder.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        """
        データを変換
        
        Args:
            X: カテゴリ特徴量のDataFrame
            
        Returns:
            numpy.ndarray: 変換後のデータ
        """
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            # numpy配列の場合はDataFrameに変換
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        # 高カーディナリティ対策: 学習時に設定した上位カテゴリ以外を「_other」に置き換え
        X_processed = X_df.copy()
        
        for col in X_df.columns:
            if col in self.top_categories_ and self.top_categories_[col] is not None:
                # 上位カテゴリ以外を「_other」に置き換え
                X_processed[col] = X_df[col].apply(
                    lambda x: x if pd.isna(x) or x in self.top_categories_[col] else '_other'
                )
        
        # 欠損値補完
        X_values = X_processed.values
        X_imputed = self.imputer.transform(X_values)
        
        # エンコーディング
        X_encoded = self.encoder.transform(X_imputed)
        
        return X_encoded
    
    def get_feature_names_out(self, input_features=None):
        """変換後の特徴量名を取得（元の列名を優先）"""
        # 入力特徴量名があればそれを使用
        base_names = input_features if input_features is not None else self.feature_names_
        
        if base_names is None:
            return None
        
        # OneHotエンコーダーの場合は展開された名前を取得
        if self.encoder_type == 'onehot' and hasattr(self.encoder, 'get_feature_names_out'):
            try:
                # OneHotEncoderは column_value 形式で返す
                return self.encoder.get_feature_names_out(base_names)
            except:
                pass
        
        # ordinal/frequency エンコーダーの場合は元の列名をそのまま返す
        return np.array(base_names)
    
    def get_params_description(self):
        """設定内容の説明を取得（ログ・レポート用）"""
        imputer_desc = {
            'missing': '"missing"として扱う',
            'most_frequent': '最頻値で補完',
        }
        encoder_desc = {
            'onehot': 'ワンホットエンコーディング',
            'ordinal': '順序エンコーディング',
            'frequency': '頻度エンコーディング',
        }
        return {
            'imputer': imputer_desc.get(self.imputer_strategy, self.imputer_strategy),
            'encoder': encoder_desc.get(self.encoder_type, self.encoder_type),
            'max_categories': self.max_categories,
        }
