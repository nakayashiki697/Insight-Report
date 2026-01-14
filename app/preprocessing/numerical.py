"""
数値特徴量処理モジュール
FR-020: 数値特徴量処理
FR-080: 数値データの前処理カスタマイズ（Phase 7.2）
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    数値特徴量の前処理トランスフォーマー
    
    処理内容:
    - 欠損値補完: median / mean / most_frequent（カスタマイズ可能）
    - スケーリング: standard / minmax / robust / none（カスタマイズ可能）
    
    Args:
        imputer_strategy: 欠損値補完方法 ('median', 'mean', 'most_frequent')
        scaler_type: スケーリング方法 ('standard', 'minmax', 'robust', 'none')
    """
    
    # 有効なオプション値
    VALID_IMPUTER_STRATEGIES = ['median', 'mean', 'most_frequent']
    VALID_SCALER_TYPES = ['standard', 'minmax', 'robust', 'none']
    
    def __init__(self, imputer_strategy='median', scaler_type='standard'):
        # バリデーション
        if imputer_strategy not in self.VALID_IMPUTER_STRATEGIES:
            raise ValueError(
                f"無効なimputer_strategy: '{imputer_strategy}'. "
                f"有効な値: {self.VALID_IMPUTER_STRATEGIES}"
            )
        if scaler_type not in self.VALID_SCALER_TYPES:
            raise ValueError(
                f"無効なscaler_type: '{scaler_type}'. "
                f"有効な値: {self.VALID_SCALER_TYPES}"
            )
        
        self.imputer_strategy = imputer_strategy
        self.scaler_type = scaler_type
        
        # インピュータの設定
        self.imputer = SimpleImputer(strategy=imputer_strategy)
        
        # スケーラーの設定
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:  # 'none'
            self.scaler = None
        
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        """
        トランスフォーマーを学習
        
        Args:
            X: 数値特徴量のDataFrame
            y: ターゲット（使用しない）
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_values = X.values
        else:
            X_values = X
        
        # 欠損値補完を学習
        self.imputer.fit(X_values)
        
        # スケーリングを学習（スケーラーがある場合のみ）
        if self.scaler is not None:
            X_imputed = self.imputer.transform(X_values)
            self.scaler.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        """
        データを変換
        
        Args:
            X: 数値特徴量のDataFrame
            
        Returns:
            numpy.ndarray: 変換後のデータ
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # 欠損値補完
        X_imputed = self.imputer.transform(X_values)
        
        # スケーリング（スケーラーがある場合のみ）
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
            return X_scaled
        
        return X_imputed
    
    def get_feature_names_out(self, input_features=None):
        """変換後の特徴量名を取得（元の列名を優先）"""
        # 入力特徴量名があればそれを使用
        if input_features is not None:
            return np.array(input_features)
        if self.feature_names_ is not None:
            return np.array(self.feature_names_)
        return None
    
    def get_params_description(self):
        """設定内容の説明を取得（ログ・レポート用）"""
        imputer_desc = {
            'median': '中央値',
            'mean': '平均値',
            'most_frequent': '最頻値',
        }
        scaler_desc = {
            'standard': '標準化（StandardScaler）',
            'minmax': 'Min-Max正規化',
            'robust': 'ロバスト標準化',
            'none': 'なし',
        }
        return {
            'imputer': imputer_desc.get(self.imputer_strategy, self.imputer_strategy),
            'scaler': scaler_desc.get(self.scaler_type, self.scaler_type),
        }
