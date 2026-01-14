"""
CSVファイル読み込みモジュール
FR-001: CSVアップロード
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from app.config import Config


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """
    CSVファイルを読み込む
    
    Args:
        file_path: CSVファイルのパス
        
    Returns:
        pandas.DataFrame: 読み込んだデータ
        
    Raises:
        ValueError: ファイルが存在しない、またはCSV形式でない場合
        pd.errors.EmptyDataError: ファイルが空の場合
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValueError(f"ファイルが見つかりません: {file_path}")
    
    if not file_path.suffix.lower() == '.csv':
        raise ValueError(f"CSVファイルではありません: {file_path}")
    
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(file_path)
        
        if df.empty:
            raise ValueError("CSVファイルが空です")
        
        return df
    
    except pd.errors.EmptyDataError:
        raise ValueError("CSVファイルが空です")
    except pd.errors.ParserError as e:
        raise ValueError(f"CSVファイルの解析に失敗しました: {str(e)}")
    except Exception as e:
        raise ValueError(f"ファイルの読み込みに失敗しました: {str(e)}")

