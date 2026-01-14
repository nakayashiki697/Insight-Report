"""
データ検証モジュール
FR-001: CSVアップロード時の検証

非エンジニア向けにわかりやすいエラーメッセージを提供
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from app.config import Config


class ValidationError(Exception):
    """検証エラー"""
    pass


# エラーメッセージの定義（非エンジニア向け）
ERROR_MESSAGES: Dict[str, Dict[str, str]] = {
    "file_too_large": {
        "title": "ファイルが大きすぎます",
        "message": "ファイルサイズが{current_size}MBありますが、最大{max_size}MBまでしか対応していません。",
        "solution": "データの一部を抽出して、小さいファイルを作成してください。"
    },
    "too_many_rows": {
        "title": "データの行数が多すぎます",
        "message": "現在{current:,}行ありますが、最大{max:,}行までしか対応していません。",
        "solution": "必要なデータだけを抽出して、行数を減らしてください。"
    },
    "too_many_columns": {
        "title": "列の数が多すぎます",
        "message": "現在{current}列ありますが、最大{max}列までしか対応していません。",
        "solution": "分析に必要な列だけを残して、不要な列を削除してください。"
    },
    "empty_data": {
        "title": "データが空です",
        "message": "ファイルにデータが入っていないようです。",
        "solution": "ファイルにデータが正しく保存されているか確認してください。"
    },
    "no_columns": {
        "title": "列が見つかりません",
        "message": "CSVファイルに列（ヘッダー）が見つかりません。",
        "solution": "CSVファイルの1行目に列名が入っているか確認してください。"
    },
    "parse_error": {
        "title": "ファイルを読み込めません",
        "message": "CSVファイルの形式に問題があるようです。",
        "solution": "ファイルがCSV形式（カンマ区切り）で保存されているか確認してください。Excelで「CSV UTF-8」形式で保存し直すと解決することがあります。"
    },
    "encoding_error": {
        "title": "文字コードの問題",
        "message": "ファイルの文字コードを正しく読み取れませんでした。",
        "solution": "Excelで開いて「CSV UTF-8」形式で保存し直してください。"
    },
    "duplicate_columns": {
        "title": "列名が重複しています",
        "message": "同じ名前の列が複数あります: {columns}",
        "solution": "列名が重複しないように、異なる名前をつけてください。"
    },
    "unknown_error": {
        "title": "予期しないエラー",
        "message": "処理中にエラーが発生しました。",
        "solution": "ファイルの形式を確認するか、別のデータで試してください。エラー詳細: {detail}"
    }
}


def format_error_message(error_type: str, **kwargs) -> str:
    """
    エラーメッセージを整形する
    
    Args:
        error_type: エラーの種類
        **kwargs: メッセージに埋め込む値
        
    Returns:
        str: 整形されたエラーメッセージ
    """
    error_info = ERROR_MESSAGES.get(error_type, ERROR_MESSAGES["unknown_error"])
    
    title = error_info["title"]
    message = error_info["message"].format(**kwargs) if kwargs else error_info["message"]
    solution = error_info["solution"].format(**kwargs) if kwargs else error_info["solution"]
    
    return f"【{title}】\n{message}\n\n💡 解決方法: {solution}"


def validate_file_size(file_path: str | Path) -> None:
    """
    ファイルサイズを検証
    
    Args:
        file_path: ファイルパス
        
    Raises:
        ValidationError: ファイルサイズが制限を超えている場合
    """
    file_path = Path(file_path)
    file_size = file_path.stat().st_size
    
    if file_size > Config.MAX_FILE_SIZE:
        raise ValidationError(
            format_error_message(
                "file_too_large",
                current_size=f"{file_size / (1024 * 1024):.1f}",
                max_size=f"{Config.MAX_FILE_SIZE / (1024 * 1024):.0f}"
            )
        )


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    DataFrameの行数・列数を検証
    
    Args:
        df: 検証するDataFrame
        
    Returns:
        Tuple[bool, Optional[str]]: (検証成功, エラーメッセージ)
    """
    num_rows, num_cols = df.shape
    
    # データが空かチェック
    if num_rows == 0:
        return False, format_error_message("empty_data")
    
    if num_cols == 0:
        return False, format_error_message("no_columns")
    
    # 行数チェック
    if num_rows > Config.MAX_ROWS:
        return False, format_error_message(
            "too_many_rows",
            current=num_rows,
            max=Config.MAX_ROWS
        )
    
    # 列数チェック
    if num_cols > Config.MAX_COLUMNS:
        return False, format_error_message(
            "too_many_columns",
            current=num_cols,
            max=Config.MAX_COLUMNS
        )
    
    # 列名の重複チェック
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        return False, format_error_message(
            "duplicate_columns",
            columns=", ".join(duplicate_cols[:5]) + ("..." if len(duplicate_cols) > 5 else "")
        )
    
    return True, None


def validate_csv_file(file_path: str | Path) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    CSVファイルを検証して読み込む
    
    Args:
        file_path: CSVファイルのパス
        
    Returns:
        Tuple[pd.DataFrame, Optional[str]]: (DataFrame, エラーメッセージ)
    """
    try:
        # ファイルサイズの検証
        validate_file_size(file_path)
        
        # CSVファイルの読み込み
        from app.data.loader import load_csv
        df = load_csv(file_path)
        
        # DataFrameの検証
        is_valid, error_msg = validate_dataframe(df)
        
        if not is_valid:
            return None, error_msg
        
        return df, None
    
    except ValidationError as e:
        return None, str(e)
    except pd.errors.ParserError as e:
        return None, format_error_message("parse_error")
    except UnicodeDecodeError as e:
        return None, format_error_message("encoding_error")
    except ValueError as e:
        error_str = str(e).lower()
        if "encoding" in error_str or "decode" in error_str:
            return None, format_error_message("encoding_error")
        return None, format_error_message("parse_error")
    except Exception as e:
        return None, format_error_message("unknown_error", detail=str(e)[:100])

