"""
ファイル自動クリーンアップモジュール
アップロードされたファイルや一時ファイルを定期的に削除
"""

import os
import time
import random
from pathlib import Path
from typing import List
from app.config import Config


def cleanup_old_files(max_age_hours: int = 24) -> dict:
    """
    指定時間以上経過したファイルを削除
    
    Args:
        max_age_hours: 削除対象とするファイルの経過時間（時間単位）
        
    Returns:
        削除結果の辞書
    """
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    
    deleted_files: List[str] = []
    errors: List[str] = []
    
    # クリーンアップ対象のフォルダ
    folders = [
        Config.UPLOAD_FOLDER,
        Config.OUTPUT_FOLDER,
        Config.TEMP_FOLDER,
    ]
    
    for folder in folders:
        if not folder.exists():
            continue
            
        for file_path in folder.glob('*'):
            # .gitkeepは削除しない
            if file_path.name == '.gitkeep':
                continue
                
            # ディレクトリは再帰的に処理
            if file_path.is_dir():
                result = _cleanup_directory(file_path, now, max_age_seconds)
                deleted_files.extend(result['deleted'])
                errors.extend(result['errors'])
                continue
            
            # ファイルの経過時間をチェック
            try:
                file_age = now - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    deleted_files.append(str(file_path))
            except Exception as e:
                errors.append(f"{file_path}: {str(e)}")
    
    return {
        'deleted_count': len(deleted_files),
        'deleted_files': deleted_files,
        'error_count': len(errors),
        'errors': errors,
    }


def _cleanup_directory(dir_path: Path, now: float, max_age_seconds: float) -> dict:
    """
    ディレクトリ内のファイルをクリーンアップ
    空になったディレクトリも削除
    """
    deleted_files: List[str] = []
    errors: List[str] = []
    
    try:
        for file_path in dir_path.glob('**/*'):
            if file_path.is_file():
                try:
                    file_age = now - file_path.stat().st_mtime
                    if file_age > max_age_seconds:
                        file_path.unlink()
                        deleted_files.append(str(file_path))
                except Exception as e:
                    errors.append(f"{file_path}: {str(e)}")
        
        # 空のディレクトリを削除
        if dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            deleted_files.append(str(dir_path))
            
    except Exception as e:
        errors.append(f"{dir_path}: {str(e)}")
    
    return {'deleted': deleted_files, 'errors': errors}


def maybe_cleanup(probability: float = 0.1, max_age_hours: int = 24) -> dict | None:
    """
    確率的にクリーンアップを実行（リクエストごとに呼び出し用）
    
    Args:
        probability: クリーンアップを実行する確率（0.0〜1.0）
        max_age_hours: 削除対象とするファイルの経過時間
        
    Returns:
        クリーンアップが実行された場合は結果、実行されなかった場合はNone
    """
    if random.random() < probability:
        return cleanup_old_files(max_age_hours)
    return None


def get_storage_usage() -> dict:
    """
    ストレージ使用状況を取得
    
    Returns:
        各フォルダのファイル数とサイズ
    """
    folders = {
        'uploads': Config.UPLOAD_FOLDER,
        'outputs': Config.OUTPUT_FOLDER,
        'temp': Config.TEMP_FOLDER,
    }
    
    usage = {}
    total_size = 0
    total_files = 0
    
    for name, folder in folders.items():
        if not folder.exists():
            usage[name] = {'files': 0, 'size_mb': 0}
            continue
            
        files = list(folder.glob('**/*'))
        files = [f for f in files if f.is_file() and f.name != '.gitkeep']
        
        size = sum(f.stat().st_size for f in files)
        size_mb = round(size / (1024 * 1024), 2)
        
        usage[name] = {
            'files': len(files),
            'size_mb': size_mb,
        }
        
        total_size += size
        total_files += len(files)
    
    usage['total'] = {
        'files': total_files,
        'size_mb': round(total_size / (1024 * 1024), 2),
    }
    
    return usage

