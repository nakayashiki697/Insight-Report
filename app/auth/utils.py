"""
認証ユーティリティ
パスワードのハッシュ化と検証
"""

from werkzeug.security import generate_password_hash, check_password_hash


def hash_password(password: str) -> str:
    """
    パスワードをハッシュ化
    
    Args:
        password: 平文のパスワード
        
    Returns:
        ハッシュ化されたパスワード
    """
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """
    パスワードを検証
    
    Args:
        password_hash: ハッシュ化されたパスワード
        password: 検証する平文のパスワード
        
    Returns:
        パスワードが一致する場合True
    """
    return check_password_hash(password_hash, password)
