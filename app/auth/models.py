"""
ユーザーモデル
JSONファイルベースのユーザー管理
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from app.config import Config
from app.auth.utils import hash_password


class User:
    """ユーザーモデル"""
    
    DATA_FILE = Config.BASE_DIR / 'data' / 'users.json'
    
    def __init__(self, user_id: str, username: str, email: str, password_hash: str, 
                 created_at: str, is_admin: bool = False):
        self.id = user_id
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.created_at = created_at
        self.is_admin = is_admin
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'created_at': self.created_at,
            'is_admin': self.is_admin
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """辞書からユーザーオブジェクトを作成"""
        return cls(
            user_id=data['id'],
            username=data['username'],
            email=data['email'],
            password_hash=data['password_hash'],
            created_at=data['created_at'],
            is_admin=data.get('is_admin', False)
        )
    
    @classmethod
    def _load_data(cls) -> Dict:
        """JSONファイルからデータを読み込む"""
        # データディレクトリが存在しない場合は作成
        cls.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if not cls.DATA_FILE.exists():
            return {'users': []}
        
        try:
            with open(cls.DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {'users': []}
    
    @classmethod
    def _save_data(cls, data: Dict):
        """JSONファイルにデータを保存"""
        cls.DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def find_by_username(cls, username: str) -> Optional['User']:
        """ユーザー名でユーザーを検索"""
        data = cls._load_data()
        for user_data in data.get('users', []):
            if user_data['username'] == username:
                return cls.from_dict(user_data)
        return None
    
    @classmethod
    def find_by_id(cls, user_id: str) -> Optional['User']:
        """IDでユーザーを検索"""
        data = cls._load_data()
        for user_data in data.get('users', []):
            if user_data['id'] == user_id:
                return cls.from_dict(user_data)
        return None
    
    @classmethod
    def find_by_email(cls, email: str) -> Optional['User']:
        """メールアドレスでユーザーを検索"""
        data = cls._load_data()
        for user_data in data.get('users', []):
            if user_data['email'] == email:
                return cls.from_dict(user_data)
        return None
    
    @classmethod
    def create(cls, username: str, email: str, password: str, is_admin: bool = False) -> 'User':
        """新しいユーザーを作成"""
        # 既存ユーザーのチェック
        if cls.find_by_username(username):
            raise ValueError('このユーザー名は既に使用されています')
        if cls.find_by_email(email):
            raise ValueError('このメールアドレスは既に使用されています')
        
        # 新しいユーザーを作成
        user = cls(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=hash_password(password),
            created_at=datetime.now().isoformat(),
            is_admin=is_admin
        )
        
        # データを保存
        data = cls._load_data()
        data['users'].append(user.to_dict())
        cls._save_data(data)
        
        return user
    
    def verify_password(self, password: str) -> bool:
        """パスワードを検証"""
        from app.auth.utils import verify_password as check_password
        return check_password(self.password_hash, password)
