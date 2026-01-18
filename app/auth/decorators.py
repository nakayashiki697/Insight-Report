"""
認証デコレータ
ログイン必須や管理者権限のチェック
"""

from functools import wraps
from flask import session, redirect, url_for, flash, request
from app.auth.models import User


def login_required(f):
    """
    ログイン必須デコレータ
    ログインしていない場合はログインページにリダイレクト
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('ログインが必要です', 'error')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def admin_required(f):
    """
    管理者権限必須デコレータ
    管理者でない場合はアクセス拒否
    """
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        user_id = session.get('user_id')
        if user_id:
            user = User.find_by_id(user_id)
            if not user or not user.is_admin:
                flash('管理者権限が必要です', 'error')
                return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
