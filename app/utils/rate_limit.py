"""
レート制限ユーティリティ
IPアドレスベースのシンプルなレート制限
"""

import time
from collections import defaultdict
from functools import wraps
from flask import request, jsonify


# IPアドレスごとのリクエスト記録
_rate_limit_store = defaultdict(list)


def rate_limit(max_requests: int = 10, window_seconds: int = 60):
    """
    レート制限デコレータ
    
    Args:
        max_requests: 時間窓内の最大リクエスト数
        window_seconds: 時間窓（秒）
    
    Usage:
        @app.route('/upload')
        @rate_limit(max_requests=5, window_seconds=60)
        def upload():
            ...
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # IPアドレスを取得
            ip_address = request.remote_addr or 'unknown'
            current_time = time.time()
            
            # 古いリクエスト記録を削除
            _rate_limit_store[ip_address] = [
                req_time for req_time in _rate_limit_store[ip_address]
                if current_time - req_time < window_seconds
            ]
            
            # リクエスト数をチェック
            if len(_rate_limit_store[ip_address]) >= max_requests:
                return jsonify({
                    'error': 'リクエストが多すぎます。しばらく待ってから再度お試しください。',
                    'retry_after': window_seconds
                }), 429
            
            # リクエストを記録
            _rate_limit_store[ip_address].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator
