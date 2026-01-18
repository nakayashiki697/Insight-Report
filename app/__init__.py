"""
Insight Report Application
Flaskアプリケーションの初期化
"""

from flask import Flask
from app.config import Config


def create_app(config_class=Config):
    """アプリケーションファクトリー"""
    # テンプレートと静的ファイルのパスを明示的に指定
    template_folder = str(config_class.BASE_DIR / 'templates')
    static_folder = str(config_class.BASE_DIR / 'static')
    
    app = Flask(
        __name__,
        template_folder=template_folder,
        static_folder=static_folder
    )
    app.config.from_object(config_class)
    
    # セッション設定
    app.config['SECRET_KEY'] = config_class.SECRET_KEY
    app.config['SESSION_PERMANENT'] = config_class.SESSION_PERMANENT
    app.config['PERMANENT_SESSION_LIFETIME'] = config_class.PERMANENT_SESSION_LIFETIME
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SECURE'] = config_class.SESSION_COOKIE_SECURE
    
    # 本番環境でSECRET_KEYが環境変数で設定されていない場合の警告
    if config_class.IS_PRODUCTION and not config_class._secret_key_env:
        import warnings
        warnings.warn(
            "警告: 本番環境でSECRET_KEYが環境変数で設定されていません。"
            "セッションが再起動のたびに無効になります。"
            "環境変数SECRET_KEYを設定してください。",
            UserWarning
        )
    
    # ルーティングの登録
    from app.routes import register_routes
    register_routes(app)
    
    # 認証ルートの登録
    from app.auth.routes import register_auth_routes
    register_auth_routes(app)
    
    return app

