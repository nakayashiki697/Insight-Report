"""
認証ルート
ログイン・登録・ログアウト機能
"""

from flask import render_template, request, redirect, url_for, flash, session
from app.auth.models import User


def register_auth_routes(app):
    """認証ルートを登録"""
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """ログインページ"""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '')
            next_url = request.args.get('next') or url_for('index')
            
            if not username or not password:
                flash('ユーザー名とパスワードを入力してください', 'error')
                return render_template('login.html')
            
            # ユーザーを検索
            user = User.find_by_username(username)
            if not user:
                # セキュリティのため、具体的なエラー内容は表示しない
                flash('ユーザー名またはパスワードが正しくありません', 'error')
                return render_template('login.html')
            
            # パスワードを検証
            if not user.verify_password(password):
                flash('ユーザー名またはパスワードが正しくありません', 'error')
                return render_template('login.html')
            
            # セッションにユーザー情報を保存
            session['user_id'] = user.id
            session['username'] = user.username
            session['is_admin'] = user.is_admin
            
            flash(f'ようこそ、{user.username}さん', 'success')
            return redirect(next_url)
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """ユーザー登録ページ"""
        if request.method == 'POST':
            username = request.form.get('username', '').strip()
            email = request.form.get('email', '').strip()
            password = request.form.get('password', '')
            password_confirm = request.form.get('password_confirm', '')
            
            # バリデーション
            if not username or not email or not password:
                flash('すべての項目を入力してください', 'error')
                return render_template('register.html')
            
            if password != password_confirm:
                flash('パスワードが一致しません', 'error')
                return render_template('register.html')
            
            if len(password) < 6:
                flash('パスワードは6文字以上で入力してください', 'error')
                return render_template('register.html')
            
            # ユーザーを作成
            try:
                user = User.create(username=username, email=email, password=password)
                flash('登録が完了しました。ログインしてください', 'success')
                return redirect(url_for('login'))
            except ValueError as e:
                flash(str(e), 'error')
                return render_template('register.html')
        
        return render_template('register.html')
    
    @app.route('/logout')
    def logout():
        """ログアウト"""
        username = session.get('username', 'ユーザー')
        session.clear()
        flash(f'{username}さん、ログアウトしました', 'info')
        return redirect(url_for('index'))
