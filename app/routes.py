"""
ルーティング定義
"""

import uuid
import pickle
import threading
from datetime import datetime
import numpy as np
from flask import render_template, jsonify, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from app.config import Config
from app.data.validator import validate_csv_file
from app.data.detector import detect_problem_type, get_column_info
from app.utils.progress import create_progress_tracker, get_progress_tracker
from app.utils.cleanup import maybe_cleanup, cleanup_old_files, get_storage_usage


def _clean_feature_names(feature_names):
    """
    scikit-learnの特徴量名をユーザーフレンドリーに整理
    
    変換例:
    - 'num__age' → 'age'
    - 'cat__gender_male' → 'gender=male'
    - 'cat__city_freq' → 'city'
    - 'remainder__id' → 'id'
    
    Args:
        feature_names: scikit-learnから取得した特徴量名のリスト
        
    Returns:
        list: 整理された特徴量名のリスト
    """
    cleaned = []
    for name in feature_names:
        # numpy配列の場合は文字列に変換
        name = str(name)
        
        # プレフィックス削除 (num__, cat__, remainder__)
        if name.startswith('num__'):
            name = name[5:]
        elif name.startswith('cat__'):
            name = name[5:]
        elif name.startswith('remainder__'):
            name = name[11:]
        
        # 頻度エンコーディングのサフィックス削除 (_freq)
        if name.endswith('_freq'):
            name = name[:-5]
        
        # OneHotエンコーディングの場合: column_value → column=value
        # (最後のアンダースコアをイコールに置換)
        # ただし、元々アンダースコアを含む列名もあるので注意
        # 簡易的に、最後の_をスペースに変換して読みやすくする
        # 例: gender_male → gender (male) または gender=male
        
        cleaned.append(name)
    
    return cleaned


def allowed_file(filename: str) -> bool:
    """許可されたファイル拡張子かチェック"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def register_routes(app):
    """ルーティングを登録"""
    
    # リクエストごとに10%の確率でクリーンアップを実行
    @app.before_request
    def auto_cleanup():
        """確率的に古いファイルをクリーンアップ"""
        # 静的ファイルへのリクエストはスキップ
        if request.path.startswith('/static'):
            return
        # 10%の確率で24時間以上経過したファイルを削除
        maybe_cleanup(probability=0.1, max_age_hours=24)
    
    @app.route('/health')
    def health_check():
        """ヘルスチェックエンドポイント"""
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
        }), 200
    
    @app.route('/api/storage')
    def storage_status():
        """ストレージ使用状況を返す（管理用）"""
        usage = get_storage_usage()
        return jsonify(usage), 200
    
    @app.route('/api/cleanup', methods=['POST'])
    def manual_cleanup():
        """手動クリーンアップを実行（管理用）"""
        max_age_hours = request.args.get('max_age_hours', 24, type=int)
        result = cleanup_old_files(max_age_hours)
        return jsonify(result), 200
    
    @app.route('/')
    def index():
        """トップページ"""
        return render_template('index.html')
    
    @app.route('/glossary')
    def glossary():
        """用語集ページ"""
        return render_template('glossary.html')
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        """CSVファイルアップロード（FR-001）"""
        if request.method == 'POST':
            # ファイルがリクエストに含まれているか確認
            if 'file' not in request.files:
                flash('ファイルが選択されていません', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            
            # ファイル名が空でないか確認
            if file.filename == '':
                flash('ファイルが選択されていません', 'error')
                return redirect(request.url)
            
            # ファイル拡張子を確認
            if not allowed_file(file.filename):
                flash('CSVファイルのみアップロード可能です', 'error')
                return redirect(request.url)
            
            # セキュアなファイル名を生成
            filename = secure_filename(file.filename)
            session_id = str(uuid.uuid4())
            file_path = Config.UPLOAD_FOLDER / f"{session_id}_{filename}"
            
            # ファイルを保存
            file.save(str(file_path))
            
            # ファイルを検証
            df, error_msg = validate_csv_file(file_path)
            
            if error_msg:
                # エラーの場合、ファイルを削除
                if file_path.exists():
                    file_path.unlink()
                flash(error_msg, 'error')
                return redirect(request.url)
            
            # セッションにデータを保存（JSONシリアライズ可能な型に変換）
            session['session_id'] = session_id
            session['file_path'] = str(file_path)
            session['filename'] = filename
            session['data_shape'] = {'rows': int(len(df)), 'columns': int(len(df.columns))}
            
            # 列情報を取得
            column_info = get_column_info(df)
            session['column_info'] = column_info
            
            # ターゲット列選択ページにリダイレクト
            return redirect(url_for('select_target'))
        
        return render_template('upload.html')
    
    @app.route('/select-target', methods=['GET', 'POST'])
    def select_target():
        """ターゲット列選択（FR-002）"""
        # セッションチェック
        if 'file_path' not in session:
            flash('ファイルがアップロードされていません', 'error')
            return redirect(url_for('upload'))
        
        file_path = session['file_path']
        column_info = session.get('column_info', {})
        
        if request.method == 'POST':
            target_column = request.form.get('target_column')
            
            if not target_column or target_column not in column_info:
                flash('有効なターゲット列を選択してください', 'error')
                return redirect(request.url)
            
            # CSVファイルを読み込み
            from app.data.loader import load_csv
            df = load_csv(file_path)
            
            # 問題種別を判定（FR-003）
            try:
                problem_type = detect_problem_type(df, target_column)
            except ValueError as e:
                flash(str(e), 'error')
                return redirect(request.url)
            
            # セッションに保存
            session['target_column'] = target_column
            session['problem_type'] = problem_type
            
            # 次のステップ（EDA実行）に進む
            return redirect(url_for('run_analysis'))
        
        # GETリクエスト時はproblem_typeはNone
        return render_template('select_target.html', column_info=column_info, problem_type=None)
    
    @app.route('/run-analysis')
    def run_analysis():
        """分析実行ページ（自動EDA実行）"""
        # セッションチェック
        if 'file_path' not in session or 'target_column' not in session:
            flash('データが準備されていません', 'error')
            return redirect(url_for('upload'))
        
        # セッション情報を取得
        file_path = session['file_path']
        target_column = session['target_column']
        problem_type = session.get('problem_type', 'unknown')
        data_shape = session.get('data_shape', {})
        
        # CSVファイルを読み込み
        from app.data.loader import load_csv
        df = load_csv(file_path)
        
        # EDA実行
        from app.eda import (
            calculate_basic_statistics,
            get_summary_statistics,
            analyze_missing_values,
            create_missing_heatmap,
            create_missing_barplot,
            create_all_distributions,
            analyze_correlations,
            generate_insights
        )
        
        # 出力ディレクトリの準備
        session_id = session.get('session_id', 'default')
        eda_output_dir = Config.TEMP_FOLDER / f'eda_{session_id}'
        eda_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 基本統計量
        basic_stats = calculate_basic_statistics(df)
        summary_stats = get_summary_statistics(df)
        
        # 2. 欠損値分析
        missing_info = analyze_missing_values(df)
        missing_heatmap_path = eda_output_dir / 'missing_heatmap.png'
        missing_barplot_path = eda_output_dir / 'missing_barplot.png'
        create_missing_heatmap(df, missing_heatmap_path)
        create_missing_barplot(df, missing_barplot_path)
        
        # 3. 特徴量分布
        distribution_plots = create_all_distributions(df, eda_output_dir / 'distributions', target_column)
        
        # 4. 相関分析
        correlation_results = analyze_correlations(df, eda_output_dir)
        
        # 5. 自動インサイト
        insights = generate_insights(df, target_column, problem_type)
        
        # セッションにEDA結果を保存（軽量化：必要なデータのみ保存）
        
        # correlation_resultsの軽量化（相関行列全体は除外、トップ5のみ保持）
        correlation_results_light = {
            'top_correlations': correlation_results.get('top_correlations', [])[:5],
            'heatmap_path': f'/temp/eda_{session_id}/correlation_heatmap.png' if correlation_results.get('heatmap_path') else None,
            'numeric_columns_count': int(correlation_results.get('numeric_columns_count', 0))
        }
        
        # summary_statisticsの軽量化（主要な統計のみ）
        summary_stats_light = {
            'total_rows': int(summary_stats.get('total_rows', 0)),
            'total_columns': int(summary_stats.get('total_columns', 0)),
            'numeric_columns': int(summary_stats.get('numeric_columns', 0)),
            'categorical_columns': int(summary_stats.get('categorical_columns', 0)),
            'missing_cells': int(summary_stats.get('missing_cells', 0)),
            'missing_ratio': float(summary_stats.get('missing_ratio', 0))
        }
        
        # basic_statisticsの軽量化（上位15列のみ、必要な統計のみ）
        basic_stats_light = {}
        for i, (col, stats) in enumerate(basic_stats.items()):
            if i >= 15:
                break
            basic_stats_light[col] = {
                'is_numeric': stats.get('is_numeric', False),
                'mean': round(stats.get('mean', 0), 2) if stats.get('mean') is not None else None,
                'median': round(stats.get('median', 0), 2) if stats.get('median') is not None else None,
                'std': round(stats.get('std', 0), 2) if stats.get('std') is not None else None,
                'null_count': int(stats.get('null_count', 0))
            }
        
        # missing_infoの軽量化（既にtop_missing_columnsがあるのでそれを使用）
        # analyze_missing_valuesの戻り値構造: {'column_info': {...}, 'top_missing_columns': [...], ...}
        missing_info_light = {
            'top_missing_columns': missing_info.get('top_missing_columns', [])[:10]
        }
        
        # insightsの軽量化（各カテゴリ上位3件まで）
        insights_light = {}
        for key, value in insights.items():
            if isinstance(value, list):
                insights_light[key] = value[:3]
            else:
                insights_light[key] = value
        
        session['eda_results'] = {
            'summary_statistics': summary_stats_light,
            'basic_statistics': basic_stats_light,
            'missing_info': missing_info_light,
            'missing_heatmap_path': f'/temp/eda_{session_id}/missing_heatmap.png',
            'correlation_results': correlation_results_light,
            'insights': insights_light,
            'eda_output_dir': f'eda_{session_id}'
        }
        
        print(f"[DEBUG] EDA results saved to session")
        
        # EDA結果表示ページにリダイレクト
        return redirect(url_for('eda_results'))
    
    @app.route('/eda-results')
    def eda_results():
        """EDA結果表示ページ"""
        # セッションチェック
        if 'eda_results' not in session:
            flash('EDA結果がありません', 'error')
            return redirect(url_for('upload'))
        
        eda_results = session['eda_results']
        target_column = session.get('target_column', '')
        problem_type = session.get('problem_type', 'unknown')
        data_shape = session.get('data_shape', {})
        
        return render_template(
            'eda_results.html',
            eda_results=eda_results,
            target_column=target_column,
            problem_type=problem_type,
            data_shape=data_shape
        )
    
    @app.route('/preprocessing-settings', methods=['GET', 'POST'])
    def preprocessing_settings():
        """前処理設定ページ（Phase 7.2）"""
        # セッションチェック
        if 'file_path' not in session or 'target_column' not in session:
            flash('データが準備されていません', 'error')
            return redirect(url_for('upload'))
        
        from app.preprocessing.pipeline import (
            get_default_preprocessing_config,
            validate_preprocessing_config,
            config_to_session_format,
            session_format_to_config
        )
        from app.data.loader import load_csv
        import numpy as np
        
        file_path = session['file_path']
        target_column = session['target_column']
        problem_type = session.get('problem_type', 'unknown')
        
        # データを読み込んで列情報を取得
        df = load_csv(file_path)
        feature_df = df.drop(columns=[target_column])
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()
        all_columns = feature_df.columns.tolist()
        
        if request.method == 'POST':
            use_default = request.form.get('use_default') == 'true'
            
            if use_default:
                # デフォルト設定を使用
                config = get_default_preprocessing_config()
            else:
                # フォームから設定を取得
                config = {
                    'numerical': {
                        'imputer': request.form.get('num_imputer', 'median'),
                        'scaler': request.form.get('num_scaler', 'standard'),
                    },
                    'categorical': {
                        'imputer': request.form.get('cat_imputer', 'missing'),
                        'encoder': request.form.get('cat_encoder', 'onehot'),
                        'max_categories': int(request.form.get('cat_max_categories', 20)),
                    },
                    'exclude_columns': request.form.getlist('exclude_columns'),
                }
            
            # バリデーション
            config = validate_preprocessing_config(config)
            
            # セッションに保存（軽量化フォーマット）
            session['preprocessing_config'] = config_to_session_format(config)
            session.modified = True
            
            # モデル学習ページへリダイレクト
            return redirect(url_for('train_models'))
        
        # GETリクエスト時
        # セッションから既存の設定を取得、なければデフォルト
        if 'preprocessing_config' in session:
            current_config = session_format_to_config(session['preprocessing_config'])
        else:
            current_config = get_default_preprocessing_config()
        
        return render_template(
            'preprocessing_settings.html',
            target_column=target_column,
            problem_type=problem_type,
            numeric_count=len(numeric_columns),
            categorical_count=len(categorical_columns),
            all_columns=all_columns,
            current_config=current_config,
            preprocessing_options=Config.PREPROCESSING_OPTIONS
        )
    
    @app.route('/train-models')
    def train_models():
        """前処理・モデル学習実行ページ（進捗表示付き）"""
        # セッションチェック
        if 'file_path' not in session or 'target_column' not in session:
            flash('データが準備されていません', 'error')
            return redirect(url_for('upload'))
        
        session_id = session.get('session_id', 'default')
        problem_type = session.get('problem_type', 'unknown')
        
        # 進捗トラッカーを作成
        progress_tracker = create_progress_tracker(session_id)
        
        # モデル数を取得して動的にステップを定義
        from app.models.classifiers import get_classification_models
        from app.models.regressors import get_regression_models
        
        if problem_type == 'classification':
            models = get_classification_models()
        else:
            models = get_regression_models()
        
        num_models = len(models)
        
        # ステップを定義（動的にモデル数を考慮）
        # 進捗率の合計が100%になるように調整
        preprocessing_pipeline_percentage = 5
        preprocessing_apply_percentage = 10
        best_model_selection_percentage = 5
        # 残り80%をモデル数で分割
        model_percentage = 80.0 / num_models if num_models > 0 else 0
        
        progress_tracker.add_step('前処理パイプライン構築中...', preprocessing_pipeline_percentage)
        progress_tracker.add_step('前処理適用中...', preprocessing_apply_percentage)
        
        # 各モデルの学習ステップを動的に追加
        for i, model_name in enumerate(models.keys(), 1):
            progress_tracker.add_step(f'{model_name}学習中...', model_percentage)
        
        progress_tracker.add_step('ベストモデル選択中...', best_model_selection_percentage)
        
        # バックグラウンドで処理を開始
        thread = threading.Thread(
            target=_train_models_background,
            args=(session_id,),
            daemon=True
        )
        thread.start()
        
        # 進捗表示ページを表示
        return render_template('train_models.html', session_id=session_id)
    
    def _train_models_background(session_id: str):
        """バックグラウンドでモデル学習を実行"""
        progress_tracker = get_progress_tracker(session_id)
        if not progress_tracker:
            return
        
        progress_tracker.start()
        
        try:
            # セッション情報を取得（新しいリクエストコンテキストが必要）
            from flask import has_request_context, current_app
            with current_app.app_context():
                # セッション情報を再取得する必要があるため、ファイルパス等を進捗トラッカーに保存
                # ここでは簡易的に、セッションIDからファイルを特定する方法を使用
                # 実際の実装では、セッション情報を別途保存する必要がある
                pass
            
            # セッション情報を取得（簡易版：グローバルストレージを使用）
            # 実際の実装では、セッション情報を進捗トラッカーに保存する必要がある
            from app.data.loader import load_csv
            from app.preprocessing.pipeline import build_preprocessing_pipeline, apply_preprocessing
            from app.preprocessing.logger import create_preprocessing_log
            from app.models.trainer import train_all_models
            
            # セッション情報を取得するためのヘルパー関数
            # 実際の実装では、セッション情報を別途保存する必要がある
            # ここでは簡易的に、セッションIDからファイルパスを取得する方法を使用
            
            # ファイルパスとターゲット列を進捗トラッカーに保存する必要がある
            # 簡易実装として、グローバルストレージを使用
            _session_data = getattr(_train_models_background, '_session_data', {})
            file_path = _session_data.get(session_id, {}).get('file_path')
            target_column = _session_data.get(session_id, {}).get('target_column')
            problem_type = _session_data.get(session_id, {}).get('problem_type', 'unknown')
            
            if not file_path or not target_column:
                progress_tracker.set_error('セッション情報が見つかりません')
                return
            
            # 1. 前処理パイプラインの構築
            progress_tracker.start_step(0)
            df = load_csv(file_path)
            preprocessor, feature_names, preprocessing_info = build_preprocessing_pipeline(df, target_column)
            preprocessing_log = create_preprocessing_log(preprocessing_info)
            progress_tracker.complete_step(0)
            
            # 2. 前処理の適用
            progress_tracker.start_step(1)
            X = df.drop(columns=[target_column])
            y = df[target_column].values
            X_transformed, _ = apply_preprocessing(preprocessor, X)
            progress_tracker.complete_step(1)
            
            # 3. モデル学習（各モデルごとに進捗を更新）
            from app.models.classifiers import get_classification_models
            from app.models.regressors import get_regression_models
            from app.models.trainer import train_all_models
            
            if problem_type == 'classification':
                models = get_classification_models()
            else:
                models = get_regression_models()
            
            model_names = list(models.keys())
            
            # 各モデルの学習進捗を更新（各モデルを個別に学習）
            from app.models.classifiers import train_classification_model
            from app.models.regressors import train_regression_model
            from app.models.selector import evaluate_classification_model, evaluate_regression_model, select_best_model
            from sklearn.model_selection import train_test_split
            
            # データを分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, random_state=Config.RANDOM_STATE,
                stratify=y if problem_type == 'classification' else None
            )
            
            model_results = {}
            model_comparison = []
            
            # 各モデルを個別に学習（進捗を個別に更新）
            for i, model_name in enumerate(model_names):
                if i + 2 < len(progress_tracker.steps):
                    progress_tracker.start_step(i + 2)
                    progress_tracker.steps[i + 2]['name'] = f'{model_name}学習中...'
                
                try:
                    if problem_type == 'classification':
                        result = train_classification_model(model_name, X_train, y_train, cv=Config.CV_FOLDS, n_iter=Config.N_ITER_SEARCH)
                        evaluation = evaluate_classification_model(result['model'], X_train, y_train, cv=Config.CV_FOLDS)
                    else:
                        result = train_regression_model(model_name, X_train, y_train, cv=Config.CV_FOLDS, n_iter=Config.N_ITER_SEARCH)
                        evaluation = evaluate_regression_model(result['model'], X_train, y_train, cv=Config.CV_FOLDS)
                    
                    model_results[model_name] = {
                        **result,
                        **evaluation,
                        'training_time': 0.0
                    }
                    
                    if problem_type == 'classification':
                        model_comparison.append({
                            'model_name': model_name,
                            'auc': evaluation.get('auc'),
                            'accuracy': evaluation.get('accuracy'),
                            'primary_score': evaluation.get('primary_score')
                        })
                    else:
                        model_comparison.append({
                            'model_name': model_name,
                            'rmse': evaluation.get('rmse'),
                            'primary_score': evaluation.get('primary_score')
                        })
                    
                    if i + 2 < len(progress_tracker.steps):
                        progress_tracker.complete_step(i + 2)
                        
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    if i + 2 < len(progress_tracker.steps):
                        progress_tracker.complete_step(i + 2)
                    continue
            
            # ベストモデル選択
            best_model_name, best_model_result = select_best_model(model_results, problem_type)
            
            training_results = {
                'model_results': model_results,
                'best_model_name': best_model_name,
                'best_model': best_model_result['model'],
                'model_comparison': model_comparison,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            # ベストモデル選択
            progress_tracker.start_step(len(progress_tracker.steps) - 1)
            progress_tracker.complete_step(len(progress_tracker.steps) - 1)
            
            # 結果を保存（セッションに保存する必要があるが、バックグラウンドスレッドからはアクセスできない）
            # 簡易実装として、結果を進捗トラッカーに保存
            progress_tracker._results = {
                'preprocessing_log': preprocessing_log,
                'model_results': {
                    'best_model_name': training_results['best_model_name'],
                    'model_comparison': training_results['model_comparison'],
                    'problem_type': problem_type
                },
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'best_model': training_results['best_model']
            }
            
            # モデルを一時保存
            model_dir = Config.TEMP_FOLDER / f'models_{session_id}'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'best_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': training_results['best_model'],
                    'preprocessor': preprocessor,
                    'feature_names': feature_names
                }, f)
            
            progress_tracker._model_path = str(model_path.relative_to(Config.BASE_DIR))
            progress_tracker.complete()
            
        except Exception as e:
            progress_tracker.set_error(str(e))
            import traceback
            traceback.print_exc()
    
    # セッションデータを保存するためのグローバルストレージ
    _session_data_storage = {}
    
    @app.route('/train-models-start', methods=['POST'])
    def train_models_start():
        """モデル学習を開始（POST）"""
        # セッションチェック
        if 'file_path' not in session or 'target_column' not in session:
            return jsonify({'error': 'データが準備されていません'}), 400
        
        session_id = session.get('session_id', 'default')
        
        # セッションデータを保存（前処理設定を含む）
        _session_data_storage[session_id] = {
            'file_path': session['file_path'],
            'target_column': session['target_column'],
            'problem_type': session.get('problem_type', 'unknown'),
            'preprocessing_config': session.get('preprocessing_config')  # Phase 7.2: 前処理設定
        }
        
        # 進捗トラッカーを作成
        progress_tracker = create_progress_tracker(session_id)
        
        # モデル数を取得して動的にステップを定義
        from app.models.classifiers import get_classification_models
        from app.models.regressors import get_regression_models
        
        problem_type = _session_data_storage.get(session_id, {}).get('problem_type', 'unknown')
        
        if problem_type == 'classification':
            models = get_classification_models()
        else:
            models = get_regression_models()
        
        num_models = len(models)
        
        # ステップを定義（動的にモデル数を考慮）
        # 進捗率の合計が100%になるように調整
        preprocessing_pipeline_percentage = 5
        preprocessing_apply_percentage = 10
        best_model_selection_percentage = 5
        # 残り80%をモデル数で分割
        model_percentage = 80.0 / num_models if num_models > 0 else 0
        
        progress_tracker.add_step('前処理パイプライン構築中...', preprocessing_pipeline_percentage)
        progress_tracker.add_step('前処理適用中...', preprocessing_apply_percentage)
        
        # 各モデルの学習ステップを動的に追加
        for i, model_name in enumerate(models.keys(), 1):
            progress_tracker.add_step(f'{model_name}学習中...', model_percentage)
        
        progress_tracker.add_step('ベストモデル選択中...', best_model_selection_percentage)
        
        # バックグラウンドで処理を開始
        thread = threading.Thread(
            target=_train_models_background_with_storage,
            args=(session_id,),
            daemon=True
        )
        thread.start()
        
        return jsonify({'status': 'started', 'session_id': session_id})
    
    def _train_models_background_with_storage(session_id: str):
        """バックグラウンドでモデル学習を実行（ストレージからセッションデータを取得）"""
        progress_tracker = get_progress_tracker(session_id)
        if not progress_tracker:
            return
        
        progress_tracker.start()
        
        try:
            # セッションデータを取得
            session_data = _session_data_storage.get(session_id)
            if not session_data:
                progress_tracker.set_error('セッション情報が見つかりません')
                return
            
            file_path = session_data['file_path']
            target_column = session_data['target_column']
            problem_type = session_data['problem_type']
            preprocessing_config_session = session_data.get('preprocessing_config')  # Phase 7.2
            
            from app.data.loader import load_csv
            from app.preprocessing.pipeline import build_preprocessing_pipeline, apply_preprocessing, session_format_to_config
            from app.preprocessing.logger import create_preprocessing_log
            from app.models.trainer import train_all_models
            
            # Phase 7.2: セッションフォーマットから設定を復元
            preprocessing_config = None
            if preprocessing_config_session:
                preprocessing_config = session_format_to_config(preprocessing_config_session)
            
            # 1. 前処理パイプラインの構築（カスタム設定を適用）
            progress_tracker.start_step(0)
            df = load_csv(file_path)
            preprocessor, feature_names, preprocessing_info = build_preprocessing_pipeline(
                df, target_column, config=preprocessing_config
            )
            preprocessing_log = create_preprocessing_log(preprocessing_info)
            progress_tracker.complete_step(0)
            
            # 2. 前処理の適用
            progress_tracker.start_step(1)
            # 除外列を処理（Phase 7.2）
            exclude_columns = preprocessing_info.get('exclude_columns', [])
            columns_to_drop = [target_column] + [col for col in exclude_columns if col in df.columns]
            X = df.drop(columns=columns_to_drop)
            y = df[target_column].values
            X_transformed, _ = apply_preprocessing(preprocessor, X)
            progress_tracker.complete_step(1)
            
            # 3. モデル学習
            from app.models.classifiers import get_classification_models
            from app.models.regressors import get_regression_models
            
            if problem_type == 'classification':
                models = get_classification_models()
            else:
                models = get_regression_models()
            
            model_names = list(models.keys())
            
            # 各モデルを個別に学習して進捗を更新
            from app.models.classifiers import train_classification_model
            from app.models.regressors import train_regression_model
            from app.models.selector import evaluate_classification_model, evaluate_regression_model, select_best_model
            from sklearn.model_selection import train_test_split
            
            # データを分割
            X_train, X_test, y_train, y_test = train_test_split(
                X_transformed, y, test_size=0.2, random_state=Config.RANDOM_STATE,
                stratify=y if problem_type == 'classification' else None
            )
            
            model_results = {}
            model_comparison = []
            
            # 各モデルを個別に学習（進捗を個別に更新）
            for i, model_name in enumerate(model_names):
                if i + 2 < len(progress_tracker.steps):
                    progress_tracker.start_step(i + 2)
                    progress_tracker.steps[i + 2]['name'] = f'{model_name}学習中...'
                
                try:
                    # モデル学習
                    if problem_type == 'classification':
                        result = train_classification_model(model_name, X_train, y_train, cv=Config.CV_FOLDS, n_iter=Config.N_ITER_SEARCH)
                        evaluation = evaluate_classification_model(result['model'], X_train, y_train, cv=Config.CV_FOLDS)
                        model_comparison.append({
                            'model_name': model_name,
                            'auc': evaluation.get('auc'),
                            'accuracy': evaluation.get('accuracy'),
                            'primary_score': evaluation.get('primary_score'),
                            'best_params': result.get('best_params', {})
                        })
                    else:
                        result = train_regression_model(model_name, X_train, y_train, cv=Config.CV_FOLDS, n_iter=Config.N_ITER_SEARCH)
                        evaluation = evaluate_regression_model(result['model'], X_train, y_train, cv=Config.CV_FOLDS)
                        model_comparison.append({
                            'model_name': model_name,
                            'rmse': evaluation.get('rmse'),
                            'primary_score': evaluation.get('primary_score'),
                            'best_params': result.get('best_params', {})
                        })
                    
                    model_results[model_name] = {
                        **result,
                        **evaluation
                    }
                    
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    if i + 2 < len(progress_tracker.steps):
                        progress_tracker.set_error(f'{model_name}の学習に失敗しました: {str(e)}')
                
                # ステップを完了
                if i + 2 < len(progress_tracker.steps):
                    progress_tracker.complete_step(i + 2)
            
            # ベストモデルを選択
            best_model_name, best_model_result = select_best_model(model_results, problem_type)
            
            training_results = {
                'model_results': model_results,
                'best_model_name': best_model_name,
                'best_model': best_model_result['model'],
                'model_comparison': model_comparison,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
            
            # ベストモデル選択
            progress_tracker.start_step(len(progress_tracker.steps) - 1)
            progress_tracker.complete_step(len(progress_tracker.steps) - 1)
            
            # 結果を保存
            progress_tracker._results = {
                'preprocessing_log': preprocessing_log,
                'model_results': {
                    'best_model_name': training_results['best_model_name'],
                    'model_comparison': training_results['model_comparison'],
                    'problem_type': problem_type
                },
                'preprocessor': preprocessor,
                'feature_names': feature_names,
                'best_model': training_results['best_model']
            }
            
            # モデルを一時保存
            model_dir = Config.TEMP_FOLDER / f'models_{session_id}'
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / 'best_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': training_results['best_model'],
                    'preprocessor': preprocessor,
                    'feature_names': feature_names
                }, f)
            
            progress_tracker._model_path = str(model_path.relative_to(Config.BASE_DIR))
            progress_tracker.complete()
            
            # セッションデータをクリーンアップ
            if session_id in _session_data_storage:
                del _session_data_storage[session_id]
            
        except Exception as e:
            progress_tracker.set_error(str(e))
            import traceback
            traceback.print_exc()
    
    @app.route('/api/progress')
    def get_progress():
        """進捗状況を取得するAPI"""
        session_id = request.args.get('session_id') or session.get('session_id')
        if not session_id:
            return jsonify({'error': 'セッションIDがありません'}), 400
        
        progress_tracker = get_progress_tracker(session_id)
        if not progress_tracker:
            return jsonify({'error': '進捗情報が見つかりません'}), 404
        
        progress_data = progress_tracker.to_dict()
        
        # 完了した場合、結果をセッションに保存
        if progress_tracker.status == 'completed' and hasattr(progress_tracker, '_results'):
            session['preprocessing_log'] = progress_tracker._results['preprocessing_log']
            
            # model_resultsを軽量化（best_paramsを削除、GradientBoostingClassifierを除外）
            model_results_raw = progress_tracker._results['model_results']
            model_comparison_light = []
            
            for model in model_results_raw.get('model_comparison', []):
                # GradientBoostingClassifierを除外
                if model.get('model_name') == 'GradientBoostingClassifier':
                    continue
                
                # 最小限の情報のみ保存
                model_light = {
                    'model_name': model.get('model_name'),
                    'auc': model.get('auc'),
                    'accuracy': model.get('accuracy'),
                    'rmse': model.get('rmse'),
                    # best_paramsは除外（セッションクッキーサイズ削減のため）
                }
                model_comparison_light.append(model_light)
            
            session['model_results'] = {
                'best_model_name': model_results_raw.get('best_model_name'),
                'model_comparison': model_comparison_light,
                'problem_type': model_results_raw.get('problem_type')
            }
            
            if hasattr(progress_tracker, '_model_path'):
                session['model_path'] = progress_tracker._model_path
                session['preprocessor_path'] = progress_tracker._model_path
        
        return jsonify(progress_data)
    
    @app.route('/model-comparison')
    def model_comparison():
        """モデル比較結果表示ページ"""
        # セッションチェック
        if 'model_results' not in session:
            flash('モデル学習結果がありません', 'error')
            return redirect(url_for('upload'))
        
        model_results = session['model_results']
        preprocessing_log = session.get('preprocessing_log', {})
        target_column = session.get('target_column', '')
        problem_type = session.get('problem_type', 'unknown')
        
        return render_template(
            'model_comparison.html',
            model_results=model_results,
            preprocessing_log=preprocessing_log,
            target_column=target_column,
            problem_type=problem_type
        )
    
    @app.route('/evaluate-model')
    def evaluate_model():
        """モデル評価実行ページ"""
        # セッションチェック
        if 'model_path' not in session or 'model_results' not in session:
            flash('モデルが学習されていません', 'error')
            return redirect(url_for('upload'))
        
        try:
            # セッション情報を取得
            session_id = session.get('session_id', 'default')
            model_path_str = session.get('model_path')
            if not model_path_str:
                flash('モデルパスがセッションに保存されていません', 'error')
                return redirect(url_for('model_comparison'))
            
            model_path = Config.BASE_DIR / model_path_str
            if not model_path.exists():
                flash(f'モデルファイルが見つかりません: {model_path}', 'error')
                return redirect(url_for('model_comparison'))
            
            problem_type = session.get('problem_type', 'unknown')
            preprocessing_log = session.get('preprocessing_log', {})
            
            # モデルと前処理器を読み込み
            import pickle
            print(f"[DEBUG] Loading model from: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data.get('model')
            preprocessor = model_data.get('preprocessor')
            
            if model is None or preprocessor is None:
                flash('モデルまたは前処理器の読み込みに失敗しました', 'error')
                return redirect(url_for('model_comparison'))
            
            print(f"[DEBUG] Model and preprocessor loaded successfully")
            
            # テストデータを準備
            from app.data.loader import load_csv
            file_path = session.get('file_path')
            target_column = session.get('target_column')
            
            if not file_path or not target_column:
                flash('ファイルパスまたはターゲット列がセッションに保存されていません', 'error')
                return redirect(url_for('model_comparison'))
            
            print(f"[DEBUG] Loading CSV from: {file_path}")
            df = load_csv(file_path)
            
            X = df.drop(columns=[target_column])
            y = df[target_column].values
            
            print(f"[DEBUG] Data shape: X={X.shape}, y={y.shape}")
            
            # テストデータに分割（train_all_modelsと同じ分割を使用）
            from sklearn.model_selection import train_test_split
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=Config.RANDOM_STATE,
                stratify=y if problem_type == 'classification' else None
            )
            
            print(f"[DEBUG] Test data shape: X_test={X_test_raw.shape}, y_test={y_test.shape}")
            
            # 前処理を適用（前処理器は既にfitされているのでtransformのみ）
            print(f"[DEBUG] Applying preprocessing...")
            X_test_transformed = preprocessor.transform(X_test_raw)
            print(f"[DEBUG] Transformed shape: {X_test_transformed.shape}")
            
            # 評価を実行
            from app.evaluation import (
                evaluate_classification,
                evaluate_regression,
                generate_evaluation_summary,
                generate_improvement_suggestions
            )
            
            output_dir = Config.TEMP_FOLDER / f'evaluation_{session_id}'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] Output directory: {output_dir}")
            
            if problem_type == 'classification':
                print(f"[DEBUG] Evaluating classification model...")
                evaluation_results = evaluate_classification(model, X_test_transformed, y_test, output_dir)
            else:
                print(f"[DEBUG] Evaluating regression model...")
                evaluation_results = evaluate_regression(model, X_test_transformed, y_test, output_dir)
            
            print(f"[DEBUG] Evaluation completed: {list(evaluation_results.keys())}")
            
            # 評価サマリを生成（短縮版）
            evaluation_summary = generate_evaluation_summary(evaluation_results, problem_type)
            # サマリテキストを短縮（セッションクッキーサイズ制限のため）
            summary_text = evaluation_summary.get('summary', '')
            if len(summary_text) > 100:
                summary_text = summary_text[:100] + '...'
            
            print(f"[DEBUG] Summary generated")
            
            # 改善提案はセッションに保存せず、評価結果ページで再生成する
            # これによりセッションクッキーのサイズを削減
            
            # セッションに保存（JSONシリアライズ可能な形式に変換）
            # 大きなデータ（confusion_matrix、classification_report、roc_curve、improvement_suggestionsなど）は除外
            # これらは必要に応じてファイルから読み込むか、再生成する
            # セッションクッキーのサイズ制限（4093バイト）を超えないように最小限のデータのみ保存
            # パスも短縮（ディレクトリ名のみ）、キー名も短縮
            output_dir_name = f'evaluation_{session_id}'
            
            # 問題タイプに応じて必要な指標のみ保存
            if problem_type == 'classification':
                session['evaluation_results'] = {
                    'a': round(evaluation_results.get('accuracy', 0), 3) if evaluation_results.get('accuracy') else None,
                    'p': round(evaluation_results.get('precision', 0), 3) if evaluation_results.get('precision') else None,
                    'r': round(evaluation_results.get('recall', 0), 3) if evaluation_results.get('recall') else None,
                    'f': round(evaluation_results.get('f1_score', 0), 3) if evaluation_results.get('f1_score') else None,
                    'auc': round(evaluation_results.get('roc_auc', 0), 3) if evaluation_results.get('roc_auc') else None,
                    'roc': 'roc_curve.png' if evaluation_results.get('roc_plot_path') else None,
                    'cm': 'confusion_matrix.png' if evaluation_results.get('cm_plot_path') else None,
                    'd': output_dir_name,
                    't': 'c'
                }
            else:
                session['evaluation_results'] = {
                    'rmse': round(evaluation_results.get('rmse', 0), 3) if evaluation_results.get('rmse') else None,
                    'mae': round(evaluation_results.get('mae', 0), 3) if evaluation_results.get('mae') else None,
                    'r2': round(evaluation_results.get('r2_score', 0), 3) if evaluation_results.get('r2_score') else None,
                    'scatter': 'scatter_plot.png' if evaluation_results.get('scatter_plot_path') else None,
                    'residual': 'residual_plot.png' if evaluation_results.get('residual_plot_path') else None,
                    'd': output_dir_name,
                    't': 'r'
                }
            session.modified = True  # セッションの変更を明示的にマーク
            print(f"[DEBUG] Evaluation results saved to session")
            
            # 評価結果表示ページにリダイレクト
            flash('モデル評価が完了しました', 'success')
            return redirect(url_for('evaluation_results'))
            
        except FileNotFoundError as e:
            error_msg = f'ファイルが見つかりません: {str(e)}'
            print(f"[ERROR] {error_msg}")
            flash(error_msg, 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('model_comparison'))
        except Exception as e:
            error_msg = f'モデル評価中にエラーが発生しました: {str(e)}'
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            flash(error_msg, 'error')
            import traceback
            traceback.print_exc()
            return redirect(url_for('model_comparison'))
    
    @app.route('/evaluation-results')
    def evaluation_results():
        """モデル評価結果表示ページ"""
        # セッションチェック
        if 'evaluation_results' not in session:
            # デバッグ情報を追加
            missing_keys = []
            if 'model_path' not in session:
                missing_keys.append('model_path')
            if 'model_results' not in session:
                missing_keys.append('model_results')
            if 'file_path' not in session:
                missing_keys.append('file_path')
            if 'target_column' not in session:
                missing_keys.append('target_column')
            
            if missing_keys:
                flash(f'評価結果がありません。不足しているセッション情報: {", ".join(missing_keys)}。モデル評価を実行してください。', 'error')
            else:
                flash('評価結果がありません。モデル評価を実行してください。', 'error')
            return redirect(url_for('model_comparison'))
        
        evaluation_results = session['evaluation_results']
        model_results = session.get('model_results', {})
        target_column = session.get('target_column', '')
        
        # 問題タイプを復元（短縮キーから）
        problem_type_code = evaluation_results.get('t', 'c')
        if problem_type_code == 'c':
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        preprocessing_log = session.get('preprocessing_log', {})
        
        # 改善提案を再生成（セッションクッキーサイズ制限のため）
        from app.evaluation import generate_improvement_suggestions, generate_evaluation_summary
        
        # 評価結果から完全なサマリを再生成（短縮キーから復元）
        problem_type_code = evaluation_results.get('t', 'c')
        if problem_type_code == 'c':
            problem_type = 'classification'
            full_evaluation_results = {
                'accuracy': evaluation_results.get('a'),
                'precision': evaluation_results.get('p'),
                'recall': evaluation_results.get('r'),
                'f1_score': evaluation_results.get('f'),
                'roc_auc': evaluation_results.get('auc'),
                'rmse': None,
                'mae': None,
                'r2_score': None
            }
        else:
            problem_type = 'regression'
            full_evaluation_results = {
                'accuracy': None,
                'precision': None,
                'recall': None,
                'f1_score': None,
                'roc_auc': None,
                'rmse': evaluation_results.get('rmse'),
                'mae': evaluation_results.get('mae'),
                'r2_score': evaluation_results.get('r2')
            }
        
        # 完全なサマリを再生成
        full_summary = generate_evaluation_summary(full_evaluation_results, problem_type)
        
        # 改善提案を生成
        improvement_suggestions = generate_improvement_suggestions(
            full_evaluation_results,
            problem_type,
            preprocessing_log
        )
        
        # パスを復元
        output_dir = evaluation_results.get('d', '')
        
        # 問題タイプに応じてプロットパスを設定
        if problem_type == 'classification':
            plot_paths = {
                'roc_plot_path': f'{output_dir}/{evaluation_results.get("roc")}' if evaluation_results.get('roc') else None,
                'cm_plot_path': f'{output_dir}/{evaluation_results.get("cm")}' if evaluation_results.get('cm') else None,
                'scatter_plot_path': None,
                'residual_plot_path': None
            }
        else:
            plot_paths = {
                'roc_plot_path': None,
                'cm_plot_path': None,
                'scatter_plot_path': f'{output_dir}/{evaluation_results.get("scatter")}' if evaluation_results.get('scatter') else None,
                'residual_plot_path': f'{output_dir}/{evaluation_results.get("residual")}' if evaluation_results.get('residual') else None
            }
        
        # 評価結果に改善提案と完全なサマリを追加
        evaluation_results_with_suggestions = {
            **full_evaluation_results,
            **plot_paths,
            'summary': full_summary,
            'improvement_suggestions': improvement_suggestions,
            'output_dir': output_dir
        }
        
        return render_template(
            'evaluation_results.html',
            evaluation_results=evaluation_results_with_suggestions,
            model_results=model_results,
            target_column=target_column,
            problem_type=problem_type
        )
    
    @app.route('/xai-analysis')
    def xai_analysis():
        """XAI分析実行ページ"""
        # セッションチェック
        if 'model_path' not in session or 'model_results' not in session:
            flash('モデルが学習されていません', 'error')
            return redirect(url_for('upload'))
        
        try:
            # セッション情報を取得
            session_id = session.get('session_id', 'default')
            model_path_str = session.get('model_path')
            if not model_path_str:
                flash('モデルパスがセッションに保存されていません', 'error')
                return redirect(url_for('evaluation_results'))
            
            model_path = Config.BASE_DIR / model_path_str
            if not model_path.exists():
                flash(f'モデルファイルが見つかりません: {model_path}', 'error')
                return redirect(url_for('evaluation_results'))
            
            problem_type = session.get('problem_type', 'unknown')
            
            # モデルと前処理器を読み込み
            import pickle
            print(f"[DEBUG] Loading model for XAI from: {model_path}")
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model = model_data.get('model')
            preprocessor = model_data.get('preprocessor')
            feature_names_raw = model_data.get('feature_names', [])
            
            if model is None or preprocessor is None:
                flash('モデルまたは前処理器の読み込みに失敗しました', 'error')
                return redirect(url_for('evaluation_results'))
            
            print(f"[DEBUG] Model and preprocessor loaded for XAI")
            
            # データを準備
            from app.data.loader import load_csv
            from sklearn.model_selection import train_test_split
            file_path = session.get('file_path')
            target_column = session.get('target_column')
            
            if not file_path or not target_column:
                flash('ファイルパスまたはターゲット列がセッションに保存されていません', 'error')
                return redirect(url_for('evaluation_results'))
            
            df = load_csv(file_path)
            X = df.drop(columns=[target_column])
            y = df[target_column].values
            
            # データを分割（訓練データとテストデータ）
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=Config.RANDOM_STATE,
                stratify=y if problem_type == 'classification' else None
            )
            
            # 前処理を適用
            X_train_transformed = preprocessor.transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            # 前処理後の特徴量名を取得
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names_raw = preprocessor.get_feature_names_out()
                    # 特徴量名を整理（プレフィックス削除、ユーザーフレンドリーに）
                    feature_names = _clean_feature_names(feature_names_raw)
                else:
                    # 前処理後の特徴量数を取得
                    n_features = X_train_transformed.shape[1]
                    feature_names = [f'feature_{i}' for i in range(n_features)]
                    print(f"[WARNING] get_feature_names_out not available, using generic names for {n_features} features")
            except Exception as e:
                # フォールバック: 前処理後の特徴量数から生成
                n_features = X_train_transformed.shape[1]
                feature_names = [f'feature_{i}' for i in range(n_features)]
                print(f"[WARNING] Failed to get feature names: {e}, using generic names for {n_features} features")
            
            print(f"[DEBUG] Feature names count: {len(feature_names)}, Transformed data shape: {X_train_transformed.shape}")
            
            # 出力ディレクトリを作成
            output_dir = Config.TEMP_FOLDER / f'xai_{session_id}'
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEBUG] XAI output directory: {output_dir}")
            
            # Permutation Importanceの計算
            from app.xai import calculate_permutation_importance
            permutation_results = calculate_permutation_importance(
                model,
                X_test_transformed,
                y_test,
                feature_names,
                problem_type,
                output_dir
            )
            
            print(f"[DEBUG] Permutation Importance completed")
            
            # PDPの生成
            from app.xai import generate_pdp_plots
            top_indices = permutation_results.get('top_indices', [])[:3]
            pdp_results = generate_pdp_plots(
                model,
                X_train_transformed,
                feature_names,
                top_indices,
                output_dir
            )
            
            print(f"[DEBUG] PDP plots generated")
            
            # XAI解説文の生成
            from app.xai import generate_xai_summary
            xai_summary = generate_xai_summary(
                permutation_results,
                pdp_results,
                problem_type
            )
            
            print(f"[DEBUG] XAI summary generated")
            
            # セッションに保存（全特徴量の重要度を含む）
            output_dir_name = f'xai_{session_id}'
            
            # 全特徴量の重要度を取得
            all_importances = permutation_results.get('importances_mean', [])
            all_features_full = feature_names  # 全特徴量名
            
            # 重要度でソート（降順）
            sorted_indices = np.argsort(all_importances)[::-1]
            sorted_features = [all_features_full[i] if i < len(all_features_full) else f'feature_{i}' for i in sorted_indices]
            sorted_importances = [round(all_importances[i], 4) for i in sorted_indices]
            
            # 特徴量名を短縮（長すぎる場合）
            sorted_features_short = [f[:30] + '...' if len(f) > 30 else f for f in sorted_features]
            
            # トップ5（互換性のため）
            top_features = sorted_features[:5]
            top_importances = sorted_importances[:5]
            top_features_short = sorted_features_short[:5]
            
            session['xai_results'] = {
                'f': top_features_short,  # トップ5（互換性のため）
                'i': top_importances,  # トップ5（互換性のため）
                'all_f': sorted_features_short,  # 全特徴量名（ソート済み）
                'all_i': sorted_importances,  # 全重要度（ソート済み）
                'p': 'permutation_importance.png',  # グラフ（使用しないが互換性のため）
                'n': len(pdp_results.get('plot_paths', [])),  # プロット数
                'idx': permutation_results.get('top_indices', [])[:3],  # PDPの特徴量インデックス
                'd': output_dir_name,  # 出力ディレクトリ
                't': problem_type[:1]  # 'c' or 'r'
            }
            session.modified = True
            print(f"[DEBUG] XAI results saved to session")
            
            # XAI結果表示ページにリダイレクト
            return redirect(url_for('xai_results'))
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'XAI分析中にエラーが発生しました: {str(e)}', 'error')
            return redirect(url_for('evaluation_results'))
    
    @app.route('/xai-results')
    def xai_results():
        """XAI結果表示ページ"""
        # セッションチェック
        if 'xai_results' not in session:
            flash('XAI分析結果がありません。説明可能性分析を実行してください。', 'error')
            return redirect(url_for('evaluation_results'))
        
        xai_results = session['xai_results']
        model_results = session.get('model_results', {})
        target_column = session.get('target_column', '')
        
        # 問題タイプを復元
        problem_type_code = xai_results.get('t', 'c')
        if problem_type_code == 'c':
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        # XAI解説文を再生成（セッションクッキーサイズ制限のため）
        from app.xai import generate_xai_summary
        
        # 全特徴量の重要度データを取得
        all_features = xai_results.get('all_f', [])
        all_importances = xai_results.get('all_i', [])
        
        # トップ3とトップ5（互換性のため）
        top_features = xai_results.get('f', [])
        top_importances = xai_results.get('i', [])
        
        # 全特徴量がない場合はトップ5から構築（後方互換性）
        if not all_features or not all_importances:
            all_features = top_features
            all_importances = top_importances
        
        # Permutation Importance結果を再構築（短縮キーから復元）
        permutation_results = {
            'top_features': top_features,
            'top_importances': top_importances
        }
        
        # PDP結果を再構築（プロットパスのみ）
        output_dir = xai_results.get('d', '')
        pdp_indices = xai_results.get('idx', [])
        pdp_count = xai_results.get('n', 0)
        
        # PDPプロットパスを生成
        pdp_plot_paths = []
        for idx in pdp_indices[:pdp_count]:
            pdp_plot_paths.append(f'{output_dir}/pdp_feature_{idx}.png')
        
        pdp_results = {
            'pdp_plots': [],  # トレンド情報は省略
            'plot_paths': pdp_plot_paths
        }
        
        # XAI解説文を生成
        xai_summary = generate_xai_summary(
            permutation_results,
            pdp_results,
            problem_type
        )
        
        # 完全なXAI結果を構築
        xai_results_complete = {
            'top_features': top_features[:3],  # トップ3のみ
            'top_importances': top_importances[:3],  # トップ3のみ
            'all_features': all_features,  # 全特徴量
            'all_importances': all_importances,  # 全重要度
            'permutation_plot': f'{output_dir}/{xai_results.get("p", "permutation_importance.png")}',
            'pdp_plots': pdp_plot_paths,
            'permutation_summary': xai_summary.get('permutation_summary', ''),
            'pdp_summary': xai_summary.get('pdp_summary', ''),
            'overall_summary': xai_summary.get('overall_summary', ''),
            'output_dir': output_dir
        }
        
        return render_template(
            'xai_results.html',
            xai_results=xai_results_complete,
            model_results=model_results,
            target_column=target_column,
            problem_type=problem_type
        )
    
    @app.route('/generate-report')
    def generate_report():
        """PDFレポート生成エンドポイント"""
        # セッションチェック
        if 'xai_results' not in session:
            flash('XAI分析結果がありません。説明可能性分析を実行してください。', 'error')
            return redirect(url_for('evaluation_results'))
        
        try:
            from app.report.generator import generate_pdf_report
            
            # セッションデータを収集
            session_id = session.get('session_id', 'default')
            model_results = session.get('model_results', {})
            # best_model または best_model_name を確認
            best_model = (session.get('best_model') or 
                         model_results.get('best_model_name', 'unknown'))
            
            session_data = {
                'session_id': session_id,
                'filename': session.get('filename', 'unknown.csv'),
                'target_column': session.get('target_column', 'unknown'),
                'problem_type': session.get('problem_type', 'unknown'),
                'data_shape': session.get('data_shape', {}),
                'eda_results': session.get('eda_results', {}),
                'preprocessing_log': session.get('preprocessing_log', {}),
                'model_results': model_results,
                'best_model': best_model,
                'evaluation_results': session.get('evaluation_results', {}),
                'xai_results': session.get('xai_results', {})
            }
            
            # PDF出力パス
            output_filename = f'report_{session_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
            output_path = Config.OUTPUT_FOLDER / output_filename
            
            # PDFを生成
            generate_pdf_report(session_data, output_path, Config.TEMP_FOLDER)
            
            # PDFファイルをダウンロード
            return send_from_directory(
                Config.OUTPUT_FOLDER,
                output_filename,
                as_attachment=True,
                download_name=f'xai_report_{session.get("filename", "report").replace(".csv", "")}.pdf'
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            flash(f'PDFレポート生成中にエラーが発生しました: {str(e)}', 'error')
            return redirect(url_for('xai_results'))
    
    @app.route('/api/columns')
    def get_columns():
        """列情報を取得するAPI"""
        if 'column_info' not in session:
            return jsonify({'error': 'ファイルがアップロードされていません'}), 400
        
        return jsonify({
            'columns': list(session['column_info'].keys()),
            'column_info': session['column_info']
        })
    
    @app.route('/temp/<path:filename>')
    def serve_temp_file(filename):
        """一時ファイルを提供"""
        return send_from_directory(str(Config.TEMP_FOLDER), filename)

