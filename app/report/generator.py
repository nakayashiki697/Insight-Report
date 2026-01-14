"""
PDFレポート生成モジュール
"""

import base64
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from io import BytesIO

# WeasyPrintのインポート（エラーハンドリング付き）
try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_ERROR = str(e)

# ReportLabのインポート（フォールバック用）
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import platform
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_pdf_report(
    session_data: Dict[str, Any],
    output_path: Path,
    temp_dir: Path
) -> Path:
    """
    PDFレポートを生成する
    
    Args:
        session_data: セッションデータ（EDA、前処理、モデル、評価、重要度分析結果を含む）
        output_path: PDF出力パス
        temp_dir: 一時ファイルディレクトリ（画像ファイルのパス解決用）
    
    Returns:
        生成されたPDFファイルのパス
    
    Raises:
        RuntimeError: PDF生成ライブラリが利用できない場合
    """
    # WeasyPrintが利用可能な場合は使用
    if WEASYPRINT_AVAILABLE:
        return _generate_with_weasyprint(session_data, output_path, temp_dir)
    
    # ReportLabが利用可能な場合はフォールバック
    if REPORTLAB_AVAILABLE:
        return _generate_with_reportlab(session_data, output_path, temp_dir)
    
    # どちらも利用できない場合
    error_msg = (
        "PDF生成ライブラリが利用できません。\n"
        f"WeasyPrintエラー: {WEASYPRINT_ERROR if not WEASYPRINT_AVAILABLE else 'N/A'}\n\n"
        "解決方法:\n"
        "1. ReportLabをインストール: uv add reportlab\n"
        "2. または、WeasyPrint用にGTK+ランタイムをインストール（詳細は docs/phase6_windows_setup.md を参照）"
    )
    raise RuntimeError(error_msg)


def _generate_with_weasyprint(
    session_data: Dict[str, Any],
    output_path: Path,
    temp_dir: Path
) -> Path:
    """WeasyPrintを使用してPDFを生成"""
    # HTMLテンプレートにデータを埋め込む
    html_content = _build_html_report(session_data, temp_dir)
    
    # CSSを読み込む
    css_path = Path(__file__).parent.parent.parent / 'static' / 'css' / 'report.css'
    css_content = None
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = CSS(string=f.read())
    
    # PDFを生成
    font_config = FontConfiguration()
    html_doc = HTML(string=html_content)
    
    if css_content:
        html_doc.write_pdf(output_path, stylesheets=[css_content], font_config=font_config)
    else:
        html_doc.write_pdf(output_path, font_config=font_config)
    
    return output_path


def _register_japanese_fonts():
    """日本語フォントを登録"""
    if platform.system() == 'Windows':
        fonts_dir = Path(r'C:\Windows\Fonts')
        
        # 日本語フォントの候補（優先順位順）
        # TTCファイルを優先（Windows標準の日本語フォント）
        ttc_candidates = [
            ('msgothic.ttc', 'Gothic', 0),  # MSゴシック
            ('msgothic.ttc', 'Gothic', 1),
            ('msmincho.ttc', 'Mincho', 0),  # MS明朝
            ('msmincho.ttc', 'Mincho', 1),
            ('meiryo.ttc', 'Meiryo', 0),   # メイリオ
            ('meiryo.ttc', 'Meiryo', 1),
            ('BIZ-UDGothicR.ttc', 'Gothic', 0),  # BIZ UDゴシック
            ('BIZ-UDMinchoM.ttc', 'Mincho', 0),  # BIZ UD明朝
        ]
        
        # TTCファイルを試す
        for font_filename, font_name, subfont_idx in ttc_candidates:
            font_files = list(fonts_dir.glob(font_filename))
            if not font_files:
                font_files = list(fonts_dir.glob(font_filename.upper()))
            if not font_files:
                font_files = list(fonts_dir.glob(font_filename.lower()))
            
            if font_files:
                font_path = font_files[0]
                try:
                    font_reg_name = f'Japanese{font_name}'
                    # TTCファイルの場合は、フォントインデックスを指定
                    pdfmetrics.registerFont(TTFont(font_reg_name, str(font_path), subfontIndex=subfont_idx))
                    test_font = pdfmetrics.getFont(font_reg_name)
                    if test_font is not None:
                        return font_reg_name
                except Exception:
                    continue
        
        # TTFファイルも試す（フォールバック）
        ttf_candidates = [
            ('msgothic.ttf', 'Gothic', None),
            ('msmincho.ttf', 'Mincho', None),
            ('meiryo.ttf', 'Meiryo', None),
            ('yumin.ttf', 'YuMin', None),
        ]
        
        for font_filename, font_name, subfont_idx in ttf_candidates:
            font_files = list(fonts_dir.glob(font_filename))
            if not font_files:
                font_files = list(fonts_dir.glob(font_filename.upper()))
            if not font_files:
                font_files = list(fonts_dir.glob(font_filename.lower()))
            if not font_files:
                font_files = [f for f in fonts_dir.iterdir() if f.name.upper() == font_filename.upper()]
            
            if font_files:
                font_path = font_files[0]
                try:
                    font_reg_name = f'Japanese{font_name}'
                    # TTFファイルの場合は、subfontIndexを指定しない
                    pdfmetrics.registerFont(TTFont(font_reg_name, str(font_path)))
                    test_font = pdfmetrics.getFont(font_reg_name)
                    if test_font is not None:
                        return font_reg_name
                except Exception:
                    continue
        
        # フォントが見つからない場合はHelveticaを使用（日本語は表示されない）
        return 'Helvetica'
    else:
        # Linux/Mac環境では、Notoフォントなどを使用
        return 'Helvetica'


def _generate_with_reportlab(
    session_data: Dict[str, Any],
    output_path: Path,
    temp_dir: Path
) -> Path:
    """ReportLabを使用してPDFを生成"""
    import sys
    # 日本語フォントを登録
    japanese_font = _register_japanese_fonts()
    
    # フォントが正しく登録されているか確認
    if japanese_font == 'Helvetica':
        print("[PDF生成] 警告: 日本語フォントが登録されていません。日本語文字が正しく表示されない可能性があります。", file=sys.stderr)
    else:
        # 登録されたフォントを確認
        registered_font = pdfmetrics.getFont(japanese_font)
        if registered_font is None:
            print(f"[PDF生成] エラー: フォント '{japanese_font}' が登録されていません。", file=sys.stderr)
            japanese_font = 'Helvetica'  # フォールバック
        else:
            print(f"[PDF生成] 使用フォント: {japanese_font}", file=sys.stderr)
    
    # PDFドキュメントを作成
    doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                           rightMargin=2*cm, leftMargin=2*cm,
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # スタイルを取得
    styles = getSampleStyleSheet()
    
    # カスタムスタイルを定義（日本語フォントを使用）
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=japanese_font,
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName=japanese_font,
        fontSize=18,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=15,
    )
    
    # 通常テキスト用のスタイルも日本語フォントを使用
    normal_style = ParagraphStyle(
        'JapaneseNormal',
        parent=styles['Normal'],
        fontName=japanese_font,
        fontSize=11,
    )
    
    # 見出し3用のスタイル
    heading3_style = ParagraphStyle(
        'JapaneseHeading3',
        parent=styles['Heading3'],
        fontName=japanese_font,
        fontSize=14,
    )
    
    # 見出し4用のスタイル
    heading4_style = ParagraphStyle(
        'JapaneseHeading4',
        parent=styles['Heading4'],
        fontName=japanese_font,
        fontSize=12,
    )
    
    # コンテンツを構築
    story = []
    
    # 表紙
    story.extend(_build_cover_with_reportlab(session_data, title_style, normal_style))
    story.append(PageBreak())
    
    # 各セクションを追加
    story.extend(_build_data_overview_with_reportlab(session_data, heading_style, normal_style, japanese_font))
    story.append(PageBreak())
    
    story.extend(_build_eda_with_reportlab(session_data, heading_style, normal_style, heading3_style, temp_dir))
    story.append(PageBreak())
    
    story.extend(_build_preprocessing_with_reportlab(session_data, heading_style, normal_style, japanese_font))
    story.append(PageBreak())
    
    story.extend(_build_model_comparison_with_reportlab(session_data, heading_style, normal_style, japanese_font))
    story.append(PageBreak())
    
    story.extend(_build_evaluation_with_reportlab(session_data, heading_style, normal_style, heading3_style, temp_dir, japanese_font))
    story.append(PageBreak())
    
    story.extend(_build_xai_with_reportlab(session_data, heading_style, normal_style, heading3_style, heading4_style, temp_dir, japanese_font))
    story.append(PageBreak())
    
    story.extend(_build_conclusion_with_reportlab(session_data, heading_style, normal_style, heading3_style))
    story.append(PageBreak())
    
    story.extend(_build_appendix_with_reportlab(session_data, heading_style, normal_style, heading3_style))
    
    # PDFを生成
    doc.build(story)
    
    return output_path


def _build_html_report(session_data: Dict[str, Any], temp_dir: Path) -> str:
    """
    HTMLレポートを構築する
    
    Args:
        session_data: セッションデータ
        temp_dir: 一時ファイルディレクトリ
    
    Returns:
        HTMLコンテンツ
    """
    # 基本情報を取得
    filename = session_data.get('filename', 'unknown.csv')
    target_column = session_data.get('target_column', 'unknown')
    problem_type = session_data.get('problem_type', 'unknown')
    data_shape = session_data.get('data_shape', {})
    # best_model または best_model_name を確認
    best_model = (session_data.get('best_model') or 
                 session_data.get('model_results', {}).get('best_model_name', 'unknown'))
    
    # 各セクションのHTMLを生成
    cover_html = _generate_cover_section(filename, target_column, problem_type, data_shape)
    data_overview_html = _generate_data_overview_section(data_shape, target_column, problem_type)
    eda_html = _generate_eda_section(session_data.get('eda_results', {}), temp_dir)
    preprocessing_html = _generate_preprocessing_section(session_data.get('preprocessing_log', {}))
    model_comparison_html = _generate_model_comparison_section(
        session_data.get('model_results', {}),
        best_model
    )
    evaluation_html = _generate_evaluation_section(
        session_data.get('evaluation_results', {}),
        problem_type,
        temp_dir
    )
    xai_html = _generate_xai_section(
        session_data.get('xai_results', {}),
        problem_type,
        temp_dir
    )
    conclusion_html = _generate_conclusion_section(
        session_data.get('evaluation_results', {}),
        session_data.get('xai_results', {}),
        problem_type
    )
    appendix_html = _generate_appendix_section(
        session_data.get('model_results', {}),
        best_model
    )
    
    # HTML全体を組み立て
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>Insight Report - {filename}</title>
</head>
<body>
    {cover_html}
    <div style="page-break-before: always;"></div>
    {data_overview_html}
    <div style="page-break-before: always;"></div>
    {eda_html}
    <div style="page-break-before: always;"></div>
    {preprocessing_html}
    <div style="page-break-before: always;"></div>
    {model_comparison_html}
    <div style="page-break-before: always;"></div>
    {evaluation_html}
    <div style="page-break-before: always;"></div>
    {xai_html}
    <div style="page-break-before: always;"></div>
    {conclusion_html}
    <div style="page-break-before: always;"></div>
    {appendix_html}
</body>
</html>
"""
    return html


def _generate_cover_section(filename: str, target_column: str, problem_type: str, data_shape: Dict) -> str:
    """表紙セクションを生成"""
    problem_type_jp = '分類' if problem_type == 'classification' else '回帰'
    rows = data_shape.get('rows', 0)
    columns = data_shape.get('columns', 0)
    
    return f"""
    <div style="text-align: center; padding-top: 200px;">
        <h1 style="font-size: 48px; color: #667eea; margin-bottom: 30px;">Insight Report</h1>
        <h2 style="font-size: 32px; color: #764ba2; margin-bottom: 50px;">分析レポート</h2>
        <div style="font-size: 18px; line-height: 2;">
            <p><strong>データファイル:</strong> {filename}</p>
            <p><strong>ターゲット列:</strong> {target_column}</p>
            <p><strong>問題種別:</strong> {problem_type_jp}</p>
            <p><strong>データサイズ:</strong> {rows:,}行 × {columns}列</p>
            <p style="margin-top: 50px;"><strong>生成日時:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        </div>
    </div>
"""


def _generate_data_overview_section(data_shape: Dict, target_column: str, problem_type: str) -> str:
    """データ概要セクションを生成"""
    rows = data_shape.get('rows', 0)
    columns = data_shape.get('columns', 0)
    problem_type_jp = '分類' if problem_type == 'classification' else '回帰'
    
    return f"""
    <h1>1. データ概要</h1>
    <table>
        <tr>
            <th>項目</th>
            <th>値</th>
        </tr>
        <tr>
            <td>データ行数</td>
            <td>{rows:,}</td>
        </tr>
        <tr>
            <td>特徴量数</td>
            <td>{columns - 1}</td>
        </tr>
        <tr>
            <td>ターゲット列</td>
            <td>{target_column}</td>
        </tr>
        <tr>
            <td>問題種別</td>
            <td>{problem_type_jp}</td>
        </tr>
    </table>
"""


def _generate_eda_section(eda_results: Dict, temp_dir: Path) -> str:
    """EDAセクションを生成"""
    html = "<h1>2. 探索的データ分析（EDA）</h1>"
    
    # 基本統計量
    if 'basic_statistics' in eda_results:
        html += "<h2>2.1 基本統計量</h2>"
        html += "<p>データの基本統計量を以下に示します。</p>"
        # 詳細な統計量は簡略化して表示
    
    # 欠損値
    if 'missing_info' in eda_results:
        missing_info = eda_results['missing_info']
        html += "<h2>2.2 欠損値分析</h2>"
        if missing_info.get('total_missing_count', 0) > 0:
            html += f"<p>欠損値の総数: {missing_info.get('total_missing_count', 0):,}</p>"
            html += f"<p>欠損値の割合: {missing_info.get('missing_percentage', 0):.2f}%</p>"
        else:
            html += "<p>欠損値はありません。</p>"
    
    # 相関分析
    if 'correlation_results' in eda_results:
        corr_results = eda_results['correlation_results']
        html += "<h2>2.3 相関分析</h2>"
        if corr_results.get('top_correlations'):
            html += "<p>強い相関を持つ特徴量ペア（上位5組）:</p>"
            html += "<ul>"
            for corr in corr_results['top_correlations'][:5]:
                col1 = corr.get('column1', corr.get('feature1', ''))
                col2 = corr.get('column2', corr.get('feature2', ''))
                corr_value = corr.get('correlation', 0)
                html += f"<li>{col1} - {col2}: {corr_value:.3f}</li>"
            html += "</ul>"
    
    # インサイト
    if 'insights' in eda_results:
        insights = eda_results['insights']
        html += "<h2>2.4 自動インサイト</h2>"
        if insights.get('summary'):
            html += f"<p>{insights['summary']}</p>"
    
    return html


def _generate_preprocessing_section(preprocessing_log: Dict) -> str:
    """前処理セクションを生成"""
    html = "<h1>3. 前処理内容</h1>"
    
    if not preprocessing_log:
        html += "<p>前処理ログがありません。</p>"
        return html
    
    config = preprocessing_log.get('config_summary', {})
    num_cfg = config.get('numerical', {})
    cat_cfg = config.get('categorical', {})
    numeric_count = preprocessing_log.get('numeric_count', len(preprocessing_log.get('numeric_columns', [])))
    categorical_count = preprocessing_log.get('categorical_count', len(preprocessing_log.get('categorical_columns', [])))
    exclude_columns = preprocessing_log.get('exclude_columns', [])
    
    html += "<table>"
    html += "<tr><th>処理項目</th><th>内容</th></tr>"
    html += f"<tr><td>数値特徴量</td><td>{numeric_count}列</td></tr>"
    html += f"<tr><td>欠損値補完（数値）</td><td>{num_cfg.get('imputer_label', '中央値で補完')}</td></tr>"
    html += f"<tr><td>スケーリング（数値）</td><td>{num_cfg.get('scaler_label', '標準化')}</td></tr>"
    html += f"<tr><td>カテゴリ特徴量</td><td>{categorical_count}列</td></tr>"
    default_cat_imputer = '"missing"で補完'
    html += f"<tr><td>欠損値補完（カテゴリ）</td><td>{cat_cfg.get('imputer_label', default_cat_imputer)}</td></tr>"
    html += f"<tr><td>エンコーディング</td><td>{cat_cfg.get('encoder_label', 'ワンホットエンコーディング')}</td></tr>"
    if 'max_categories' in cat_cfg:
        html += f"<tr><td>カテゴリ上限</td><td>{cat_cfg.get('max_categories')}</td></tr>"
    html += f"<tr><td>除外列</td><td>{', '.join(exclude_columns) if exclude_columns else 'なし'}</td></tr>"
    html += "</table>"
    
    return html


def _generate_model_comparison_section(model_results: Dict, best_model: str) -> str:
    """モデル比較セクションを生成"""
    html = "<h1>4. モデル比較</h1>"
    
    # model_comparison または comparison を確認
    comparison = None
    if model_results:
        comparison = model_results.get('model_comparison') or model_results.get('comparison')
    
    if not comparison:
        html += "<p>モデル比較結果がありません。</p>"
        return html
    
    html += "<table>"
    html += "<tr><th>モデル名</th><th>スコア</th><th>状態</th></tr>"
    
    for model_data in comparison:
        model_name = model_data.get('model_name', 'unknown')
        # primary_score, auc, accuracy, rmse のいずれかを使用
        score = (model_data.get('primary_score') or 
                model_data.get('auc') or 
                model_data.get('accuracy') or 
                (1.0 / (1.0 + model_data.get('rmse', 1.0)) if model_data.get('rmse') else 0))
        is_best = model_name == best_model
        
        html += "<tr>"
        html += f"<td>{model_name}</td>"
        html += f"<td>{score:.4f}</td>"
        html += f"<td>{'✓ 採用' if is_best else ''}</td>"
        html += "</tr>"
    
    html += "</table>"
    html += f"<p><strong>採用モデル:</strong> {best_model}</p>"
    
    return html


def _generate_evaluation_section(evaluation_results: Dict, problem_type: str, temp_dir: Path) -> str:
    """評価セクションを生成"""
    html = "<h1>5. モデル評価</h1>"
    
    if not evaluation_results:
        html += "<p>評価結果がありません。</p>"
        return html
    
    output_dir_name = evaluation_results.get('d', '')
    output_dir = temp_dir / output_dir_name
    
    if problem_type == 'classification':
        html += "<h2>5.1 分類評価指標</h2>"
        html += "<table>"
        html += "<tr><th>指標</th><th>値</th></tr>"
        
        if evaluation_results.get('a') is not None:
            html += f"<tr><td>Accuracy</td><td>{evaluation_results['a']:.3f}</td></tr>"
        if evaluation_results.get('p') is not None:
            html += f"<tr><td>Precision</td><td>{evaluation_results['p']:.3f}</td></tr>"
        if evaluation_results.get('r') is not None:
            html += f"<tr><td>Recall</td><td>{evaluation_results['r']:.3f}</td></tr>"
        if evaluation_results.get('f') is not None:
            html += f"<tr><td>F1-score</td><td>{evaluation_results['f']:.3f}</td></tr>"
        if evaluation_results.get('auc') is not None:
            html += f"<tr><td>AUC</td><td>{evaluation_results['auc']:.3f}</td></tr>"
        
        html += "</table>"
        
        # ROC曲線
        if evaluation_results.get('roc'):
            roc_path = output_dir / evaluation_results['roc']
            if roc_path.exists():
                img_base64 = _image_to_base64(roc_path)
                html += f'<h2>5.2 ROC曲線</h2><img src="data:image/png;base64,{img_base64}" style="max-width: 100%;" />'
        
        # 混同行列
        if evaluation_results.get('cm'):
            cm_path = output_dir / evaluation_results['cm']
            if cm_path.exists():
                img_base64 = _image_to_base64(cm_path)
                html += f'<h2>5.3 混同行列</h2><img src="data:image/png;base64,{img_base64}" style="max-width: 100%;" />'
    
    else:  # regression
        html += "<h2>5.1 回帰評価指標</h2>"
        html += "<table>"
        html += "<tr><th>指標</th><th>値</th></tr>"
        
        if evaluation_results.get('rmse') is not None:
            html += f"<tr><td>RMSE</td><td>{evaluation_results['rmse']:.3f}</td></tr>"
        if evaluation_results.get('mae') is not None:
            html += f"<tr><td>MAE</td><td>{evaluation_results['mae']:.3f}</td></tr>"
        if evaluation_results.get('r2') is not None:
            html += f"<tr><td>R²</td><td>{evaluation_results['r2']:.3f}</td></tr>"
        
        html += "</table>"
        
        # 散布図
        if evaluation_results.get('scatter'):
            scatter_path = output_dir / evaluation_results['scatter']
            if scatter_path.exists():
                img_base64 = _image_to_base64(scatter_path)
                html += f'<h2>5.2 予測 vs 実測</h2><img src="data:image/png;base64,{img_base64}" style="max-width: 100%;" />'
        
        # 残差プロット
        if evaluation_results.get('residual'):
            residual_path = output_dir / evaluation_results['residual']
            if residual_path.exists():
                img_base64 = _image_to_base64(residual_path)
                html += f'<h2>5.3 残差プロット</h2><img src="data:image/png;base64,{img_base64}" style="max-width: 100%;" />'
    
    return html


def _generate_xai_section(xai_results: Dict, problem_type: str, temp_dir: Path) -> str:
    """重要度分析セクションを生成"""
    html = "<h1>6. 重要度分析</h1>"
    
    if not xai_results:
        html += "<p>重要度分析結果がありません。</p>"
        return html
    
    # Permutation Importance（テーブル表示）
    html += "<h2>6.1 特徴量重要度（Permutation Importance）</h2>"
    all_features = xai_results.get('all_f') or xai_results.get('f', [])
    all_importances = xai_results.get('all_i') or xai_results.get('i', [])
    
    if all_features and all_importances:
        max_imp = all_importances[0] if all_importances else 1
        html += "<table>"
        html += "<tr><th>順位</th><th>特徴量</th><th>重要度</th><th>相対重要度</th></tr>"
        for idx, (feature, importance) in enumerate(zip(all_features, all_importances)):
            percentage = (importance / max_imp * 100) if max_imp else 0
            html += f"<tr><td>{idx + 1}</td><td>{feature}</td><td>{importance:.4f}</td><td>{percentage:.1f}%</td></tr>"
        html += "</table>"
    else:
        html += "<p>重要度データがありません。</p>"
    
    # PDPはUI非表示に合わせて省略
    html += "<p style=\"margin-top: 12px; color: #555;\">※ 本レポートではPDPグラフは省略しています。</p>"
    
    return html


def _generate_conclusion_section(evaluation_results: Dict, xai_results: Dict, problem_type: str) -> str:
    """結論セクションを生成"""
    html = "<h1>7. 結論と次のアクション提案</h1>"
    
    html += "<h2>7.1 分析結果のまとめ</h2>"
    
    if problem_type == 'classification':
        if evaluation_results.get('auc'):
            auc = evaluation_results['auc']
            html += f"<p>採用モデルのAUCは {auc:.3f} です。"
            if auc >= 0.9:
                html += "非常に高い性能を示しています。"
            elif auc >= 0.8:
                html += "良好な性能を示しています。"
            elif auc >= 0.7:
                html += "中程度の性能を示しています。"
            else:
                html += "改善の余地があります。"
            html += "</p>"
    else:
        if evaluation_results.get('r2'):
            r2 = evaluation_results['r2']
            html += f"<p>採用モデルのR²は {r2:.3f} です。"
            if r2 >= 0.9:
                html += "非常に高い説明力を持っています。"
            elif r2 >= 0.7:
                html += "良好な説明力を持っています。"
            elif r2 >= 0.5:
                html += "中程度の説明力を持っています。"
            else:
                html += "改善の余地があります。"
            html += "</p>"
    
    # 重要特徴量
    all_features = xai_results.get('all_f') or xai_results.get('f', [])
    if all_features:
        html += "<h2>7.2 重要な特徴量</h2>"
        html += "<p>Permutation Importance分析により、以下の特徴量が重要であることが判明しました:</p>"
        html += "<ul>"
        for feature in all_features[:3]:
            html += f"<li>{feature}</li>"
        html += "</ul>"
    
    html += "<h2>7.3 次のアクション提案</h2>"
    html += "<ul>"
    html += "<li>より多くのデータを収集してモデルの性能向上を図る</li>"
    html += "<li>重要な特徴量を活用した特徴量エンジニアリングを検討する</li>"
    html += "<li>ハイパーパラメータのさらなる調整を検討する</li>"
    html += "<li>他のモデルアルゴリズムの試行を検討する</li>"
    html += "</ul>"
    
    return html


def _generate_appendix_section(model_results: Dict, best_model: str) -> str:
    """付録セクションを生成"""
    html = "<h1>8. 付録</h1>"
    
    html += "<h2>8.1 モデル設定値</h2>"
    html += f"<p>採用モデル: {best_model}</p>"
    html += "<p>詳細なハイパーパラメータはモデルファイルに保存されています。</p>"
    
    html += "<h2>8.2 技術情報</h2>"
    html += "<ul>"
    html += "<li>フレームワーク: scikit-learn</li>"
    html += "<li>前処理: StandardScaler, OneHotEncoder</li>"
    html += "<li>重要度分析手法: Permutation Importance, Partial Dependence Plot</li>"
    html += "</ul>"
    
    return html


def _image_to_base64(image_path: Path) -> str:
    """画像ファイルをBase64エンコードする"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
    except Exception:
        return ""


# ============================================================================
# ReportLab用の関数（WeasyPrintが利用できない場合のフォールバック）
# ============================================================================

def _build_cover_with_reportlab(session_data: Dict, title_style, normal_style):
    """表紙セクションをReportLabで構築"""
    story = []
    filename = session_data.get('filename', 'unknown.csv')
    target_column = session_data.get('target_column', 'unknown')
    problem_type = session_data.get('problem_type', 'unknown')
    data_shape = session_data.get('data_shape', {})
    problem_type_jp = '分類' if problem_type == 'classification' else '回帰'
    rows = data_shape.get('rows', 0)
    columns = data_shape.get('columns', 0)
    
    story.append(Spacer(1, 8*cm))
    story.append(Paragraph("Insight Report", title_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("分析レポート", title_style))
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph(f"データファイル: {filename}", normal_style))
    story.append(Paragraph(f"ターゲット列: {target_column}", normal_style))
    story.append(Paragraph(f"問題種別: {problem_type_jp}", normal_style))
    story.append(Paragraph(f"データサイズ: {rows:,}行 × {columns}列", normal_style))
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}", normal_style))
    
    return story


def _build_data_overview_with_reportlab(session_data: Dict, heading_style, normal_style, japanese_font):
    """データ概要セクションをReportLabで構築"""
    story = []
    data_shape = session_data.get('data_shape', {})
    target_column = session_data.get('target_column', 'unknown')
    problem_type = session_data.get('problem_type', 'unknown')
    problem_type_jp = '分類' if problem_type == 'classification' else '回帰'
    rows = data_shape.get('rows', 0)
    columns = data_shape.get('columns', 0)
    
    story.append(Paragraph("1. データ概要", heading_style))
    
    data = [
        ['項目', '値'],
        ['データ行数', f'{rows:,}'],
        ['特徴量数', f'{columns - 1}'],
        ['ターゲット列', target_column],
        ['問題種別', problem_type_jp]
    ]
    
    table = Table(data, colWidths=[6*cm, 10*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), japanese_font),
        ('FONTNAME', (0, 0), (-1, 0), japanese_font),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    return story


def _build_eda_with_reportlab(session_data: Dict, heading_style, normal_style, heading3_style, temp_dir: Path):
    """EDAセクションをReportLabで構築"""
    story = []
    eda_results = session_data.get('eda_results', {})
    
    story.append(Paragraph("2. 探索的データ分析（EDA）", heading_style))
    
    if 'missing_info' in eda_results:
        missing_info = eda_results['missing_info']
        story.append(Paragraph("2.2 欠損値分析", heading3_style))
        if missing_info.get('total_missing_count', 0) > 0:
            story.append(Paragraph(f"欠損値の総数: {missing_info.get('total_missing_count', 0):,}", normal_style))
            story.append(Paragraph(f"欠損値の割合: {missing_info.get('missing_percentage', 0):.2f}%", normal_style))
        else:
            story.append(Paragraph("欠損値はありません。", normal_style))
    
    if 'correlation_results' in eda_results:
        corr_results = eda_results['correlation_results']
        story.append(Paragraph("2.3 相関分析", heading3_style))
        if corr_results.get('top_correlations'):
            story.append(Paragraph("強い相関を持つ特徴量ペア（上位5組）:", normal_style))
            for corr in corr_results['top_correlations'][:5]:
                col1 = corr.get('column1', corr.get('feature1', ''))
                col2 = corr.get('column2', corr.get('feature2', ''))
                corr_value = corr.get('correlation', 0)
                story.append(Paragraph(
                    f"• {col1} - {col2}: {corr_value:.3f}",
                    normal_style
                ))
    
    return story


def _build_preprocessing_with_reportlab(session_data: Dict, heading_style, normal_style, japanese_font):
    """前処理セクションをReportLabで構築"""
    story = []
    preprocessing_log = session_data.get('preprocessing_log', {})
    
    story.append(Paragraph("3. 前処理内容", heading_style))
    
    if preprocessing_log:
        config = preprocessing_log.get('config_summary', {})
        num_cfg = config.get('numerical', {})
        cat_cfg = config.get('categorical', {})
        numeric_count = preprocessing_log.get('numeric_count', len(preprocessing_log.get('numeric_columns', [])))
        categorical_count = preprocessing_log.get('categorical_count', len(preprocessing_log.get('categorical_columns', [])))
        exclude_columns = preprocessing_log.get('exclude_columns', [])
        
        data = [
            ['処理項目', '内容'],
            ['数値特徴量', f'{numeric_count}列'],
            ['欠損値補完（数値）', num_cfg.get('imputer_label', '中央値で補完')],
            ['スケーリング（数値）', num_cfg.get('scaler_label', '標準化')],
            ['カテゴリ特徴量', f'{categorical_count}列'],
            ['欠損値補完（カテゴリ）', cat_cfg.get('imputer_label', '"missing"で補完')],
            ['エンコーディング', cat_cfg.get('encoder_label', 'ワンホットエンコーディング')],
        ]
        if 'max_categories' in cat_cfg:
            data.append(['カテゴリ上限', str(cat_cfg.get('max_categories'))])
        data.append(['除外列', ', '.join(exclude_columns) if exclude_columns else 'なし'])
        
        table = Table(data, colWidths=[6*cm, 10*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), japanese_font),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    
    return story


def _build_model_comparison_with_reportlab(session_data: Dict, heading_style, normal_style, japanese_font):
    """モデル比較セクションをReportLabで構築"""
    story = []
    model_results = session_data.get('model_results', {})
    # best_model または best_model_name を確認
    best_model = (session_data.get('best_model') or 
                 model_results.get('best_model_name', 'unknown'))
    
    story.append(Paragraph("4. モデル比較", heading_style))
    
    # model_comparison または comparison を確認
    comparison = None
    if model_results:
        comparison = model_results.get('model_comparison') or model_results.get('comparison')
    
    if comparison:
        data = [['モデル名', 'スコア', '状態']]
        for model_data in comparison:
            model_name = model_data.get('model_name', 'unknown')
            # primary_score, auc, accuracy, rmse のいずれかを使用
            score = (model_data.get('primary_score') or 
                    model_data.get('auc') or 
                    model_data.get('accuracy') or 
                    (1.0 / (1.0 + model_data.get('rmse', 1.0)) if model_data.get('rmse') else 0))
            is_best = model_name == best_model
            status = '✓ 採用' if is_best else ''
            data.append([model_name, f'{score:.4f}', status])
        
        table = Table(data, colWidths=[6*cm, 4*cm, 6*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), japanese_font),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Paragraph(f"<b>採用モデル:</b> {best_model}", normal_style))
    else:
        story.append(Paragraph("モデル比較結果がありません。", normal_style))
    
    return story


def _build_evaluation_with_reportlab(session_data: Dict, heading_style, normal_style, heading3_style, temp_dir: Path, japanese_font):
    """評価セクションをReportLabで構築"""
    story = []
    evaluation_results = session_data.get('evaluation_results', {})
    problem_type = session_data.get('problem_type', 'unknown')
    
    story.append(Paragraph("5. モデル評価", heading_style))
    
    if not evaluation_results:
        story.append(Paragraph("評価結果がありません。", normal_style))
        return story
    
    output_dir_name = evaluation_results.get('d', '')
    output_dir = temp_dir / output_dir_name
    
    if problem_type == 'classification':
        story.append(Paragraph("5.1 分類評価指標", heading3_style))
        data = [['指標', '値']]
        
        if evaluation_results.get('a') is not None:
            data.append(['Accuracy', f"{evaluation_results['a']:.3f}"])
        if evaluation_results.get('p') is not None:
            data.append(['Precision', f"{evaluation_results['p']:.3f}"])
        if evaluation_results.get('r') is not None:
            data.append(['Recall', f"{evaluation_results['r']:.3f}"])
        if evaluation_results.get('f') is not None:
            data.append(['F1-score', f"{evaluation_results['f']:.3f}"])
        if evaluation_results.get('auc') is not None:
            data.append(['AUC', f"{evaluation_results['auc']:.3f}"])
        
        table = Table(data, colWidths=[6*cm, 10*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), japanese_font),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        # 画像を追加
        if evaluation_results.get('roc'):
            roc_path = output_dir / evaluation_results['roc']
            if roc_path.exists():
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("5.2 ROC曲線", heading3_style))
                img = Image(str(roc_path), width=14*cm, height=10*cm)
                story.append(img)
        
        if evaluation_results.get('cm'):
            cm_path = output_dir / evaluation_results['cm']
            if cm_path.exists():
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("5.3 混同行列", heading3_style))
                img = Image(str(cm_path), width=14*cm, height=10*cm)
                story.append(img)
    
    else:  # regression
        story.append(Paragraph("5.1 回帰評価指標", heading3_style))
        data = [['指標', '値']]
        
        if evaluation_results.get('rmse') is not None:
            data.append(['RMSE', f"{evaluation_results['rmse']:.3f}"])
        if evaluation_results.get('mae') is not None:
            data.append(['MAE', f"{evaluation_results['mae']:.3f}"])
        if evaluation_results.get('r2') is not None:
            data.append(['R²', f"{evaluation_results['r2']:.3f}"])
        
        table = Table(data, colWidths=[6*cm, 10*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), japanese_font),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        
        # 画像を追加
        if evaluation_results.get('scatter'):
            scatter_path = output_dir / evaluation_results['scatter']
            if scatter_path.exists():
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("5.2 予測 vs 実測", heading3_style))
                img = Image(str(scatter_path), width=14*cm, height=10*cm)
                story.append(img)
        
        if evaluation_results.get('residual'):
            residual_path = output_dir / evaluation_results['residual']
            if residual_path.exists():
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph("5.3 残差プロット", heading3_style))
                img = Image(str(residual_path), width=14*cm, height=10*cm)
                story.append(img)
    
    return story


def _build_xai_with_reportlab(session_data: Dict, heading_style, normal_style, heading3_style, heading4_style, temp_dir: Path, japanese_font: str):
    """重要度分析セクションをReportLabで構築"""
    story = []
    xai_results = session_data.get('xai_results', {})
    problem_type = session_data.get('problem_type', 'unknown')
    
    story.append(Paragraph("6. 重要度分析", heading_style))
    
    if not xai_results:
        story.append(Paragraph("重要度分析結果がありません。", normal_style))
        return story
    
    # Permutation Importance（テーブル表示）
    story.append(Paragraph("6.1 特徴量重要度（Permutation Importance）", heading3_style))
    all_features = xai_results.get('all_f') or xai_results.get('f', [])
    all_importances = xai_results.get('all_i') or xai_results.get('i', [])
    
    if all_features and all_importances:
        data = [['順位', '特徴量', '重要度', '相対重要度']]
        max_imp = all_importances[0] if all_importances else 1
        for idx, (feature, importance) in enumerate(zip(all_features, all_importances)):
            percentage = (importance / max_imp * 100) if max_imp else 0
            data.append([str(idx + 1), feature, f'{importance:.4f}', f'{percentage:.1f}%'])
        
        table = Table(data, colWidths=[2*cm, 7*cm, 4*cm, 3*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), japanese_font),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
    else:
        story.append(Paragraph("重要度データがありません。", normal_style))
    
    # PDPはUI非表示に合わせて省略
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph("※ 本レポートではPDPグラフは省略しています。", normal_style))
    
    return story


def _build_conclusion_with_reportlab(session_data: Dict, heading_style, normal_style, heading3_style):
    """結論セクションをReportLabで構築"""
    story = []
    evaluation_results = session_data.get('evaluation_results', {})
    xai_results = session_data.get('xai_results', {})
    problem_type = session_data.get('problem_type', 'unknown')
    
    story.append(Paragraph("7. 結論と次のアクション提案", heading_style))
    
    story.append(Paragraph("7.1 分析結果のまとめ", heading3_style))
    
    if problem_type == 'classification':
        if evaluation_results.get('auc'):
            auc = evaluation_results['auc']
            text = f"採用モデルのAUCは {auc:.3f} です。"
            if auc >= 0.9:
                text += "非常に高い性能を示しています。"
            elif auc >= 0.8:
                text += "良好な性能を示しています。"
            elif auc >= 0.7:
                text += "中程度の性能を示しています。"
            else:
                text += "改善の余地があります。"
            story.append(Paragraph(text, normal_style))
    else:
        if evaluation_results.get('r2'):
            r2 = evaluation_results['r2']
            text = f"採用モデルのR²は {r2:.3f} です。"
            if r2 >= 0.9:
                text += "非常に高い説明力を持っています。"
            elif r2 >= 0.7:
                text += "良好な説明力を持っています。"
            elif r2 >= 0.5:
                text += "中程度の説明力を持っています。"
            else:
                text += "改善の余地があります。"
            story.append(Paragraph(text, normal_style))
    
    # 重要特徴量
    if xai_results.get('f'):
        story.append(Paragraph("7.2 重要な特徴量", heading3_style))
        story.append(Paragraph("Permutation Importance分析により、以下の特徴量が重要であることが判明しました:", normal_style))
        for feature in xai_results['f'][:3]:
            story.append(Paragraph(f"• {feature}", normal_style))
    
    story.append(Paragraph("7.3 次のアクション提案", heading3_style))
    story.append(Paragraph("• より多くのデータを収集してモデルの性能向上を図る", normal_style))
    story.append(Paragraph("• 重要な特徴量を活用した特徴量エンジニアリングを検討する", normal_style))
    story.append(Paragraph("• ハイパーパラメータのさらなる調整を検討する", normal_style))
    story.append(Paragraph("• 他のモデルアルゴリズムの試行を検討する", normal_style))
    
    return story


def _build_appendix_with_reportlab(session_data: Dict, heading_style, normal_style, heading3_style):
    """付録セクションをReportLabで構築"""
    story = []
    model_results = session_data.get('model_results', {})
    # best_model または best_model_name を確認
    best_model = (session_data.get('best_model') or 
                 model_results.get('best_model_name', 'unknown'))
    
    story.append(Paragraph("8. 付録", heading_style))
    
    story.append(Paragraph("8.1 モデル設定値", heading3_style))
    story.append(Paragraph(f"採用モデル: {best_model}", normal_style))
    story.append(Paragraph("詳細なハイパーパラメータはモデルファイルに保存されています。", normal_style))
    
    story.append(Paragraph("8.2 技術情報", heading3_style))
    story.append(Paragraph("• フレームワーク: scikit-learn", normal_style))
    story.append(Paragraph("• 前処理: ユーザー設定（数値: 欠損補完/スケーリング, カテゴリ: 欠損補完/エンコーディング, 除外列対応）", normal_style))
    story.append(Paragraph("• 重要度分析手法: Permutation Importance（PDPは生成のみ、レポート非表示）", normal_style))
    
    return story

