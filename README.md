# Insight Report

**デモサイト**: [https://nakayashiki-insight-report.hf.space](https://nakayashiki-insight-report.hf.space)

---

## 特徴

| 機能 | 説明 |
|------|------|
| **自動EDA** | データの統計量・分布・相関を自動分析 |
| **AutoML** | 複数モデルを自動学習し、最適なモデルを選択 |
| **重要度分析** | どの特徴量が予測に重要かを可視化 |
| **PDFレポート** | 分析結果を美しいPDFレポートで出力 |

---


## 使い方

### 1. CSVファイルをアップロード

手持ちのデータ（売上データ、顧客データなど）をCSV形式で準備

### 2. 予測したい列を選択

「売上」「離脱するかどうか」など、予測したい項目を選択

### 3. 自動分析開始

ボタン1つで以下を自動実行：
- データの傾向を可視化
- 機械学習モデルを学習
- 予測精度を評価
- 重要な特徴量を特定

### 4. PDFレポートをダウンロード

分析結果を美しいPDFレポートとして出力

---

## 技術スタック

| カテゴリ | 技術 |
|----------|------|
| **バックエンド** | Python, Flask |
| **機械学習** | scikit-learn, pandas, numpy |
| **可視化** | matplotlib, seaborn |
| **PDF生成** | WeasyPrint, ReportLab |
| **デプロイ** | Docker, Hugging Face Spaces |

---

## プロジェクト構成

```
Insight-Report/
├── app/                    # アプリケーション本体
│   ├── data/              # データ入力処理
│   ├── eda/               # 自動EDA
│   ├── preprocessing/     # 前処理
│   ├── models/            # モデル学習
│   ├── evaluation/        # モデル評価
│   ├── xai/               # 重要度分析
│   └── report/            # PDFレポート生成
├── templates/             # HTMLテンプレート
├── static/                # CSS/JavaScript
└── docs/                  # ドキュメント
```

