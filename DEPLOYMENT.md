# 応募用公開チェックリスト

企業の担当者がリンクを開いたときに、Hugging Faceログインやアプリ登録で詰まらないようにするための設定です。

## Hugging Face Spacesで公開する場合

Spaceの `Settings` で次を設定します。

### Visibility

- `Public` にする
- ソースコードを見せたくない場合は `Protected` を検討する
- `Private` は応募先が開けない可能性が高いので使わない

### Variables and secrets

`SECRET_KEY` は必ず `Secret` として固定値を設定します。未設定だと再起動ごとにログインセッションが無効になります。

生成例:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

応募用にアカウント作成なしで触れるようにする場合は、以下も設定します。

```text
PORTFOLIO_DEMO_MODE=true
SESSION_PERMANENT=true
PERMANENT_SESSION_LIFETIME_SECONDS=2592000
```

`PORTFOLIO_DEMO_MODE=true` のとき、ログインが必要なページを開いた閲覧者は自動的に「デモユーザー」として扱われます。
トップページにはサンプルデータで試せる入口も表示されます。サンプルにはTitanicの乗客データを使い、`Perished`（1=死亡、0=生存）を予測する分類デモとして案内します。

### Storage

ユーザー登録情報や生成ファイルを長期間残したい場合は、永続ストレージを使ってください。通常ディスクは再起動で消える前提で考えます。

## 応募資料に貼るリンク

応募フォームには次を並べると安全です。

- ライブデモURL
- GitHubリポジトリURL
- 1枚のスクリーンショット、またはREADME内の操作イメージ

ライブデモが起動待ちになっても、担当者が内容を確認できる逃げ道を用意しておくのが目的です。
