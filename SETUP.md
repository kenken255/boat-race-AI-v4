# BOAT RACE AI Cloud v5 — セットアップ手順

V5は **GitHub + Streamlit Community Cloud + Supabase** を基本構成とします。

- GitHub: コード保管 + 自動一括学習ワーカー
- Streamlit Community Cloud: iPhoneから使うWebアプリ
- Supabase: 予想・結果・バックテストキュー・学習履歴の永続保存

V4から更新する場合も、下の `supabase_schema.sql` を再実行すればV5用テーブル/列を追加できるようにしてあります。

---

## 1. ZIPを展開

ZIPを「すべて展開」します。

主なファイル:

- `streamlit_app.py`
- `core.py`
- `backtest.py`
- `storage.py`
- `worker.py`
- `requirements.txt`
- `supabase_schema.sql`
- `.github/workflows/v5-backfill.yml`
- `.streamlit/config.toml`

クラウド版なので `start.bat` は不要です。

---

## 2. GitHubへアップロード

GitHubで新しいRepositoryを作ります。

例: `boatrace-ai-v5`

Repositoryトップに上記ファイルをアップロードします。

重要: `.github/workflows/v5-backfill.yml` もアップロードしてください。GitHub Web画面では隠しフォルダが扱いにくい場合があるため、フォルダ構造が次の形になっていることを確認します。

```text
boatrace-ai-v5/
  streamlit_app.py
  core.py
  backtest.py
  storage.py
  worker.py
  requirements.txt
  supabase_schema.sql
  .github/
    workflows/
      v5-backfill.yml
```

---

## 3. Supabaseを作成

SupabaseでProjectを作ります。

SQL Editor → New Query を開き、`supabase_schema.sql` の全文を貼り付けてRunします。

V5では次を使用します。

- `race_snapshots`: 各レースの予想時点データ・結果
- `ticket_predictions`: 券種ごとの買い目予測・オッズ・払戻
- `model_runs`: モデル再評価履歴
- `backtest_jobs`: 一括学習ジョブ設定/進捗
- `backtest_queue`: 処理対象レースキュー

### V4から更新する場合

新しい `supabase_schema.sql` をもう一度Runしてください。

`create table if not exists` / `add column if not exists` を使っているため、既存履歴を残したままV5列を追加する設計です。

---

## 4. Supabase URL / Secret Key

Supabase ProjectのConnectまたはAPI Keys画面から次を取得します。

- Project URL
- サーバー用Secret Key (`sb_secret_...` または利用中Projectのserver-side key)

Secret KeyはGitHubのコードやREADMEには書かないでください。

---

## 5. Streamlit Community Cloudへ公開

Streamlit Community CloudでCreate app。

- Repository: `あなたのGitHub名/boatrace-ai-v5`
- Branch: `main`
- Main file path: `streamlit_app.py`
- Python: 3.12推奨

Advanced settings → Secretsへ次を入れます。

```toml
[supabase]
url = "https://xxxxxxxx.supabase.co"
key = "sb_secret_xxxxxxxxxxxxxxxxx"

[app]
pin = "自分だけが知っているPIN"
```

Deployします。

発行された `https://xxxx.streamlit.app` をiPhone Safariで開きます。

Safari → 共有 → ホーム画面に追加、でアプリ風に使えます。

---

# 6. 通常予想の使い方

「🎯 予想」タブを開きます。

1. 日付
2. 開催場を自動検出
3. 場
4. レース
5. 券種
6. 出走表・オッズを取得

展示前なら「展示・直前情報も取得」をOFFにできます。

取得後は、風速・波高・気温・水温・進入・展示タイム・展示ST・チルトを手修正できます。

---

# 7. 要素重みスライダー

「🎚️ 要素の重みづけ」を開きます。

各バーは0〜200%。

- 100% = 標準
- 50% = 標準の半分
- 0% = その要素を無視
- 150% = 標準の1.5倍
- 200% = 標準の2倍

変更可能:

- コース/進入
- 級別
- 全国成績
- 当地成績
- モーター
- ボート
- 平均ST
- 展示タイム
- 展示ST
- F/Lペナルティ

プリセットを選び、「このプリセットをバーに反映」を押すこともできます。

### 自己学習モデル採用後

「自己学習モデルがある場合の手動重み反映度」を使います。

例:

- 0%: 学習モデルのみ
- 30%: 学習モデル70% + 手動重みモデル30%相当の強度混合
- 100%: 手動重みモデルのみ

※確率そのものを単純平均するのではなく、レース内で標準化した強度を混合してから確率化します。

---

# 8. 過去一括学習ジョブを作る

「⏪ 一括学習」タブを開きます。

入力:

- 開始日 / 終了日
- 対象場
- 券種
- 最大レース数
- 1レースあたりMonte Carlo回数
- 仮想レース比率
- 不確実性
- 再学習チェック間隔
- 展示・直前情報を使用するか
- 過去オッズを使用するか

ジョブ作成時に、現在の要素重みもコピーされます。

「一括学習ジョブを作成」を押すと、開催日を確認して `backtest_queue` に対象レースを作ります。

---

# 9. Streamlit画面から一括処理

ジョブを選び、

- 5R
- 20R
- 50R

の処理ボタンを押せます。

各レースが終わるごとにSupabaseへ保存するため、途中でSafariを閉じても処理済み分は失われません。

ただし、数百〜数千RをStreamlitの1セッションで走らせ続けるより、次のGitHub Actions自動ワーカーを推奨します。

---

# 10. GitHub Actionsで数百Rを自動処理

これがV5の「放置して過去学習を進める」部分です。

## 10-1. GitHub Actions用Secrets

GitHub Repository → Settings → Secrets and variables → Actions → New repository secret。

2つ作ります。

### `SUPABASE_URL`

値: Supabase Project URL

### `SUPABASE_KEY`

値: Supabase Secret Key

Streamlit Secretsとは別に、GitHub Actionsにも登録が必要です。

## 10-2. Actionsを有効化

RepositoryのActionsタブを開き、Workflowの実行を許可します。

`V5 historical backfill worker` が表示されます。

## 10-3. 手動で100R進める

Actions → V5 historical backfill worker → Run workflow。

`max_races` を100などにして実行します。

`worker.py` が未処理のジョブを探し、古いレースから順に処理します。

## 10-4. 自動継続

同梱Workflowは6時間ごとに未処理ジョブを確認します。

未処理があれば既定で最大100Rずつ進めます。

つまり500Rジョブなら、1回で全部処理できなくても、DBに進捗を残しながら次回のWorkflowが続きから処理します。

負荷を上げたくない場合は `.github/workflows/v5-backfill.yml` のscheduleを削除し、手動実行だけにできます。

---

# 11. 未来情報を混ぜない仕組み

過去レースを予想するとき、学習データは `race_date < 対象日` の確定レースだけを読みます。

したがって、2026-06-10のレースを予測するときに2026-06-11以降の結果を使わない設計です。

同日レースも、標準ではその日の結果を当日の別レース予測に混ぜません。これは実戦再現を保守的にするためです。

---

# 12. 自己改善

履歴が増えると候補Logistic Regressionモデルを学習します。

履歴を時間順に、

- 過去側: 学習
- 新しい側: 検証

へ分割します。

基準ヒューリスティックと候補モデルを、

- Log Loss
- Brier Score

で比較し、改善時だけ採用します。

十分な履歴があればIsotonic Regressionで確率校正も行います。

新モデルを作っただけで自動採用しないのは、過学習を抑えるためです。

---

# 13. 重み設定を比較する方法

例えば次の2ジョブを作ります。

### ジョブA

- 500R
- プリセット: 標準

### ジョブB

- 同じ期間/場/券種
- 500R
- プリセット: ST重視

各ジョブには重みが固定保存されるため、条件が途中で変わりません。

学習タブのBrier/Log Lossや、券種・市場×AI分類別の参考回収率を比較して、どの設定が安定しているかを確認します。

注意: 同じ過去データを見ながら何度も重みを調整すると過学習しやすいため、最後に別期間を「最終テスト期間」として残すのがおすすめです。

---

# 14. トラブル時

## Streamlitは動くが一括学習タブでDBエラー

新しい `supabase_schema.sql` をSupabase SQL Editorで再実行してください。

## GitHub ActionsでSUPABASEエラー

Repository Secretsの名前が完全に、

- `SUPABASE_URL`
- `SUPABASE_KEY`

になっているか確認します。

## 過去レース取得に失敗が多い

古い日付で直前情報/オッズが取得できない場合があります。

失敗分は「失敗分を再試行待ちに戻す」で再試行できます。直前情報やオッズなしの別ジョブを作る方法もあります。

## Actionsで取得先サイトへの通信が失敗

クラウド実行元IPや取得先側の仕様変更により発生する可能性があります。その場合はStreamlitから小分け実行するか、取得コードの修正が必要です。

---

# 15. セキュリティ

- Supabase Secret KeyをGitHubコードへ書かない
- StreamlitではSecretsへ保存
- GitHub ActionsではRepository Secretsへ保存
- PINを設定する
- `.streamlit/secrets.toml` や `.env` をGitへ追加しない

同梱 `.gitignore` はこれらを除外する設定です。
