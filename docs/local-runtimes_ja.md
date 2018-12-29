# [![Google](https://www.google.com/images/logos/google_logo_41.png)](https://www.google.com/)

## ローカルランタイム Local runtimes

コラボラトリではジュピター (Jupyter) を用いることでローカルランタイムに接続します。すなわち，ローカルのハードウェア上でコードを実行し，ローカルファイルにアクセスすることができます。

## セキュリティ問題

実行する前にノートブックの作者が信頼できることを確認しておいてください。
コラボラトリがローカル資源と接続すれば，ローカルコンピュータ上で実行可能なコードの読み，書き，削除が可能になります。

ローカルマシンで動作しているジュピターノートブックに接続すると便宜が得られる一方，リスクも存在します。
ローカルな資源と接続すると，ノートブックのコードを実行するコラボラトリはお使いの PC の計算資源を消費します。
これによりノートブックから次のようなことが可能になります。

- 任意のコマンドの呼び出し (たとえば `rm -rf /`)
- ローカルファイルへのアクセス
- 悪意あるコンテンツの実行

ローカルランタイムに接続しようとする前に、あなたがノートブックの作者を信頼し、どのコードが実行されているのか理解していることを確認してください。
ジュピターノートブックサーバーのセキュリティモデルの詳細については、[Jupyter\'s documentation](http://jupyter-notebook.readthedocs.io/en/stable/security.html) を参照してください。

## セットアップ

コラボラトリをローカルで実行されているジュピターサーバに接続するためには，以下の手順が必要になります。

### ステップ 1: ジュピターのインストール

ご自身のマシンにジュピター [Jupyter](http://jupyter.org/install) をインストール

### ステップ 2: ジュピター拡張 `jupyter_http_over_ws` のインストールと有効化(一度だけ)

`jupyter_http_over_ws` 拡張はコラボラトリチームが作成したもので，
[GitHub](https://github.com/googlecolab/jupyter_http_over_ws) から入手できます。
以下に手順を示します。

```bash
pip install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
```

### ステップ 3: サーバの起動と認証

次のようにコラボラトリのフロントエンドから webSocket コネクションを明示的にセットすることで
新ノートブックサーバを通常どおり起動します:

```bash
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
```

上記で用いたポート番号は次のステップで使用します

### ステップ 4: ローカルランタイムに接続する

ジュピターノートブックを `--no-browser` オプション付きで起動した場合，コラボラトリを接続する前に URL にアクセスする必要がある場合があります。
この URL はブラウザとコラボラトリ間の認証用にクッキーを設定します。

コラボラトリから，`Connect` をボタンをクリックし，次に `Connect to local runtime...` を選択します。
表示されるダイアログボックスで，ステップ 3 で指定したポート番号を入力し，`Connect` ボタンを押下します。
これでコラボラトリはローカルランタイムに接続されます。


## ブラウザ固有の設定

注: Mozilla Firefox をお使いならば，[Firefox config editor](https://support.mozilla.org/en-US/kb/about-config-editor-firefox) 内の設定 `network.websocket.allowInsecureFromHTTPS` をする必要があります。
コラボラトリはローカルカーネルと WebSocket を使って接続を試みます。 Firebox はデフォルトでは HTTPS ドメインからの WebScoket 接続を許可していません。

## 共有

ノートブックを他人と共有した場合，ローカルマシン上のランタイムは共有されます。
他の人がノートブックを開いた場合には通常クラウド上のランタイムに接続されます。

デフォルトでは全てのコードセルの出力はグーグルドライブに保存されます。
ローカルな機密情報をアクセスし，コードセルの抑えたければ保存時にノートブックの設定
「編集 edit」でコードセル出力の抑制を選んでください。

## グーグル計算エンジンインスタンス (Google Compute Engine instance)上のランタイムに接続する 

もしジュピターノートブックサーバで他のマシン(Google Compute Engine instance など) と接続する場合，コラボラトリに SSH ローカルポートフォワーディングを許可する必要があります。

グーグルクラウドプラットフォームは，コラボラトリのディープラーニング仮想マシンをサポートしています。
Google Compute Engine instance と SSH 接続するためには [how-to guides](https://cloud.google.com/deep-learning-vm/docs/) を参照してください。
このイメージを使う場合はステップ 4 へ飛んでください。
(using port 8888).

第一に，上で用いたジュピターノートブックサーバの設定をし，

第二に，ローカルマシンから遠隔インスタンス(Google Compute Engine instanceなど)への SSH 接続を `-L` フラグ付きで確立します。
例えば Google Compute Engine instance の 8888 番ポートで以下のようにします。

```bash
gcloud compute ssh --zone YOUR_ZONE YOUR_INSTANCE_NAME -- -L 8888:localhost:8888
```

最後にコラボラトリ内でポートフォワーディングによるポート(ステップ4で指定したローカルランタイムへの接続で用いた)を用いてください
