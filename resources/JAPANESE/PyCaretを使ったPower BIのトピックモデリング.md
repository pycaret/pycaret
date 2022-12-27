
# PyCaretを使ったPower BIのトピックモデリング

# by Moez Ali

![Power BIのNLPダッシュボード](https://cdn-images-1.medium.com/max/2624/1*SyZczsDz5Pf-4Srfj_p8vQ.png)

[前回の投稿](https://towardsdatascience.com/how-to-implement-clustering-in-power-bi-using-pycaret-4b5e34b1405b)では、Power BIにPyCaretと連携してクラスタリング分析を実装する方法を紹介しました。これにより、アナリストやデータサイエンティストは、追加のライセンスコストなしに、レポートやダッシュボードに機械学習のレイヤーを追加ができます。

今回の記事では、PyCaretを使ってPower BIにトピックモデリングを実装する方法をご紹介します。まだPyCaretを知らない方は、こちらの[お知らせ](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46)を読んで詳細を確認してください。

# このチュートリアルの学習目標

* 自然言語処理とは？

* トピックモデリングとは何か？

自然言語処理とは何か * トピックモデリングとは何か * Power BI で Latent Dirichlet Allocation モデルを学習、実装する。

* 結果を分析し、ダッシュボードで情報を可視化する。

# ♪ ♪ Before we start

Pythonを使ったことがある方は、すでにAnaconda Distributionがインストールされていると思います。もしそうでなければ、[ここをクリック](https://www.anaconda.com/distribution/)して、Python 3.7以上のAnaconda Distributionをダウンロードしてください。

![[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# 環境のセットアップ

Power BIでPyCaretの機械学習機能を使い始める前に、仮想環境を作成し、pycaretをインストールする必要があります。それは4つのステップで行われます。

[✅](https://fsymbols.com/signs/tick/) **ステップ1 - anaconda環境の作成**。

スタートメニューから**Anaconda Prompt**を開き、以下のコードを実行します。

    conda create --name **powerbi** python=3.7

"powerbi "はここで決めた環境の名前です。お好きな名前をつけてください。

[✅](https://fsymbols.com/signs/tick/) **Step 2 - PyCaret**のインストール

Anaconda Promptで以下のコードを実行します。

    pip install **pycaret**

インストールには15～20分程度かかる場合があります。インストールに問題がある場合は、[GitHub](https://www.github.com/pycaret/pycaret)のページにある既知の問題と解決策を参照してください。

[✅](https://fsymbols.com/signs/tick/)**ステップ3 - Power BIにPythonディレクトリを設定する**。

作成した仮想環境をPower BIと連携させる必要があります。これは、Power BI Desktopのグローバル設定（ファイル→オプション→グローバル→Pythonスクリプト）で行うことができます。Anaconda Environmentはデフォルトでは以下の場所にインストールされます。

C:\Users\ **username** \Anaconda3\envs\

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*3qTuOM-N6ekhoiQmDpHgXg.png)

[✅](https://fsymbols.com/signs/tick/)**Step 4 - Install Language Model** (言語モデルのインストール)

NLPタスクを実行するためには、Anaconda Promptで以下のコードを実行して言語モデルをダウンロードする必要があります。

まず、Anaconda Promptでconda環境を起動します。

    conda activate **powerbi**

英語の言語モデルをダウンロードします。

    python -m spacy download en_core_web_sm
    python -m textblob.download_corpora

![python -m spacy download en_core_web_sm](https://cdn-images-1.medium.com/max/3840/1*savPqt23x7nBcK76-0MBxw.png)

![python -m textblob.download_corpora](https://cdn-images-1.medium.com/max/3838/1*NYaSehQvRp9ANsEpC_GPQw.png)

# 自然言語処理とは？

自然言語処理（NLP）は、コンピュータと人間の言語との相互作用を扱う、コンピュータサイエンスと人工知能のサブフィールドです。特にNLPでは、大量の自然言語データを処理・分析するためにコンピュータをプログラムする方法について、幅広い技術を扱っています。

NLPを搭載したソフトウェアは、私たちの日常生活にさまざまな形で役立っており、あなたも知らず知らずのうちに利用しているかもしれません。いくつかの例を挙げてみましょう。

* **パーソナル・アシスタント**。Siri、Cortana、Alexaなど。

* **オートコンプリート**。検索エンジン（例：Google、Bing、Baidu、Yahoo）での検索。

* **スペルチェック**。ブラウザ、IDE（※例：Visual Studio）、デスクトップアプリ（※例：Microsoft Word）など、ほとんどの場所で利用できます。

* **機械翻訳**。Google翻訳。

* **Document Summarization Software:** テキストコンパクタ、Autosummarizer。

![出典 [https://clevertap.com/blog/natural-language-processing](https://clevertap.com/blog/natural-language-processing/)](https://cdn-images-1.medium.com/max/2800/1*IEuGZY5vaWoVnTqQpoZUvQ.jpeg)

トピックモデリングは、テキストデータから抽象的なトピックを発見するために使用される統計モデルの一種です。NLPにおける多くの実用的なアプリケーションの1つです。

# トピックモデリングとは？

トピックモデルとは、教師なし機械学習に属する統計モデルの一種であり、テキストデータの中から抽象的なトピックを発見するために使用されます。トピックモデリングの目的は、一連の文書の中からトピックやテーマを自動的に見つけることです。

トピックモデリングの一般的な使用例は以下の通りです。

* ドキュメントをトピックに分類することで、大規模なテキストデータを要約する（※考え方はクラスタリングに似ています）。

* **Exploratory Data Analysis**顧客のフィードバックフォーム、amazonのレビュー、アンケート結果などのデータを理解する。

* **Feature Engineering**分類や回帰などの教師付き機械学習実験のための特徴量の作成

トピックモデリングに使用されるアルゴリズムはいくつかあります。一般的なものとしては、LDA（Latent Dirichlet Allocation）、LSA（Latent Semantic Analysis）、NMF（Non-Negative Matrix Factorization）などがあります。各アルゴリズムにはそれぞれ数学的な詳細がありますが、このチュートリアルでは説明しません。このチュートリアルでは、PyCaretのNLPモジュールを使って、Power BIにLDA（Latent Dirichlet Allocation）モデルを実装します。

LDAアルゴリズムの技術的な詳細を知りたい方は、[この論文](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)をお読みください。

![出典：[https://springerplus.springeropen.com/articles/10.1186/s40064-016-3252-8](https://springerplus.springeropen.com/articles/10.1186/s40064-016-3252-8)](https://cdn-images-1.medium.com/max/2000/1*DYbV9YMI94QsUeRiiJyrSg.png)

# **トピックモデリングのためのテキスト前処理**

トピックモデリングから意味のある結果を得るためには、テキストデータをアルゴリズムに与えるまえに処理する必要があります。これは、ほとんどすべてのNLPタスクに共通しています。テキストの前処理は、構造化されたデータ(行と列のデータ)を扱うときに機械学習でよく使われる古典的な前処理技術とは異なります。

PyCaretは、**ストップワードの除去**、**トークン化**、**レンマタイズ**、バイグラム/トライグラムの抽出など、15種類以上の技術を適用してテキストデータを自動的に前処理します。PyCaretで利用できるテキスト前処理機能の詳細については、[こちら](https://www.pycaret.org/nlp)を参照してください。

# ビジネス・コンテクストの設定

Kivaは、2005年にサンフランシスコで設立された国際的な非営利団体です。Kivaは、2005年にサンフランシスコで設立された国際的な非営利団体です。

![出典 [https://www.kiva.org/about](https://www.kiva.org/about)](https://cdn-images-1.medium.com/max/2124/1*U4zzTYo6MoCk6PxuZl3FBw.png)

このチュートリアルでは、承認された6,818人のローン申請者のローン情報を含む、Kivaのオープンデータセットを使用します。このデータセットには、ローンの金額、国、性別、そして借り手が提出した申請書のテキストデータが含まれています。

![サンプルデータポイント](https://cdn-images-1.medium.com/max/3194/1*jnQvTmQHhWpOSAgSMqaspg.png)

私たちの目的は、「*en*」列のテキストデータを分析して抽象的なトピックを見つけ、それを使って特定のトピック（または特定の種類のローン）がデフォルト率に与える影響を評価することです。

# 👉始めよう

Anaconda環境のセットアップ、トピックモデリングの理解、そしてこのチュートリアルのビジネスコンテキストを理解したところで、早速始めましょう。

# 1. データの取得

最初のステップは、Power BI Desktopにデータセットを読み込むことです。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

![Power BI Desktop → データの取得 → その他 → Web](https://cdn-images-1.medium.com/max/3828/1*lGqJEUm2lVDcYNDdGNUfbw.png)

csvファイルへのリンクです。
[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv)

# 2. モデルのトレーニング

Power BIでトピックモデルを学習するには、Power Query EditorでPythonスクリプトを実行する必要があります（Power Query Editor → Transform → Run python script）。以下のコードをPythonスクリプトとして実行します。

from **pycaret.nlp** import *
データセット = **get_topics**(dataset, text='en')

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*EwC-QI4m6DORCtdakOAPmQ.png)

PyCaretには、すぐに使える5つのトピックモデルが用意されています。

![a](https://cdn-images-1.medium.com/max/2000/1*LszI1w45K6i5pOBJ0ZmndA.png)

デフォルトでは、PyCaretは4つのトピックで**Latent Dirichlet Allocation (LDA)** モデルを学習します。デフォルト値は簡単に変更できます。

* モデルタイプを変更するには、**get_topics()** の**model**パラメーターを使用します。

* トピックの数を変更するには、**num_topics**パラメーターを使用します。

6つのトピックを持つ**Non-Negative Matrix Factorization**モデルのサンプルコードを参照してください。

from **pycaret.nlp** import *
dataset = **get_topics**(Dataset, text='en', model='nmf', num_topics=6)

**アウトプット:**

![Topic Modeling Results (after execution of Python code)](https://cdn-images-1.medium.com/max/3834/1*DY70gtEWPMy5BPiuKwWPPA.png)

![最終出力(表をクリックした後)](https://cdn-images-1.medium.com/max/3840/1*lbTGdPoqZQkYejsl01D4dQ.png)

トピックの重みを含む新しい列が、元のデータセットに追加されます。クエリを適用すると、Power BIでは最終的に次のような出力になります。

![Results in Power BI Desktop (after applying query)](https://cdn-images-1.medium.com/max/3844/1*btTSFxgmmEV8e7-Nw133mw.png)

# 3. ダッシュボード

Power BIでトピック・ウェイトを取得したら、それをダッシュボードで可視化してインサイトを生成する例を紹介します。

![ダッシュボードの概要ページ](https://cdn-images-1.medium.com/max/2624/1*SyZczsDz5Pf-4Srfj_p8vQ.png)

![ダッシュボードの詳細ページ](https://cdn-images-1.medium.com/max/2660/1*SVY-1iq0qXmh_dl8D3rl0w.png)

PBIXファイルとデータセットは、弊社の[GitHub](https://github.com/pycaret/powerbi-nlp)からダウンロードできます。

PyCaretを使ったJupyter notebookでのトピックモデリングの実装について詳しく知りたい方は、こちらの2分間のビデオチュートリアルをご覧ください。

 <iframe src="https://medium.com/media/75ec7a7299cd663bd63aa14ba8716025" frameborder="0"> </iframe>。

トピックモデリングについてもっと知りたいという方は、初心者向けのNLP 101 [Notebook Tutorial](https://www.pycaret.org/nlp101)もご覧ください。

[LinkedIn](https://www.linkedin.com/company/pycaret/)をフォローしたり、[Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)のチャンネルを購読したりして、PyCaretについてもっと学びましょう。

# 重要なリンク

[ユーザーガイド / ドキュメント](https://www.pycaret.org/guide)
[GitHub リポジトリ
](https://www.github.com/pycaret/pycaret)[PyCaret のインストール](https://www.pycaret.org/install)
[ノートブックチュートリアル](https://www.pycaret.org/tutorial)
[PyCaretに貢献する](https://www.pycaret.org/contribute)

# 特定のモジュールについて学びたい？

最初のリリース1.0.0の時点で、PyCaretは以下のモジュールを利用することができます。以下のリンクをクリックすると、Pythonでのドキュメントや動作例を見ることができます。

[Classification](https://www.pycaret.org/classification)
[回帰
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[異常検知
](https://www.pycaret.org/anomaly-detection)[自然言語処理](https://www.pycaret.org/nlp)
[アソシエーション・ルール・マイニング](https://www.pycaret.org/association-rules)

# こちらもご覧ください。

PyCaret getting started tutorials in Notebook:

[Clustering](https://www.pycaret.org/clu101)
[異常検知](https://www.pycaret.org/anom101)
[自然言語処理](https://www.pycaret.org/nlp101)
[アソシエーションルールマイニング](https://www.pycaret.org/arul101)
[回帰](https://www.pycaret.org/reg101)
[分類](https://www.pycaret.org/clf101)

# Would you like to contribute?

PyCaretはオープンソースのプロジェクトです。誰でも貢献することができます。貢献したい方は、[オープンイシュー](https://github.com/pycaret/pycaret/issues)にお気軽に取り組んでください。Pull Request は dev-1.0.1 ブランチのユニットテスト付きで受け付けています。

PyCaretを気に入っていただけましたら、[GitHub repo](https://www.github.com/pycaret/pycaret)に⭐️をお願いします。

媒体 : [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn : [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter : [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
