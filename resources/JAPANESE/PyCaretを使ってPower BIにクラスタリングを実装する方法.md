
# PyCaretを使ってPower BIにクラスタリングを実装する方法

# by Moez Ali

![Clustering Dashboard in Power BI](https://cdn-images-1.medium.com/max/2632/1*sUeqYcENVII1RlyYA_-Uxg.png)

前回の投稿](https://towardsdatascience.com/build-your-first-anomaly-detector-in-power-bi-using-pycaret-2b41b363244e)では、Power BIにPyCaretを統合して異常検知器を構築する方法を紹介しました。これにより、アナリストやデータサイエンティストは、追加のライセンス費用なしに、レポートやダッシュボードに機械学習のレイヤーを追加することができます。

今回の記事では、PyCaretを使ってPower BIにクラスタリング分析を実装する方法をご紹介します。まだPyCaretを知らない方は、この[お知らせ](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46)を読んで詳細を確認してください。

# このチュートリアルの学習目標

* クラスタリングとは何か？クラスタリングの種類。

* Power BI で教師なしクラスタリングモデルを学習・実装する。

* 結果を分析し、情報をダッシュボードで可視化する。

* Clustering モデルを Power BI の本番環境に導入するには？

# 始める前に

Python を使ったことがある方は、すでに Anaconda Distribution がインストールされていると思います。もしそうでなければ、[ここをクリック](https://www.anaconda.com/distribution/)して、Python 3.7 以上の Anaconda Distribution をダウンロードしてください。

![[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# 環境のセットアップ

Power BIでPyCaretの機械学習機能を使い始める前に、仮想環境を作成し、pycaretをインストールする必要があります。これは3つのステップで行います。

[✅](https://fsymbols.com/signs/tick/) **ステップ1 - anaconda環境の作成**。

スタートメニューから**Anaconda Prompt**を開き、以下のコードを実行します。

    conda create --name **myenv** python=3.7

[✅](https://fsymbols.com/signs/tick/) **Step 2 - PyCaret** のインストール

Anaconda Promptで以下のコードを実行します。

    pip install pycaret

インストールには15～20分程度かかる場合があります。インストールに問題がある場合は、[GitHub](https://www.github.com/pycaret/pycaret)のページにある既知の問題と解決策をご覧ください。

[✅](https://fsymbols.com/signs/tick/)**ステップ3 - Power BIにPythonディレクトリを設定する**。

作成した仮想環境は、Power BIと連携する必要があります。これは、Power BI Desktopのグローバル設定（ファイル→オプション→グローバル→Pythonスクリプト）で行うことができます。Anaconda Environmentはデフォルトでは以下の場所にインストールされます。

C:Users%%***username%%***AppData%%Local%%Continuum%%anaconda3%%envs%%myenv

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*zQMKuyEk8LGrOPE-NByjrg.png)

# Clusteringとは？

クラスタリングとは、似たような特徴を持つデータポイントをグループ化する手法です。これらのグループ化は、データの調査、パターンの特定、データのサブセットの分析に役立ちます。データをクラスターにまとめることで、データの基本的な構造を特定することができ、様々な業界で応用されています。クラスタリングの一般的なビジネスユースケースは次のとおりです。

マーケティングのための顧客セグメンテーション。

プロモーションやディスカウントのための顧客の購買行動分析。

COVID-19のような伝染病のジオクラスターの特定。

# クラスタリングの種類

クラスタリングタスクの主観的な性質を考慮すると、さまざまなタイプの問題に適したさまざまなアルゴリズムがあります。それぞれのアルゴリズムには独自のルールがあり、クラスターを計算するための数学があります。

このチュートリアルでは、PyCaretというPythonライブラリを使ってPower BIにクラスタリング分析を実装します。このチュートリアルでは、PyCaretというPythonライブラリを使ったPower BIでのクラスタリング分析の実装について説明します。アルゴリズムの詳細や、アルゴリズムの背後にある数学については、このチュートリアルでは説明しません。

![Ghosal A., Nandy A., Das A.K., Goswami S., Panday M. (2020) A Short Review on Different Clustering Techniques and Their Applications.](https://cdn-images-1.medium.com/max/2726/1*2eQuIebjtTMJot27bWXgCQ.png)を参照してください。

このチュートリアルでは、最もシンプルで人気のある教師なし機械学習アルゴリズムの1つであるK-Meansアルゴリズムを使用します。K-Meansについてより詳しく知りたい方は、[この論文](https://stanford.edu/~cpiech/cs221/handouts/kmeans.html)をご覧ください。

# ビジネスコンテクストの設定

このチュートリアルでは、世界保健機関（WHO）の*** ***Global Health Expenditureデータベースから、現在の医療費データセットを使用します。このデータセットは、2000年から2017年までの200カ国以上の国の国内総生産に対する医療費の割合を含んでいます。

我々の目的は、K-Meansクラスタリングアルゴリズムを使用して、このデータのパターンとグループを見つけることです。

[ソースデータ](https://data.worldbank.org/indicator/SH.XPD.CHEX.GD.ZS)

![サンプルデータのポイント](https://cdn-images-1.medium.com/max/2366/1*E1z19x_qa7rko1FZpAw61Q.png)

# 👉 始めてみよう

Anaconda 環境の設定、PyCaret のインストール、クラスタリング分析の基本の理解、そしてこのチュートリアルのビジネスコンテキストの理解ができたところで、始めましょう。

# 1. データの取得

最初のステップは、Power BI Desktopにデータセットを読み込むことです。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

![Power BI Desktop → データの取得 → その他 → Web](https://cdn-images-1.medium.com/max/3842/1*JZ3MwRe8rJXp5e0ac7lamw.png)

csvファイルへのリンクです。
[https://github.com/pycaret/powerbi-clustering/blob/master/clustering.csv](https://github.com/pycaret/powerbi-clustering/blob/master/clustering.csv)

# 2. モデルの学習

Power BIでクラスタリングモデルをトレーニングするには、Power Query EditorでPythonスクリプトを実行する必要があります（Power Query Editor → Transform → Run python script）。以下のコードをPythonスクリプトとして実行します。

    from **pycaret.clustering** import *.
    dataset = **get_clusters**(Dataset, num_clusters=5, ignore_features=['Country'])

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*SK0XxzF9XZlwtGH1786OUQ.png)

ignore_features**パラメータを使用して、データセット内の「*Country*」列を無視しています。機械学習アルゴリズムの学習に特定の列を使用したくない理由はたくさんあるでしょう。

PyCaretでは、データセットから不要なカラムを削除するのではなく、非表示にすることができ、後の分析でそれらのカラムが必要になるかもしれません。例えば、今回のケースでは、アルゴリズムの学習に「Country」を使用したくないので、**ignore_features.**に渡しています。

PyCaretには8種類以上のすぐに使えるクラスタリングアルゴリズムがあります。

![](https://cdn-images-1.medium.com/max/2632/1*ihezKFr61Vrgu7E-0-JA5g.png)

デフォルトでは、PyCaretは4つのクラスタを持つ**K-Means Clusteringモデル**を学習します。デフォルト値は簡単に変更できます。

*モデルタイプを変更するには、**get_clusters()**の**model ***パラメータを使用します。

モデルタイプを変更するには，**get_clusters()**の**model**パラメータを使用する。

6クラスタの**K-Modes Clustering**のサンプルコードを参照してください。

    from **pycaret.clustering **import *.
    dataset = **get_clusters**(Dataset, model='kmodes', num_clusters=6, ignore_features=['Country'])

**出力:**

![Clustering Results (after execution of Python code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![Final Output (after clicking on Table)](https://cdn-images-1.medium.com/max/3848/1*a6mAzuXC8Ta6gRyolaF5uA.png)

クラスターのラベルを含む新しい列が元のデータセットに追加されます。その後、すべての年の列を*unpivoted*してデータを正規化し、Power BIでの視覚化に使用できるようにします。

最終的な出力結果をPower BIで見ると、以下のようになります。

![Results in Power BI Desktop (after applying query)](https://cdn-images-1.medium.com/max/2564/1*oy_X3VIdVPS32qQxkOeehw.png)

# 3. ダッシュボード

Power BIでクラスタラベルを取得したら、ダッシュボードで可視化してインサイトを生成する例を紹介します。

![ダッシュボードの概要ページ](https://cdn-images-1.medium.com/max/2632/1*sUeqYcENVII1RlyYA_-Uxg.png)

![Dashboardの詳細ページ](https://cdn-images-1.medium.com/max/2632/1*1ck--1zR_hRPqREKDC7ztg.png)

PBIXファイルとデータセットは、弊社の[GitHub](https://github.com/pycaret/powerbi-clustering)からダウンロードできます。

# 👉 本番でのクラスタリングの実装

以上、Power BIにClusteringを実装する簡単な方法をご紹介しました。ただし、上記の方法では、Power BIのデータセットが更新されるたびに、クラスタリングモデルのトレーニングが行われることに注意が必要です。これが問題になる理由は2つあります。

* 新しいデータでモデルを再学習すると、クラスターのラベルが変わる可能性があります（例：以前はクラスター1とラベル付けされていたデータポイントが、再学習時にはクラスター2とラベル付けされている場合があります）。

* モデルの再トレーニングに毎日何時間も費やしたくはないでしょう。

Power BIでクラスタリングを実装するためのより生産的な方法は、毎回モデルを再トレーニングするのではなく、事前にトレーニングされたモデルを使ってクラスタラベルを生成することです。

# 事前にモデルをトレーニングする

機械学習モデルのトレーニングには、統合開発環境（IDE）やノートブックを使用することができます。この例では、Visual Studio Codeを使ってクラスタリングモデルを学習しています。

![Model Training in Visual Studio Code](https://cdn-images-1.medium.com/max/2000/1*5roevyCmjxWthy0bYyf4ow.png)

学習したモデルをpickleファイルとして保存し、Power Queryにインポートしてクラスタラベルを生成します。

![Clustering Pipeline saved as a pickle file](https://cdn-images-1.medium.com/max/2000/1*XxknQxv_O_Cx1WJ4kzwPkQ.png)

PyCaretを使ったJupyter notebookでのクラスタリング分析の実装について詳しく知りたい方は、こちらの2分間のビデオチュートリアルをご覧ください。

 <iframe src="https://medium.com/media/ac70d2254314877ee7e9e524e1f2b1bf" frameborder=0></iframe>

# Pre-trained modelの使用

以下のコードをPythonスクリプトとして実行すると、事前に学習したモデルからラベルが生成されます。

    from **pycaret.clustering **import *.
    dataset = **predict_model**('c:/.../clustering_deployment_20052020, data = dataset)

これの出力は、上で見たものと同じになります。違いは、事前に学習したモデルを使う場合、モデルを再学習するのではなく、同じモデルを使って新しいデータセットでラベルを生成することです。

# Power BI Serviceで動作させる

.pbixファイルをPower BIサービスにアップロードしたら、機械学習パイプラインをデータパイプラインにシームレスに統合するために、さらにいくつかのステップが必要です。その内容は以下の通りです。

* **Enable scheduled refresh for the dataset** - Pythonスクリプトでデータセットを含むワークブックのスケジュール更新を有効にするには、[Configuring scheduled refresh](https://docs.microsoft.com/en-us/power-bi/connect-data/refresh-scheduled-refresh)を参照してください。また、**Personal Gateway**に関する情報も含まれています。

* **Personal Gateway**のインストール - ファイルが置かれているマシンとPythonがインストールされているマシンに**Personal Gateway**がインストールされている必要があり、Power BIサービスはそのPython環境にアクセスできなければなりません。パーソナルゲートウェイのインストールと設定](https://docs.microsoft.com/en-us/power-bi/connect-data/service-gateway-personal-mode)の詳細については、こちらをご覧ください。

クラスタリング分析について詳しく知りたい方は、[ノートブックチュートリアル](https://www.pycaret.org/clu101)をご覧ください。

# PyCaret 1.0.1 がリリースされます!

PyCaret 1.0.1 がリリースされます！ コミュニティからの圧倒的なサポートとフィードバックを受けています。私たちは、PyCaretの改善と次のリリースに向けて積極的に取り組んでいます。**PyCaret 1.0.1 は、より大きく、より良くなります**。もし、あなたがフィードバックを共有し、私たちのさらなる改善に役立てたいとお考えでしたら、ウェブサイト上の [fill this form](https://www.pycaret.org/feedback) や、私たちの [GitHub ](https://www.github.com/pycaret/)や [LinkedIn](https://www.linkedin.com/company/pycaret/) ページにコメントを残してください。

LinkedIn](https://www.linkedin.com/company/pycaret/)をフォローしたり、[Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)のチャンネルを購読したりして、PyCaretについてもっと知ってください。

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
