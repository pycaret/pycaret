
# PyCaretを使ったPower BIでのはじめての異常検知器の構築

# Power BIに異常検知機能を実装するためのステップバイステップのチュートリアル

# by Moez Ali

![Anomaly Detection Dashboard in Power BI](https://cdn-images-1.medium.com/max/2000/1*sh9LrK5WiF1pBDDR1PCK0g.png)

前回の記事[Machine Learning in Power BI using PyCaret](https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a)では、PyCaretをPower BIに統合することで、アナリストやデータサイエンティストが追加のライセンスコストなしに、ダッシュボードやレポートに機械学習のレイヤーを追加することができるという**ステップバイステップのチュートリアル**を紹介しました。

この記事では、PyCaretを使ってPower BIに異常検知器を実装する方法を紹介します。まだPyCaretを知らない方は、こちらの[お知らせ](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46)をご覧ください。

# このチュートリアルの学習目標

* 異常検知とは何か？異常検知の種類は？

* Power BIで教師なしの異常検知器を訓練し、実装する。

* 結果を分析し、ダッシュボードで情報を可視化します。

* Power BIの本番環境に異常検知器を導入するには？

# 始める前に

Pythonを使ったことがある方は、すでにAnaconda Distributionがインストールされていると思います。もしそうでなければ、[ここをクリック](https://www.anaconda.com/distribution/)して、Python 3.7以上のAnaconda Distributionをダウンロードしてください。

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

# Anomaly Detection とは？

異常検知 **** は、大部分のデータとは大きく異なることで疑念を抱かせる**希少なアイテム**、**イベント**、**観察**を特定するために使用される機械学習の手法です。

一般的に、異常な項目は、銀行詐欺、構造的欠陥、医療問題、エラーなど、何らかの問題に結びつきます。異常検出器の実装には3つの方法があります。

**(a) 監視型。(a) 監視型： **データセットに、どの取引が異常でどの取引が正常かを示すラベルがある場合に使用されます。*(これは教師付き分類問題に似ています)*。

**(b) Semi-Supervised: **半教師付き異常検知の考え方は、異常のない正常なデータのみでモデルを学習するというものです。訓練されたモデルを未知のデータポイントに使用すると、新しいデータポイントが正常かどうか（訓練されたモデル内のデータの分布に基づいて）を予測することができます。

**(c) Unsupervised (教師なし) (c) Unsupervised: **「教師なし」とは、その名の通り、ラベルがなく、訓練データセットとテストデータセットがないことを意味します。教師なし学習では、データセット全体でモデルを学習し、インスタンスの大部分が正常であると仮定します。その一方で、残りの部分に最も適合していないと思われるインスタンスを探します。教師なしの異常検知アルゴリズムには、Isolation ForestやOne-Class Support Vector Machineなどがあります。それぞれがデータセットの異常を識別する独自の方法を持っています。

このチュートリアルでは、PyCaretというPythonライブラリを使って、Power BIに教師なし異常検知を実装します。これらのアルゴリズムの背後にある具体的な詳細や数学についての議論は、このチュートリアルでは対象外です。

![Goldstein M, Uchida S (2016) A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data. PLo](https://cdn-images-1.medium.com/max/2800/1*-Cnyg6-F-Qd4r1Ptcf6nNw.png)

# ビジネスコンテクストの設定

多くの企業では、業務上の購買を効率的に管理するために、従業員に法人向けのクレジットカード（※パーチェスカードまたは※Pカードとも呼ばれる）を発行しています。通常は、従業員が請求書を電子的に提出するプロセスがあります。収集されるデータは、通常、取引日、ベンダー名、支出の種類、加盟店、金額などの取引に関するものです。

このチュートリアルでは、米国デラウェア州の教育省の2014年から2019年までの州職員のクレジットカード取引を使用します。このデータは、同社の[open data](https://data.delaware.gov/Government-and-Finance/Credit-Card-Spend-by-Merchant/8pzf-ge27)プラットフォームでオンラインで入手できます。

![[https://data.delaware.gov/Government-and-Finance/Credit-Card-Spend-by-Merchant/8pzf-ge27](https://data.delaware.gov/Government-and-Finance/Credit-Card-Spend-by-Merchant/8pzf-ge27)](https://cdn-images-1.medium.com/max/3058/1*c8KS7taBuTRlxJ7tTL964g.png)

**免責事項：** *このチュートリアルでは、Power BIでPyCaretを使用して異常検知器を構築する方法を説明します。このチュートリアルで構築されるサンプルダッシュボードは、決して実際の異常を反映したものではなく、また異常を特定するためのものでもありません。

# 👉 始めよう

Anaconda 環境のセットアップ、PyCaret のインストール、異常検知の基本的な理解、このチュートリアルのビジネスコンテキストの理解ができたところで、始めましょう。

# 1. データの取得

最初のステップは、Power BI Desktopにデータセットをインポートすることです。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

![Power BI Desktop → データの取得 → その他 → Web](https://cdn-images-1.medium.com/max/3840/1*WMQRdUPcw8VaG0HIOiGyQQ.png)

**csvファイルへのリンクです。[**https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/delaware_anomaly.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/delaware_anomaly.csv)

# 2. モデルの学習

Power BIで異常検知器を学習するには、Power Query EditorでPythonスクリプトを実行する必要があります（Power Query Editor → Transform → Run python script）。以下のコードをPythonスクリプトとして実行します。

    from **pycaret.anomaly** import *.
    データセット = **get_outliers**(データセット, ignore_features=['DEPT_NAME', 'MERCHANT', 'TRANS_DT'])

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*jLYtThjhL2rAlfOtZnG0gQ.png)

データセット内のいくつかの列を **ignore_features** パラメータに渡して無視しています。機械学習アルゴリズムの学習に特定のカラムを使用したくない理由はたくさんあるでしょう。

PyCaretでは、データセットから不要なカラムを削除する代わりに、後の分析に必要なカラムを隠すことができます。例えば、今回のケースでは、アルゴリズムの学習に取引日を使用したくないので、**ignore_features.**に渡しています。

PyCaretには10以上のすぐに使える異常検知アルゴリズムがあります。

![](https://cdn-images-1.medium.com/max/2000/1*piuoq_K4B2aiyzOCkDg8MA.png)

デフォルトでは、PyCaretは**K-Nearest Neighbors異常検知**を5%の割合で学習します(つまり、テーブルの全行数の5%を異常値としてフラグを立てます)。デフォルト値は簡単に変更できます。

* fractionの値を変更するには、**get_outliers( ) **関数内の**fraction ***パラメータを使用します。

* モデルタイプを変更するには，**get_outliers()**内の**model**パラメータを使用します。

フラクション0.1の**Isolation Forest**検出器を学習するサンプルコードをご覧ください。

    from **pycaret.anomaly** import *.
    データセット = **get_outliers**(データセット, モデル = 'iforest', fraction = 0.1, ignore_features=['DEPT_NAME', 'MERCHANT', 'TRANS_DT'])

**アウトプット:**

![異常検知結果（Pythonコード実行後）](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![最終出力(Tableをクリックした後)](https://cdn-images-1.medium.com/max/2280/1*dZbf7VmCxkPUcX_p7kKJ4w.png)

元の表に2つの新しい列が付けられています。ラベル（1＝外れ値、0＝外れ値）とスコア（スコアが高いデータポイントは外れ値に分類されます）です。クエリを適用すると、Power BIデータセットに結果が表示されます。

![Power BI Desktopでの結果（クエリ適用後）](https://cdn-images-1.medium.com/max/2894/1*QFJ2DJX_bGSxutOdxNmwEg.png)

# 3. ダッシュボード

Power BIで外れ値のラベルができたら、ダッシュボードで可視化する例を紹介します。

![ダッシュボードの概要ページ](https://cdn-images-1.medium.com/max/2624/1*7qWjee_M6PTrAd0PJdU1yg.png)

![Dashboardの詳細ページ](https://cdn-images-1.medium.com/max/2634/1*4ISkFG8r3LtVJq0P3793Wg.png)

PBIXファイルとデータセットは、弊社の[GitHub](https://github.com/pycaret/powerbi-anomaly-detection)からダウンロードできます。

# 👉 本番での異常検知の実装

以上、Power BIに異常検知を実装する簡単な方法をご紹介しました。しかし、注意しなければならないのは、上記の方法ではPower BIのデータセットが更新されるたびに異常検知器がトレーニングされることです。これが問題になる理由は2つあります。

* 新しいデータでモデルを再トレーニングすると、異常ラベルが変更される可能性があります（以前は異常とラベル付けされていたトランザクションが、現在は異常とは見なされない場合があります）。

* モデルの再トレーニングに毎日何時間も費やしたくない。

本番環境での使用を想定した場合、Power BIに異常検知を実装する別の方法として、Power BI自体でモデルをトレーニングするのではなく、事前にトレーニングしたモデルをPower BIに渡してラベル付けを行う方法があります。

# **モデルを事前にトレーニングする**。

機械学習モデルのトレーニングには、統合開発環境（IDE）やノートブックを使用することができます。この例では、Visual Studio Codeを使用して、異常検知モデルをトレーニングしています。

![Model Training in Visual Studio Code](https://cdn-images-1.medium.com/max/2014/1*zzymbb9ySyl3jeaFQoHxDg.png)

学習されたモデルはpickleファイルとして保存され、異常ラベル（1または0）を生成するためにPower Queryにインポートされます。

![Anomaly Detection Pipeline saved as a file](https://cdn-images-1.medium.com/max/2000/1*fLnTzbd-dTRtqwxmPqI4kw.png)

PyCaretを使ったJupyter notebookへのAnomaly Detectionの実装について詳しく知りたい方は、こちらの2分間のビデオチュートリアルをご覧ください。

 <iframe src="https://medium.com/media/6905eb28ff917a759fe2bed97292795b" frameborder=0></iframe>

# 事前学習したモデルの利用

以下のコードをPythonスクリプトとして実行すると、事前に学習されたモデルからラベルが生成されます。

    from **pycaret.anomaly** import *.
    データセット = **predict_model**('c:/.../anomaly_deployment_13052020, data = dataset)

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*VMSuDzp7FpJgddT-NjTtUQ.png)

これで出力されるものは、上で見たものと同じになります。ただし、事前に学習したモデルを使用した場合は、Power BIデータセットを更新するたびにモデルを再学習するのではなく、同じモデルを使用して新しいデータセットでラベルが生成される点が異なります。

![最終出力（表をクリックした後）](https://cdn-images-1.medium.com/max/2280/1*dZbf7VmCxkPUcX_p7kKJ4w.png)

# **Power BI Serviceで動作させる**。

.pbixファイルをPower BIサービスにアップロードした後、機械学習パイプラインをデータパイプラインにシームレスに統合するために、さらにいくつかのステップが必要です。具体的には以下の通りです。

* **データセットのスケジュール更新を有効にする** - Pythonスクリプトでデータセットを含むワークブックのスケジュール更新を有効にするには、[Configuring scheduled refresh](https://docs.microsoft.com/en-us/power-bi/connect-data/refresh-scheduled-refresh)を参照してください。また、**Personal Gateway**に関する情報もあります。

* **Personal Gateway**のインストール - ファイルが置かれているマシンとPythonがインストールされているマシンに**Personal Gateway**がインストールされている必要があり、Power BIサービスはそのPython環境にアクセスできなければなりません。パーソナルゲートウェイのインストールと設定](https://docs.microsoft.com/en-us/power-bi/connect-data/service-gateway-personal-mode)の詳細については、こちらをご覧ください。

Anomaly Detection についてもっと知りたい方は、[Notebook Tutorial](https://pycaret.org/ano101/)をご覧ください。

# PyCaret 1.0.1 がリリースされます!

PyCaret 1.0.1 がリリースされます！ コミュニティからの圧倒的なサポートとフィードバックを受けています。私たちは、PyCaretの改善と次のリリースに向けて積極的に取り組んでいます。**PyCaret 1.0.1はより大きく、より良くなります**。もし、あなたがフィードバックを共有し、私たちのさらなる改善に役立てたいとお考えでしたら、ウェブサイト上の [fill this form](https://www.pycaret.org/feedback) や、私たちの [GitHub ](https://www.github.com/pycaret/)や [LinkedIn](https://www.linkedin.com/company/pycaret/) ページにコメントを残してください。

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

PyCaretはオープンソースのプロジェクトです。誰でも貢献することができます。貢献したい方は、[オープンイシュー](https://github.com/pycaret/pycaret/issues)にお気軽にご参加ください。Pull Request は dev-1.0.1 ブランチのユニットテスト付きで受け付けています。

PyCaretを気に入っていただけましたら、[GitHub repo](https://www.github.com/pycaret/pycaret)に⭐️をお願いします。

媒体 : [https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn : [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter : [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
