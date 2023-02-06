
# PyCaretを使ったPower BIでの機械学習

# Power BIに機械学習を数分で実装するためのステップバイステップのチュートリアル

# by Moez Ali

![Machine Learning Meets Business Intelligence](https://cdn-images-1.medium.com/max/2000/1*Q34J2tT_yGrVV0NU38iMig.jpeg)

# **PyCaret 1.0.0**のご紹介

先週、私たちは、**ローコード**環境で機械学習モデルをトレーニングし、デプロイするPythonのオープンソース機械学習ライブラリである[PyCaret](https://www.pycaret.org)を発表しました。[前回の記事](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46)では、PyCaretをJupyter Notebookで使用し、Pythonで機械学習モデルを学習・展開する方法を紹介しました。

この記事では、PyCaretを[Power BI](https://powerbi.microsoft.com/en-us/)に統合する方法を**ステップバイステップ**で紹介します。これにより、アナリストやデータサイエンティストは、追加のライセンスやソフトウェアコストなしに、ダッシュボードやレポートに機械学習のレイヤーを追加することができます。PyCaretは、オープンソースで**無料で**使えるPythonライブラリで、Power BIで動作するように専用に作られた幅広い機能を備えています。

この記事を読み終える頃には、Power BIで以下を実装する方法を学ぶことができます。

* **クラスタリング** - 類似した特徴を持つデータポイントをグループ化します。

* **Anomaly Detection**- データ内の稀な観測値や外れ値を特定します。

* **自然言語処理**- トピックモデリングによってテキストデータを分析します。

* **Association Rule Mining**- データ内の興味深い関係を見つける。

* **分類**- バイナリ（1または0）であるカテゴリクラスラベルを予測する。

* **Regression**- 売上、価格などの連続値を予測する。
  
> "PyCaretは、ビジネスアナリスト、ドメインエキスパート、市民データサイエンティスト、経験豊富なデータサイエンティストのための**フリー、オープンソース、ローコード**の機械学習ソリューションを提供することで、機械学習と高度な分析の利用を民主化しています".

# Microsoft Power BI

Power BIは、データを可視化して組織全体でインサイトを共有したり、アプリやウェブサイトに埋め込んだりできるビジネス分析ソリューションです。このチュートリアルでは、[Power BI Desktop](https://powerbi.microsoft.com/en-us/downloads/)を使って、PyCaretライブラリをPower BIにインポートして機械学習を行います。

# 始める前に

Pythonを使ったことがある方は、すでにAnaconda Distributionがインストールされていると思います。もしそうでなければ、[ここをクリック](https://www.anaconda.com/distribution/)して、Python 3.7以上のAnaconda Distributionをダウンロードしてください。

![[https://www.anaconda.com/distribution/](https://www.anaconda.com/distribution/)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# 環境のセットアップ

Power BIでPyCaretの機械学習機能を使い始める前に、仮想環境を作成し、pycaretをインストールする必要があります。これは3つのステップで行います。

[✅](https://fsymbols.com/signs/tick/) **ステップ1 - anaconda環境の作成**。

スタートメニューから**Anaconda Prompt**を開き、以下のコードを実行します。

    conda create --name **myenv** python=3.6

![Anaconda Prompt - Creating an environment](https://cdn-images-1.medium.com/max/2198/1*Yv-Ee99UJXCW2iTL1HUr5Q.png)

[✅](https://fsymbols.com/signs/tick/) **Step 2 - PyCaret**のインストール

Anaconda Promptで以下のコードを実行します。

    conda activate **myenv** を実行します。
    pip install pycaret

インストールには10～15分程度かかる場合があります。

[✅](https://fsymbols.com/signs/tick/)**ステップ3 - Power BI**のPythonディレクトリの設定

作成した仮想環境は、Power BIとリンクさせる必要があります。これは、Power BIデスクトップのグローバル設定（ファイル → オプション → グローバル → Pythonスクリプト）で行うことができます。Anaconda Environmentはデフォルトでは以下の場所にインストールされます。

C:Users%% **username%%** AppData%%Local%%Continuum%%anaconda3%%envs%%myenv

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*zQMKuyEk8LGrOPE-NByjrg.png)

# 📘 例1 - Power BIでのクラスタリング

クラスタリングは、類似した特徴を持つデータポイントをグループ化する機械学習の手法です。これらのグループ化は、データの探索、パターンの特定、データのサブセットの分析に役立ちます。クラスタリングの一般的なビジネスユースケースは以下の通りです。

マーケティングのための顧客セグメンテーション。

プロモーションやディスカウントのための顧客の購買行動分析。

COVID-19のような伝染病の発生におけるジオクラスターの特定。

このチュートリアルでは、PyCaretの[github repository](https://github.com/pycaret/pycaret/blob/master/datasets/jewellery.csv)で公開されている**'jewelry.csv'**ファイルを使用します。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

**csvファイルへのリンク：**[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/jewellery.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/jewellery.csv)

![Power BI Desktop → データを取得 → その他 → Web](https://cdn-images-1.medium.com/max/2000/1*MdUeug0LSZu451-fBI5J_Q.png)

![*Sample data points from jewellery.csv*](https://cdn-images-1.medium.com/max/2000/1*XhXJjUHpEqOc7-RQ1fWoYQ.png)

# **K-Meansクラスタリング** (K-Means Clustering)

クラスタリングモデルを学習するために、Power Query EditorでPythonスクリプトを実行します（Power Query Editor → Transform → Run python script）。

![Power Query Editorのリボン](https://cdn-images-1.medium.com/max/2000/1*F18LNIkoWtAFr4P80J-U8Q.png)

次のコードをPythonスクリプトとして実行します。

    from **pycaret.clustering **import *****
    データセット = **get_clusters**(data = dataset)

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*nYqJWQM6NI3q3tLJXIVxtg.png)

# **アウトプット:**

![Clustering Results (after execution of code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![Final Output (after clicking on Table)](https://cdn-images-1.medium.com/max/2000/1*PXWUtrYrNikCRDqhn_TgDw.png)

元のテーブルに、ラベルを含んだ新しい列**'Cluster' **が付けられます。

クエリを適用すると（Power Query Editor → Home → Close & Apply）、Power BIでクラスターを可視化する方法は以下の通りです。

![](https://cdn-images-1.medium.com/max/2000/1*8im-qPdXXBblPD7jiodQpg.png)

デフォルトでは、PyCaretは4つのクラスタを持つ**K-Means**クラスタリングモデルを学習します(*つまり、テーブルのすべてのデータポイントは4つのグループに分類されます*)。デフォルト値は簡単に変更できます。

* クラスターの数を変更するには、**get_clusters( )**関数内の**num_clusters**パラメーターを使用します。

* モデルタイプを変更するには、**get_clusters()**の**model**パラメーターを使用します。

以下は、6クラスターのK-Modesモデルを学習するコード例です。

    from **pycaret.clustering **import *.
    データセット = **get_clusters**(データセット, モデル = 'kmodes', num_clusters = 6)

PyCaretには9つのすぐに使えるクラスタリングアルゴリズムがあります。

![あ](https://cdn-images-1.medium.com/max/2000/1*Wdy201wGxmV3NwS9lzHwsA.png)

クラスタリングモデルを学習するために必要な [missing value imputation](https://pycaret.org/missing-values/) (テーブルに欠損値や **null**value がある場合), [normalization](https://www.pycaret.org/normalization), [one-hot-encoding](https://pycaret.org/one-hot-encoding/) などの前処理は、クラスタリングモデルを学習する前にすべて自動的に行われます。PyCaretの前処理機能については[こちら](https://www.pycaret.org/preprocessing)を参照してください。

💡 この例では、**get_clusters()**関数を使って、元のテーブルのクラスタラベルを割り当てています。クエリが更新されるたびに、クラスターが再計算されます。別の実装方法としては、PythonやPower BIで**事前に学習したモデル**を使って**predict_model()** 関数を使ってクラスターラベルを予測する方法があります（*Power BI環境で機械学習モデルを学習する方法については、以下の例5を参照してください）。

💡 Jupyter Notebookを使ってPythonでクラスタリングモデルを学習する方法を知りたい方は、[Clustering 101 Beginner's Tutorial](https://www.pycaret.org/clu101)をご覧ください。*(コーディングの知識は必要ありません).*

# 📘 例2 - Power BIでの異常検知

異常検知は、テーブルの中で大部分の行と大きく異なる行をチェックすることで、**希少なアイテム**、**イベント**、**または観測**を識別するために使用される機械学習技術です。一般的に、異常な項目は、銀行詐欺、構造的欠陥、医療上の問題、エラーなど、何らかの問題につながります。異常検知の一般的なビジネスユースケースは以下の通りです。

金融データを使った不正行為の検知（クレジットカード、保険など）。

侵入検知(システムセキュリティ、マルウェア)や、ネットワークトラフィックの急上昇・急降下の監視。

✔ データセットの多変量異常値の識別。

このチュートリアルでは、PyCaretの[githubリポジトリ](https://github.com/pycaret/pycaret/blob/master/datasets/anomaly.csv)で入手できる **'anomaly.csv'** ファイルを使用します。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)となります。

**csvファイル**へのリンクです。[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/anomaly.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/anomaly.csv)

![*Sample data points from anomaly.csv*](https://cdn-images-1.medium.com/max/2476/1*M0uBBbcEYizdZgpeKlftlQ.png)

# K-Nearest Neighbors Anomaly Detector

クラスタリングと同様に、Power Query EditorからPythonスクリプトを実行して（Transform → Run python script）、異常検知モデルを学習させます。以下のコードをPythonスクリプトとして実行します。

    from **pycaret.anomaly **import *****
    データセット = **get_outliers**(data = dataset)

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*re7Oj-bPUHok7pCbmeWFuw.png)

# **アウトプット:**

![異常検知結果（コード実行後）](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![最終出力(表をクリックした後)](https://cdn-images-1.medium.com/max/2002/1*J7_5ZAM7dFNVnMcgxV_N1A.png)

元の表に2つの新しい列が付けられています。Label（1＝外れ値、0＝外れ値）とScore（スコアが高いデータポイントは外れ値に分類されます）です。

クエリを適用すると、異常検知の結果をPower BIで可視化する方法は次のとおりです。

![l](https://cdn-images-1.medium.com/max/2000/1*tfn6W5vV1pUE11hTPCzdpA.png)

デフォルトでは、PyCaretは**K-Nearest Neighbors異常検知**を5%の割合で学習します(つまり、テーブルの全行数の5%が異常値としてフラグを立てられます)。デフォルト値は簡単に変更できます。

* **get_outliers()** 関数内の**fraction**パラメーターを使用することで、端数の値を変更できます。

* モデルタイプを変更するには、**get_outliers()** 内の **model** parameterを使用します。

以下のコードでは、0.1 fractionの**Isolation Forest**モデルを学習しています。

    from **pycaret.anomaly **import *.
    データセット = **get_outliers**(データセット, モデル = 'iforest', fraction = 0.1)

PyCaretには10以上のすぐに使える異常検知アルゴリズムがあります。

![](https://cdn-images-1.medium.com/max/2000/1*piuoq_K4B2aiyzOCkDg8MA.png)

[欠損値の入力](https://pycaret.org/missing-values/) (テーブルに欠損値や**null**値がある場合)、[正規化](https://www.pycaret.org/normalization)、[ワンショットエンコーディング](https://pycaret.org/one-hot-encoding/)など、異常検知モデルの学習に必要な前処理がすべて自動的に実行されます。PyCaretの前処理機能については[こちら](https://www.pycaret.org/preprocessing)を参照してください。

この例では、**get_outliers()** 関数を使用して、異常値のラベルとスコアを割り当てて分析しています。クエリが更新されるたびに、外れ値が再計算されます。別の実装方法としては、PythonやPower BIで事前に学習したモデルを使って、**predict_model()** 関数を使って外れ値を予測する方法があります（*Power BI環境で機械学習モデルを学習する方法については、以下の例5を参照してください）。

💡 Jupyter Notebookを使ってPythonで異常検知器を学習する方法を知りたい方は、【異常検知101初心者向けチュートリアル】(https://www.pycaret.org/ano101)をご覧ください。*(コーディングの知識は必要ありません).*

# 📘 例3 - 自然言語処理

テキストデータの分析にはいくつかの手法が用いられますが、その中でも **トピックモデル** はよく知られています。トピックモデルとは、ドキュメントのコレクションの中から抽象的なトピックを発見するための統計モデルの一種です。トピックモデリングは、テキストデータの中に隠された意味構造を発見するためのテキストマイニングツールとしてよく使われています。

このチュートリアルでは、PyCaretの[githubリポジトリ](https://github.com/pycaret/pycaret/blob/master/datasets/kiva.csv)で公開されている **'kiva.csv'** ファイルを使用します。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

**csvファイルへのリンクです。**[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv)

# **レーテント・ディリクレ・アロケーション** （Latent Dirichlet Allocation

以下のコードをPythonスクリプトとしてPower Query Editorで実行します。

    from **pycaret.nlp **import *****
    データセット = **get_topics**(データ = データセット, テキスト = 'en')

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*QNaOFbKVJtkG6TjH-z0nxw.png)

**'en'** は、テーブル **'kiva'** のテキストを含む列の名前です。

# 出力

![Topic Modeling Results (after execution of code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![最終出力(表をクリックした後)](https://cdn-images-1.medium.com/max/2536/1*kP9luTZMmeo7-uEI1lYKlQ.png)が表示されます。

コードが実行されると、トピックの重みと支配的なトピックを示す新しい列が元のテーブルに追加されます。トピック モデルの出力を Power BI で視覚化する方法はたくさんあります。以下の例をご覧ください。

![](https://cdn-images-1.medium.com/max/2000/1*yZHDO-9UXZ3L1lFBXMMCPg.png)

デフォルトでは、PyCaretは4つのトピックを持つLatent Dirichlet Allocationモデルを学習します。デフォルト値は簡単に変更できます。

* トピックの数を変更するには、**get_topics()** 関数内の**num_topics** パラメータを使用します。

* モデルタイプを変更するには、**get_topics()** 関数内の**model**パラメーターを使用します。

**非負行列因子分解モデル**を10個のトピックで学習させるサンプルコードをご覧ください。

    from **pycaret.nlp **import *.
    データセット = **get_topics**(データセット, 'en', モデル = 'nmf', num_topics = 10)

PyCaretには、トピックモデリングのための以下のアルゴリズムが用意されています。

![a](https://cdn-images-1.medium.com/max/2000/1*YhRd9GgWw1kblnJezqZd5w.png)

# 📘 例4- Power BIでのアソシエーションルールマイニング

Association Rule Mining **** は、データベース内の変数間の興味深い関係を発見するための**ルールベースの機械学習**手法です。これは、面白さの尺度を使用して強力なルールを識別することを目的としています。アソシエーションルールマイニングの一般的なビジネスユースケースは以下の通りです。

よく一緒に買われる商品を把握するためのマーケットバスケット分析。

医療診断では、医師が要因と症状から病気の発生確率を判断するのに役立ちます。

このチュートリアルでは、PyCaretの[github repository](https://github.com/pycaret/pycaret/blob/master/datasets/france.csv)にある**'france.csv'**ファイルを使用します。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

**csvファイルへのリンクです。**[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/france.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/france.csv)

![*Sample data points from france.csv*](https://cdn-images-1.medium.com/max/2484/1*2S-OwdafFh30hWTzFDC_WQ.png)

# Apriori アルゴリズム

全てのPyCaret関数は、Power Query EditorでPythonスクリプトとして実行されることは、もうお分かりだと思います（Transform → Run python script）。以下のコードを実行して、Aprioriアルゴリズムを用いた連想ルールモデルを学習します。

    from **pycaret.arules** import *.
    dataset = **get_rules**(Dataset, transaction_id = 'InvoiceNo', item_id = 'Description')

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*c2QmWam_1008OCEf0Ct46w.png)

**'InvoiceNo'** はトランザクションIDを含む列で、**'Description'** は対象となる変数（製品名）を含みます。

# **Output:**

![Association Rule Mining Results (after execution of code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![Final Output (after clicking on Table)](https://cdn-images-1.medium.com/max/2518/1*H4rGqsxDtJyVu24yc_UWHw.png)

先行詞と後続詞を、サポート、コンフィデンス、リフトなどの関連指標とともに表にして返します。PyCaretでのAssociation Rules Miningの詳細は[こちら](https://www.pycaret.org/association-rule)をご覧ください。

# 📘 例5 - Power BIでの分類

分類は、カテゴライズされた**クラスラベル**（バイナリ変数としても知られる）を予測するために使用される教師付き機械学習技術です。分類の一般的なビジネスユースケースは以下の通りです。

顧客のローン/クレジットカードのデフォルトを予測する。

顧客の解約を予測する（顧客が残るか離れるか）。

患者の転帰の予測（患者が病気かどうか）。

このチュートリアルでは、PyCaretの[github repository](https://github.com/pycaret/pycaret/blob/master/datasets/employee.csv)にある **'employee.csv'** ファイルを使用します。データの読み込みには、Webコネクタを使用します。(Power BI Desktop → Get Data → From Web)を実行します。

**csvファイルへのリンクです。** [https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/employee.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/employee.csv)

**目的**テーブル **'employee'** には、会社に在籍していた期間、月の平均労働時間、昇進履歴、部署など、会社の15,000人のアクティブな従業員の情報が含まれています。これらの列（機械学習の用語では*features*と呼ばれる）に基づいて、目的は、従業員が会社を辞めるかどうかを予測することであり、これは列**'left'**（1はYes、0はNo）で表される。

分類は、クラスタリングや異常検知、NLPのような教師なし機械学習とは異なり、 **教師あり** の技術であるため、2つのパートに分けて実施します。

# Part 1: Power BIで分類モデルを学習する**

まず、Power Query Editorで、モデルの学習に使用するテーブル **'employee'** の複製を作成します。

![Power Query Editor → 右クリック 'employee' → Duplicate](https://cdn-images-1.medium.com/max/2760/1*9t8FyRshmdBqzONMgMRQcQ.png)

新たに作成した複製テーブル **'employee (model training)'** で以下のコードを実行し、分類モデルを学習させます。

    # import classification module and setup environment

    from **pycaret.classification **import *****
    clf1 = **setup**(dataset, target = 'left', silent = True)

    # xgboostモデルの学習と保存

    xgboost = **create_model**('xgboost', verbose = False)
    final_xgboost = **finalize_model**(xgboost)
    **save_model**(final_xgboost, 'C:/Users/*username*/xgboost_powerbi')

![Power Query Editor (Transform → Run python script)](https://cdn-images-1.medium.com/max/2000/1*0qLtTngg_uI31JTSPLNSiQ.png)

# 出力

このスクリプトの出力は、定義された場所に保存された**pickleファイル**になります。このpickleファイルには、データ変換パイプライン全体と学習済みモデルオブジェクトが含まれています。

💡 Power BIではなく、Jupyter notebookでモデルを学習することも可能です。この場合、Power BIは、Jupyter notebookで事前に学習したモデルをPickleファイルとしてPower BIにインポートして、フロントエンドで予測を生成するためにのみ使用されます（以下のPart 2に従ってください）。PythonでPyCaretを使う方法については、[こちら](https://www.pycaret.org/tutorial)をご覧ください。

💡 Jupyter Notebookを使ってPythonで分類モデルを学習する方法を知りたい方は、[Binary Classification 101 Beginner's Tutorial](https://www.pycaret.org/clf101)をご覧ください。*(no coding background needed).*

PyCaretには18種類の分類アルゴリズムが用意されています。

![](https://cdn-images-1.medium.com/max/2000/1*hvcdSTqA6Qla7YlWMkBmhA.png)

# Part 2: 学習したモデルを使った予測の生成

では、元の **'employee'** table に対して学習したモデルを使って、その社員が会社を辞めるかどうか（1か0か）とその確率%を予測してみましょう。以下のコードをpythonスクリプトとして実行し、予測値を生成します。

    from **pycaret.classification** import *****
    xgboost = **load_model**('c:/users/*username*/xgboost_powerbi')
    データセット = **predict_model**(xgboost, data = dataset)

# 出力

![Classification Predictions (after execution of code)](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![最終出力（表をクリックした後）](https://cdn-images-1.medium.com/max/2482/1*9Ib1KC_9MTYEV_xd8fHExQ.png)

元のテーブルに2つの新しい列が追加されています。Label'**列は予測を示し、**Score'**列は結果の確率です。

この例では、モデルのトレーニングに使用したのと同じデータを使って予測を行いました。これはデモ目的のみです。実際の設定では、**'Left'** 列は実際の結果であり、予測の時点では不明です。

このチュートリアルでは、**Extreme Gradient Boosting** **('xgboost')**モデルを学習し、それを使って予測を行います。これは簡単にするためだけに行ったものです。実際には、PyCaretを使用して、あらゆるタイプのモデルやモデルの連鎖を予測することができます。

PyCaretの **predict_model()** 関数は、PyCaretを使って作成されたpickleファイルとシームレスに動作します。pickleファイルには、変換パイプライン全体と学習済みモデルオブジェクトが含まれています。[predict_model()関数の詳細はこちら](https://www.pycaret.org/predict-model)をご参照ください。

💡 [missing value imputation](https://pycaret.org/missing-values/) (テーブルに欠損や *null *value がある場合), [one-hot-encoding](https://pycaret.org/one-hot-encoding/), [target encoding](https://www.pycaret.org/one-hot-encoding) など、分類モデルの学習に必要なすべての前処理が、モデルの学習前に自動的に実行されます。PyCaretの前処理機能については、[こちら](https://www.pycaret.org/preprocessing)をご覧ください。

# 📘 Example 6- Regression in Power BI

**回帰（Regression）** は、過去のデータとそれに対応する過去の結果が与えられたときに、連続的な結果を最良の方法で予測するために使用される教師付き機械学習技術です。Yes or No (1 or 0)のような2値の結果を予測する分類とは異なり、回帰は売上、価格、数量などの連続値を予測するために使用されます。

このチュートリアルでは、pycaretの[github repository](https://github.com/pycaret/pycaret/blob/master/datasets/boston.csv)にある **'boston.csv'** ファイルを使用します。データの読み込みは、Webコネクタを使って行います。(Power BI Desktop → Get Data → From Web)を実行します。

**csvファイルへのリンクです。**
[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/boston.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/boston.csv)

**目的**テーブル **'boston'** には、平均部屋数、固定資産税率、人口など、ボストンにある506軒の家の情報が含まれています。これらの列（機械学習の用語では*features*とも呼ばれる）に基づいて、**'medv'** 列で表される家の価値の中央値を予測することが目的です。

# Part 1: Power BIで回帰モデルを学習する

最初に、Power Query Editorでモデルの学習に使用する **'boston'** テーブルの複製を作成します。

複製した新しいテーブルで以下のコードをpythonスクリプトとして実行します。

    # import regression module and setup environment

    from **pycaret.regression **import *****
    clf1 = **setup**(Dataset, target = 'medv', silent = True)

    # catboostモデルの学習と保存

    catboost = **create_model**('catboost', verbose = False)
    final_catboost = **finalize_model**(catboost)
    **save_model**(final_catboost, 'C:/Users/*username*/catboost_powerbi')

# 出力

このスクリプトの出力は、指定した場所に保存された**pickleファイル**になります。このpickleファイルには、データ変換パイプライン全体と、学習済みモデルオブジェクトが含まれています。

PyCaretには20種類以上の回帰アルゴリズムが用意されています。

![](https://cdn-images-1.medium.com/max/2000/1*2xlKljU-TjJlr7PuUzRRyA.png)

# Part 2: 学習したモデルを使って予測値を生成する

それでは、学習したモデルを使って、住宅の中央値を予測してみましょう。元のテーブル **'boston'** に以下のコードをpythonスクリプトとして実行します。

    from **pycaret.classification** import *****
    xgboost = **load_model**('c:/users/*username*/xgboost_powerbi')
    データセット = **predict_model**(xgboost, data = dataset)

# 出力

![回帰予測（コード実行後）](https://cdn-images-1.medium.com/max/2000/1*RCYtFO6XDGI2-qbZdYeMfQ.png)

![最終出力（表をクリックした後）](https://cdn-images-1.medium.com/max/2408/1*0A1cf_nsj2SULtNEjEu4tA.png)

予測値を含む新しい列 **'Label'** が元のテーブルに追加されます。

この例では、モデルのトレーニングに使用したのと同じデータを使って予測を行いました。これはデモ目的のみです。実際の設定では、**'medv'** 列は実際の結果であり、予測時には不明です。

💡 回帰モデルの学習に必要な前処理として、[欠損値インピュテーション](https://pycaret.org/missing-values/)(テーブルに欠損値や*null *値がある場合)、[ワンショットエンコーディング](https://pycaret.org/one-hot-encoding/)、[ターゲット変換](https://pycaret.org/transform-target/)などの作業が、モデル学習前に自動的に行われます。[PyCaretの前処理機能についてはこちら](https://www.pycaret.org/preprocessing)を参照してください。

# 次のチュートリアル

次回の**Machine Learning in Power BI using PyCaret**シリーズでは、PyCaretの高度な前処理機能について、より深く掘り下げて解説します。また、Power BIで機械学習ソリューションを生産化し、Power BIのフロントエンドで[PyCaret](https://www.pycaret.org)のパワーを活用する方法を見ていきます。

この内容をもっと知りたい方は、ぜひお付き合いください。

[Linkedin](https://www.linkedin.com/company/pycaret/)のページでフォローし、[Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)のチャンネルを購読してください。

# 合わせてご覧ください

初心者レベルのPythonノートブック。

[Clustering](https://www.pycaret.org/clu101)
[異常検知](https://www.pycaret.org/anom101)
[自然言語処理](https://www.pycaret.org/nlp101)
[アソシエーションルールマイニング](https://www.pycaret.org/arul101)
[回帰](https://www.pycaret.org/reg101)
[分類](https://www.pycaret.org/clf101)

# What's in the development pipeline?

私たちは、PyCaretの改良に積極的に取り組んでいます。今後の開発パイプラインには、新しい**時系列予測**モジュール、**TensorFlowとの統合、**PyCaretのスケーラビリティの大幅な改善が含まれています。ご意見やご感想をお寄せいただける場合は、ウェブサイトの [fill this form](https://www.pycaret.org/feedback)、または [Github](https://www.github.com/pycaret/)や [LinkedIn](https://www.linkedin.com/company/pycaret/)のページにコメントをお寄せください。

# 特定のモジュールについて知りたいですか？

最初のリリース1.0.0の時点で、PyCaretは以下のモジュールを使用することができます。以下のリンクをクリックすると、Pythonでのドキュメントと動作例を見ることができます。

[Classification](https://www.pycaret.org/classification)
[回帰
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[異常検知
](https://www.pycaret.org/anomaly-detection)[自然言語処理](https://www.pycaret.org/nlp)
[アソシエーションルールマイニング](https://www.pycaret.org/association-rules)

# 重要なリンク

[ユーザーガイド/ドキュメント](https://www.pycaret.org/guide)
[Github リポジトリ
](https://www.github.com/pycaret/pycaret)[PyCaret のインストール](https://www.pycaret.org/install)
[ノートブックチュートリアル](https://www.pycaret.org/tutorial)
[PyCaret での貢献](https://www.pycaret.org/contribute)

PyCaretを気に入っていただけましたら、[github repo](https://www.github.com/pycaret/pycaret)に⭐️をお願いします。

Mediumで私をフォローしてください。[https://medium.com/@moez_62905/](https://medium.com/@moez_62905/machine-learning-in-power-bi-using-pycaret-34307f09394a)
