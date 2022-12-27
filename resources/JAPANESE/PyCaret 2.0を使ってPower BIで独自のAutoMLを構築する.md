
# PyCaret 2.0を使ってPower BIで独自のAutoMLを構築する

# by Moez Ali

![PyCaret - An open source, low-code machine learning library in Python](https://cdn-images-1.medium.com/max/2664/1*Kx9YUt0hWPhU_a6h2vM5qA.png)

# **PyCaret 2.0** (パイキャレット2.0)

先週、機械学習のワークフローを自動化するオープンソースの **ローコード** 機械学習ライブラリである [PyCaret 2.0](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)を発表しました。PyCaret 2.0は、機械学習の実験サイクルを高速化し、データサイエンティストがより効率的かつ生産的になるためのエンド・ツー・エンドの機械学習およびモデル管理ツールです。

この記事では、PyCaretを使って[Power BI](https://powerbi.microsoft.com/en-us/)に機械学習の自動化ソリューションを構築する方法をステップバイステップのチュートリアルで紹介します。PyCaretは、オープンソースで**無料で**使えるPythonライブラリで、Power BIで動作するように作られた幅広い機能を備えています。

この記事を読み終える頃には、Power BIで以下を実装する方法を学ぶことができます。

* Pythonのconda環境を設定し、pycaret==2.0をインストールする。

* 新しく作成した conda 環境を Power BI とリンクする。

* Power BIで最初のAutoMLソリューションを構築し、ダッシュボードにパフォーマンスメトリクスを表示する。

* Power BIでAutoMLソリューションを製品化/展開します。

# Microsoft Power BI

Power BIは、データを視覚化して組織全体でインサイトを共有したり、アプリやウェブサイトに埋め込むことができるビジネス分析ソリューションです。このチュートリアルでは、PyCaretライブラリをPower BIにインポートすることで、[Power BI Desktop](https://powerbi.microsoft.com/en-us/downloads/)を機械学習に使用します。

# 自動機械学習とは？

自動機械学習(AutoML)とは、機械学習の時間のかかる反復作業を自動化するプロセスです。データサイエンティストやアナリストは、モデルの品質を維持しながら、効率的に機械学習モデルを構築することができます。AutoMLソリューションの最終的な目標は、いくつかの性能基準に基づいて最適なモデルを最終的に決定することです。

従来の機械学習モデルの開発プロセスはリソース集約型であり、何十ものモデルを作成して比較するためには、かなりのドメイン知識と時間が必要でした。自動化された機械学習を使えば、生産可能なMLモデルを手に入れるまでの時間を、非常に簡単かつ効率的に加速することができます。

# **PyCaret** はどのように機能しますか？

PyCaretは、教師あり、教師なしの機械学習のためのワークフロー自動化ツールです。PyCaretは6つのモジュールで構成されており、各モジュールには特定の動作を行うための関数が用意されています。各関数は入力を受け取り、出力を返します。ほとんどの場合、出力は学習済みの機械学習モデルです。2回目のリリース時点で利用可能なモジュールは以下の通りです。

* [Classification](https://www.pycaret.org/classification)

* [回帰](https://www.pycaret.org/regression)

* [Clustering](https://www.pycaret.org/clustering)

* [異常検知](https://www.pycaret.org/anomaly-detection)

* [自然言語処理](https://www.pycaret.org/nlp)

* [アソシエーションルールマイニング](https://www.pycaret.org/association-rules)

PyCaretのすべてのモジュールは、データの準備（25以上の必須の前処理技術、膨大な学習済みモデルのコレクションとカスタムモデルのサポート、ハイパーパラメーターの自動調整、モデルの分析と解釈可能性、モデルの自動選択、実験のロギング、簡単なクラウドデプロイメントオプション）をサポートしています。

![https://www.pycaret.org/guide](https://cdn-images-1.medium.com/max/2066/1*wT0m1kx8WjY_P7hrM6KDbA.png)

PyCaretについてもっと知りたい方は、[こちら](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)の公式リリースアナウンスをご覧ください。

Pythonを始めたい方は、[ここをクリック](https://github.com/pycaret/pycaret/tree/master/examples)すると、始めるためのサンプルノートブックのギャラリーを見ることができます。

> "PyCaretは、ビジネスアナリスト、ドメインエキスパート、市民データサイエンティスト、経験豊富なデータサイエンティストのために、無料、オープンソース、ローコードの機械学習ソリューションを提供することで、機械学習と高度な分析の利用を民主化します".

# 始める前に

はじめてPythonを使う場合は、Anaconda Distributionをインストールするのがもっとも簡単な方法です。[ここをクリック](https://www.anaconda.com/distribution/)して、Python 3.7以上のAnaconda Distributionをダウンロードしてください。

![[https://www.anaconda.com/products/individual](https://www.anaconda.com/products/individual)](https://cdn-images-1.medium.com/max/2612/1*sMceDxpwFVHDtdFi528jEg.png)

# 環境のセットアップ

Power BIでPyCaretの機械学習機能を使い始める前に、仮想環境の作成とpycaretのインストールが必要です。これは3つのステップで行います。

[✅](https://fsymbols.com/signs/tick/) **ステップ1 - anaconda環境の作成**。

スタートメニューから**Anaconda Prompt**を開き、以下のコードを実行します。

    conda create --name **myenv** python=3.7

![Anaconda Prompt - Creating an anaconda environment](https://cdn-images-1.medium.com/max/2194/1*2D9jKJPM4eAy1-7lvcLlJQ.png)

[✅](https://fsymbols.com/signs/tick/) **ステップ2 - PyCaret**のインストール

Anaconda Promptで以下のコードを実行します。

    pip install **pycaret==2.0**

を実行します。

インストールには15～20分程度かかる場合があります。インストールに問題がある場合は、[GitHub](https://www.github.com/pycaret/pycaret)のページで既知の問題と解決策を確認してください。

[✅](https://fsymbols.com/signs/tick/)**ステップ3 - Power BIにPythonディレクトリを設定する**。

作成した仮想環境をPower BIと連携させる必要があります。これは、Power BI Desktopのグローバル設定（ファイル → オプション → グローバル → Pythonスクリプト）で行うことができます。Anaconda Environmentはデフォルトでは以下の場所にインストールされます。

C:Users%%**username%%** AppData%%Local%%Continuum%%anaconda3%%envs%%myenv

![File → Options → Global → Python scripting](https://cdn-images-1.medium.com/max/2000/1*zQMKuyEk8LGrOPE-NByjrg.png)

# **👉 Let's get started**

# ビジネスコンテクストの設定

ある保険会社は、入院時の人口統計学的指標と基本的な患者の健康リスク指標を用いて患者の料金をより良く予測することで、キャッシュフロー予測を改善したいと考えています。

![a](https://cdn-images-1.medium.com/max/2000/1*qM1HiWZ_uigwdcZ0_ZL6yA.png)

*([データソース](https://www.kaggle.com/mirichoi0218/insurance#insurance.csv))*

# 目的

データセットに含まれる他のはじめて年齢、性別、BMI、子供、喫煙者、地域）に基づいて患者の料金を予測する、もっとも性能の良い回帰モデルを学習し、選択する。

# 👉 Step 1 - データセットの読み込み

Power BI Desktop → Get Data → Webで、GitHubから直接データセットを読み込むことができます。

データセットへのリンクです。[https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv)

![Power BI Desktop → Get Data → Web](https://cdn-images-1.medium.com/max/2000/1*zZjZzF_TJudoThDCBGK3fQ.png)

Power Queryで複製のデータセットを作成します。

![Power Query → Create a duplicate dataset](https://cdn-images-1.medium.com/max/3436/1*mU8tl4P89WKMC__k6rM-Vw.png)

# 👉 Step 2- AutoMLをPythonスクリプトとして実行する

Power Queryで以下のコードを実行します（Transform → Run Python script）。

    **# import regression モジュール**。
    from pycaret.regression import *

    **# init setup** (セットアップ)
    reg1 = setup(data=dataset, target = 'charges', silent = True, html = False)

    **# モデルの比較 **
    best_model = compare_models()

    **# ベストモデルの最終決定 **
    best = finalize_model(best_model)

    **# ベストモデルの保存 **
    save_model(best, 'c:/users/moezs/best-model-power')

    **# パフォーマンスメトリクスを返す df **
    dataset = pull()

![Power Queryのスクリプト](https://cdn-images-1.medium.com/max/2000/1*FOxy83SH1uy8pFLJT6sa3w.png)

最初の2行のコードは、関連モジュールのインポートとsetup関数の初期化です。セットアップ関数は、機械学習に必要ないくつかの必須ステップを実行します。たとえば、欠損値がある場合はそれをクリーニングし、データをトレーニングとテストに分割し、クロスバリデーション戦略を設定し、メトリクスを定義し、アルゴリズム固有の変換を実行します。

複数のモデルを学習させ、性能を比較・評価する魔法の関数が**compare_models**です。compare_models内で定義された**sort**パラメーターに基づいて最適なモデルを返します。デフォルトでは、回帰の場合は'R2'を、分類の場合は'Accuracy'を使用します。

残りの行では、compare_modelsで得られた最適なモデルを最終的に決定し、ローカルディレクトリにpickleファイルとして保存します。最後の行では、学習されたモデルの詳細とその性能評価をデータフレームで返します。

出力。

![Pythonスクリプトからの出力](https://cdn-images-1.medium.com/max/3822/1*6CSYQDLfQUZeTtYwNllFSw.png)

わずか数行で20以上のモデルが学習され、10倍のクロスバリデーションに基づくパフォーマンス指標が表に示されています。

トップパフォーマンスのモデル **Gradient Boosting Regressor** は、変換パイプライン全体とともに、ローカルディレクトリにpickleファイルとして保存されます。このファイルは、あとで新しいデータセットに予測を生成するために使用できます（以下のステップ3を参照）。

![変換パイプラインとモデルをpickleファイルとして保存](https://cdn-images-1.medium.com/max/2000/1*euQRJQVAVvP2X5ASNWjjOg.png)

PyCaretはモジュール式の自動化という考え方で動いています。そのため、リソースや時間に余裕がある場合は、スクリプトを拡張してハイパーパラメーターチューニングやアンサンブルなどのモデリング技術を実行できます。以下の例をご覧ください。

    **# import regression module**.
    from pycaret.regression import *

    **# init setup** (セットアップ)
    reg1 = setup(data=dataset, target = 'charges', silent = True, html = False)

    **# モデルの比較 **
    top5 = compare_models(n_select = 5)
    results = pull()

    **# top5モデルをチューニングする**
    tuned_top5 = [tune_model(i) for i in top5].

    **# ベストモデルの選択**
    best = automl()

    **# ベストモデルを保存する***。
    save_model(best, 'c:/users/moezs/best-model-power')

    **# パフォーマンスメトリクスを返す df **
    dataset = first_sults

これで、もっともパフォーマンスの高い1つのモデルではなく、上位5つのモデルを返すことができました。そして、上位候補モデルのハイパーパラメーターを調整するためのリスト内包（ループ）を作成し、最後に **automl関数**で最高性能のモデルを1つ選択し、pickleファイルとして保存しています（automl関数は最終的なモデルを返すため、今回は**finalize_model**を使用していないことに注意してください）。

# **サンプルダッシュボード**

サンプルダッシュボードを作成します。PBIXファイルは[ここにアップロード](https://github.com/pycaret/pycaret-powerbi-automl)してあります。

![PyCaret AutoMLの結果を利用して作成したダッシュボード](https://cdn-images-1.medium.com/max/2664/1*Kx9YUt0hWPhU_a6h2vM5qA.png)

# 👉 Step 3 - Deploy Model to generate prediction

最終的なモデルがpickleファイルとして保存されたら、それを使って新しいデータセットの料金を予測することができます。

# **新しいデータセットの読み込み**)

デモのために、同じデータセットを再度読み込み、データセットから「charges」列を削除します。Power QueryでPythonスクリプトとして以下のコードを実行し、予測値を取得します。

    **# regressionモジュールから関数をロードする **。
    from pycaret.regression import load_model, predict_model

    **# モデルを変数にロードする **
    model = load_model('c:/users/moezs/best-model-powerbi')

    **# 予測料金 **
    dataset = predict_model(model, data=dataset)

出力 :

![Power Queryでのpredict_model関数出力](https://cdn-images-1.medium.com/max/3840/1*ZYWjwtu4njS7f7XMp90ofg.png)

# **Power BI Serviceにデプロイする**。

Pythonスクリプトを含むPower BIレポートをサービスに公開すると、オンプレミスのデータゲートウェイを通じてデータがリフレッシュされたときに、これらのスクリプトも実行されます。

これを有効にするには、依存するPythonパッケージを含むPythonランタイムが、パーソナルゲートウェイをホストするマシンにもインストールされていることを確認する必要があります。注：Pythonスクリプトの実行は、複数のユーザーが共有するオンプレミスのデータゲートウェイではサポートされていません。[詳しくはこちら](https://powerbi.microsoft.com/en-us/blog/python-visualizations-in-power-bi-service/)をご覧ください。

このチュートリアルで使用するPBIXファイルは、このGitHubリポジトリにアップロードされています。[https://github.com/pycaret/pycaret-powerbi-automl](https://github.com/pycaret/pycaret-powerbi-automl)

PyCaret 2.0について詳しく知りたい方は、こちらの[お知らせ](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)をご覧ください。

以前にPyCaretを使ったことがある方は、現在のリリースについて[release notes](https://github.com/pycaret/pycaret/releases/tag/2.0)を参照してください。

このPythonの軽量なワークフロー自動化ライブラリを使って実現できることは無限にあります。もしお役に立てましたら、私たちのgithub repoに⭐️をお願いします。

PyCaretについてもっと知りたい方は、[LinkedIn](https://www.linkedin.com/company/pycaret/)や[Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)でフォローしてみてください。

# **You may be interested it:**.

[PyCaretを使ったPower BIでの機械学習](https://towardsdatascience.com/machine-learning-in-power-bi-using-pycaret-34307f09394a)
[Build your first Anomaly Detector in Power BI using PyCaret](https://towardsdatascience.com/build-your-first-anomaly-detector-in-power-bi-using-pycaret-2b41b363244e)
[PyCaretを使ったPower BIでのクラスタリングの実装方法](https://towardsdatascience.com/how-to-implement-clustering-in-power-bi-using-pycaret-4b5e34b1405b)
[PyCaretを使ったPower BIでのトピックモデリング](https://towardsdatascience.com/topic-modeling-in-power-bi-using-pycaret-54422b4e36d6)

# 重要なリンク

[ブログ](https://medium.com/@moez_62905)
[PyCaret 2.0 のリリースノート](https://github.com/pycaret/pycaret/releases/tag/2.0)
[ユーザーガイド / ドキュメント](https://www.pycaret.org/guide)[
](https://github.com/pycaret/pycaret/releases/tag/2.0)[Github](http://www.github.com/pycaret/pycaret) 
[Stackoverflow](https://stackoverflow.com/questions/tagged/pycaret)
[PyCaret のインストール](https://www.pycaret.org/install)
[ノートブックのチュートリアル](https://www.pycaret.org/tutorial)
[Contribute in PyCaret](https://www.pycaret.org/contribute)

# 特定のモジュールについて学びたいですか？

以下のリンクをクリックすると、ドキュメントや動作例を見ることができます。

[Classification](https://www.pycaret.org/classification)
[回帰
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[異常検知
](https://www.pycaret.org/anomaly-detection)[自然言語処理](https://www.pycaret.org/nlp)
[アソシエーション・ルール・マイニング](https://www.pycaret.org/association-rules)
