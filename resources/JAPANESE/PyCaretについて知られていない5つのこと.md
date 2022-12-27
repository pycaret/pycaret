
# PyCaretについて知られていない5つのこと
# by Moez Ali

![PyCaretの作者より](https://cdn-images-1.medium.com/max/2000/1*1HEakzOhZRd21FfAT3TyZw.png)

# PyCaret

PyCaretは、Pythonで作られたオープンソースの機械学習ライブラリで、教師付きおよび教師なしの機械学習モデルを **ローコード** 環境で学習・展開することができます。その使いやすさと効率の良さで知られています。

他のオープンソース機械学習ライブラリと比較して、PyCaretは、数百行のコードを数語だけで置き換えることができる代替ローコードライブラリです。

まだPyCaretを使ったことがない方や、もっと知りたい方は、[こちら](https://towardsdatascience.com/announcing-pycaret-an-open-source-low-code-machine-learning-library-in-python-4a1f1aad8d46)から始めるのが良いでしょう。
> "PyCaretを日常的に使っている多くのデータサイエンティストと話した結果、あまり知られていないが非常に強力なPyCaretの5つの機能をショートリストにしました。" - モエズ・アリ

# 👉教師なし実験で「nパラメータ」をチューニングできる

教師なし機械学習では、「nパラメータ」、つまりクラスタリング実験ではクラスタ数、異常検知では異常値の割合、トピックモデリングではトピック数が基本的に重要です。

実験の最終的な目的が、教師なし実験の結果を使って結果を予測（分類または回帰）することである場合、**pycaret.clustering **モジュール**、**pycaret.anomaly **モジュール**、**pycaret.nlp **モジュール****、tune_model()関数が非常に便利になります。

これを理解するために、「[Kiva](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/kiva.csv)」のデータセットを使った例を見てみましょう。

![](https://cdn-images-1.medium.com/max/2000/1*-161ThHhI7lMVHuY4jbsA.png)

これはマイクロバンクのローンデータセットで、各行が借り手とその情報を表しています。en」列は各借り手のローン申請テキストを表し、「status」列は借り手がデフォルトしたかどうかを表しています（default = 1またはno default = 0）。

pycaret.nlpの**tune_model**関数を使用すると、教師付き実験の目標変数に基づいて**num_topics**パラメータを最適化することができます（つまり、最終的な目標変数の予測を改善するために必要な最適なトピック数を予測することができます）。学習用モデルは**estimator**パラメータ（ここでは「xgboost」）で定義できます。この関数は、学習されたトピックモデルと、各イテレーションにおけるスーパーバイズドメトリクスを示すビジュアルを返します。

 <iframe src="https://medium.com/media/7adc412481d38c5d11fc89f4196671a3" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2314/1*RIOVzRCYsA-r-c1Iy7x_5w.png)

# 👉"n_iter "を増やすことで、ハイパーパラメータチューニングの結果を改善することができます。

pycaret.classification**モジュールおよび**pycaret.regression**モジュールの**tune_model**関数は、ハイパーパラメータチューニングのために、あらかじめ定義されたグリッドサーチよりもランダムグリッドサーチを採用しています。ここでは、デフォルトの反復回数を10に設定しています。

tune_modelの結果は、**create_modelで作成したベースモデルの結果よりも、必ずしも改善されない場合があります。**グリッド検索はランダムに行われるため、 **n_iter **パラメータを増やすことでパフォーマンスを向上させることができます。以下に例を示します。

 <iframe src="https://medium.com/media/009f98cee1bc5a231fc1342e08d406b3" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2000/1*LRu2R2f4rXYkOrWVC6ul5A.png)

# 👉setup関数でプログラム的にデータ型を定義することができる

setup **function** を初期化する際に、**データタイプの確認をユーザー入力で求められます。ワークフローの一部としてスクリプトを実行したり、リモートカーネル（Kaggle Notebooksなど）として実行したりする場合には、ユーザー入力ボックスではなく、プログラムでデータタイプを指定することが必要になります。

以下の例では、「[insurance](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/insurance.csv)」データセットを使用しています。

![](https://cdn-images-1.medium.com/max/2000/1*q2WFe3JgZ1SxSkiuuvonKQ.png)

 <iframe src="https://medium.com/media/a96c059e33ee57e12df796357fe19044" frameborder=0></iframe>

の**silent**パラメータは入力を避けるためにTrueに設定され、**categorical_features**パラメータはカテゴリカルカラムの名前を文字列として受け取り、**numeric_features**パラメータは数値カラムの名前を文字列として受け取ります。

# 👉モデル構築時に特定のカラムを無視することができる

データセットの中には、必ずしも削除したいわけではないが、機械学習モデルの学習のために無視したい特徴がある場合が多々あります。例えば、クラスタリングの問題で、クラスタ作成時には特定の特徴を無視したいが、後でクラスタラベルの分析のためにそれらの列が必要になるような場合です。このような場合には、**setup**の**ignore_features**パラメータを使用して、そのような特徴を無視することができます。

以下の例では、クラスタリング実験を行い、**'Country Name'**と**'Indicator Name'**を無視したいとします。

![](https://cdn-images-1.medium.com/max/2000/1*0xcKweKh77A-vgzb5u5_mw.png)

 <iframe src="https://medium.com/media/87c6f8d873c53b758b3ec6e2a588f20e" frameborder=0></iframe>

# 👉2値分類における確率のしきい値%を最適化することができる

分類問題では、**偽陽性**のコストと**偽陰性**のコストが同じになることはほとんどありません。このように、**タイプ1**と**タイプ2**のエラーが異なる影響を与えるようなビジネス上の問題に対してソリューションを最適化する場合、真のポジティブ、真のネガティブ、偽のポジティブ、偽のネガティブのコストを別々に定義するだけで、カスタム損失関数を最適化するための確率しきい値を分類器に設定することができます。デフォルトでは、すべての分類器のしきい値は0.5となっています。

データセット「[クレジット](https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/credit.csv)」を使った例を以下に示します。

 <iframe src="https://medium.com/media/9476e7c2d508b2711631020ceebe583f" frameborder=0></iframe>

![](https://cdn-images-1.medium.com/max/2000/1*oCsUyp91pSJSDdzi-ho6QA.png)

その後、**predict_model **関数の**probability_threshold **パラメータに**0.2 **を渡すことで、陽性クラスを分類するための閾値として0.2を使用することができます。以下の例をご覧ください。

 <iframe src="https://medium.com/media/7670ed065b5f318524592e8b84bdbf54" frameborder=0></iframe>

# PyCaret 2.0.0 is coming!

データサイエンスコミュニティからの圧倒的なサポートとフィードバックを受けています。私たちは、PyCaretの改善に積極的に取り組み、次のリリースに向けて準備を進めています。**PyCaret 2.0.0はより大きく、より良くなります**。もし、あなたがフィードバックを共有し、私たちのさらなる改善を支援したい場合は、ウェブサイト上の[fill this form](https://www.pycaret.org/feedback)や、私たちの[GitHub](https://www.github.com/pycaret/)や[LinkedIn](https://www.linkedin.com/company/pycaret/)ページにコメントを残してください。

LinkedIn](https://www.linkedin.com/company/pycaret/)をフォローしたり、[YouTube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)のチャンネルを購読して、PyCaretについてもっと知りたい方は、こちらをご覧ください。

# 特定のモジュールについて知りたいですか？

最初のリリース1.0.0の時点で、PyCaretは以下のモジュールを使用することができます。以下のリンクをクリックすると、Pythonでのドキュメントや動作例を見ることができます。

[Classification](https://www.pycaret.org/classification)
[回帰
](https://www.pycaret.org/regression)[Clustering](https://www.pycaret.org/clustering)
[異常検知
](https://www.pycaret.org/anomaly-detection)[自然言語処理](https://www.pycaret.org/nlp)
[アソシエーション・ルール・マイニング](https://www.pycaret.org/association-rules)

# こちらもご覧ください。

PyCaret getting started tutorials in Notebook:

[分類](https://www.pycaret.org/clf101)
[回帰](https://www.pycaret.org/reg101)
[Clustering](https://www.pycaret.org/clu101)
[異常検知](https://www.pycaret.org/anom101)
[自然言語処理](https://www.pycaret.org/nlp101)
[アソシエーション・ルール・マイニング](https://www.pycaret.org/arul101)

# Would you like to contribute?

PyCaretはオープンソースのプロジェクトです。誰でも貢献することができます。貢献したい方は、[オープンイシュー](https://github.com/pycaret/pycaret/issues)にお気軽にご参加ください。Pull Request は dev-1.0.1 ブランチのユニットテスト付きで受け付けています。

PyCaretを気に入っていただけましたら、[GitHub repo](https://www.github.com/pycaret/pycaret)に⭐️をお願いします。

媒体です。[https://medium.com/@moez_62905/](https://medium.com/@moez_62905/Machine-learning-in-power-bi-using-pycaret-34307f09394a)

LinkedIn [https://www.linkedin.com/in/profile-moez/](https://www.linkedin.com/in/profile-moez/)

Twitter [https://twitter.com/moezpycaretorg1](https://twitter.com/moezpycaretorg1)
