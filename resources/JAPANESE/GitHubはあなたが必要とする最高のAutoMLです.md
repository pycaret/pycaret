
# GitHubはあなたが必要とする最高のAutoMLです

# by Moez Ali

![PyCaret - An open source, low-code machine learning library in Python!](https://cdn-images-1.medium.com/max/2000/1*Qe1H5nFp506CKQJto0XU9A.png)

いつからGitHubが自動機械学習のビジネスに参入したのか、不思議に思われるかもしれません。そうではありません。しかし、自分で作ったAutoMLソフトウェアをテストするために GitHub を使うことができます。このチュートリアルでは、独自の自動機械学習ソフトウェアを構築してコンテナー化し、Dockerコンテナーを使ってGitHubでテストする方法を紹介します。

オープンソースでローコードのPython製機械学習ライブラリであるPyCaret 2.0を使用して、シンプルなAutoMLソリューションを開発し、GitHubアクションを使用してDockerコンテナーとしてデプロイします。PyCaretについてはじめて知ったという方は、PyCaret 2.0の公式アナウンス[こちら](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)や、詳細なリリースノート[こちら](https://github.com/pycaret/pycaret/releases/tag/2.0)をご覧ください。

# 👉 このチュートリアルの学習目標

* 自動機械学習とは何かを理解し、PyCaret 2.0 を用いて簡単な AutoML ソフトウェアを構築することができます。

* コンテナーとは何かを理解し、AutoMLソリューションをDockerコンテナーとしてデプロイする方法を理解する。

* GitHubアクションとは何か、AutoMLのテストにどのように使用できるか。

# Automated Machine Learningとは？

Automated Machine Learning（AutoML）とは、機械学習の時間のかかる反復作業を自動化するプロセスです。これにより、データサイエンティストやアナリストは、モデルの品質を維持しつつ、効率的に機械学習モデルを構築できます。AutoMLソフトウェアの最終目標は、いくつかの性能基準に基づいて最適なモデルを最終的に決定することです。

従来の機械学習モデルの開発プロセスはリソース集約型であり、何十ものモデルを作成して比較するためには、かなりのドメイン知識と時間が必要でした。自動機械学習を使えば、本番で使えるMLモデルの開発にかかる時間を非常に簡単かつ効率的に加速することができます。

AutoMLソフトウェアは、有料のものからオープンソースのものまで多数存在しています。そのほとんどが、同じ変換とベースアルゴリズムを使用しています。そのため、これらのソフトウェアで学習されたモデルの品質や性能はほぼ同じです。

有料のAutoMLソフトウェア・アズ・ア・サービスは非常に高価であり、何十ものユースケースを持っていなければ、経済的に不可能である。マネージドな機械学習のサービスプラットフォームは、比較的安価ですが、使い勝手が悪く、特定のプラットフォームの知識が必要です。

数あるオープンソースのAutoMLライブラリの中でも、PyCaretは比較的新しいライブラリであり、機械学習に対する独自のローコードアプローチを持っています。PyCaretのデザインと機能はシンプルで、人間に優しく、直感的です。PyCaretは、短期間で世界中の10万人以上のデータサイエンティストに採用され、開発者のコミュニティも拡大しています。

# PyCaretはどのように動作しますか？

PyCaretは、教師あり、教師なしの機械学習のためのワークフロー自動化ツールです。6つのモジュールで構成されており、各モジュールには特定の動作を行うための関数が用意されています。各関数は、入力を受け取り、出力を返します。ほとんどの場合、出力は学習された機械学習モデルです。2回目のリリース時点で利用可能なモジュールは以下の通りです。

* [Classification](https://www.pycaret.org/classification)

* [回帰](https://www.pycaret.org/regression)

* [Clustering](https://www.pycaret.org/clustering)

* [異常検知](https://www.pycaret.org/anomaly-detection)

* [自然言語処理](https://www.pycaret.org/nlp)

* [アソシエーションルールマイニング](https://www.pycaret.org/association-rules)

PyCaretの全てのモジュールは、データの準備（25以上の必須の前処理技術、膨大な学習済みモデルのコレクションとカスタムモデルのサポート、ハイパーパラメータの自動調整、モデルの分析と解釈可能性、モデルの自動選択、実験のロギング、簡単なクラウドデプロイメントオプション）をサポートしています。

![[https://www.pycaret.org/guide](https://www.pycaret.org/guide)](https://cdn-images-1.medium.com/max/2066/1*wT0m1kx8WjY_P7hrM6KDbA.png)

PyCaretについてもっと知りたい方は、[こちら](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)の公式リリースアナウンスをご覧ください。

Pythonを始めたい方は、[ここをクリック](https://github.com/pycaret/pycaret/tree/master/examples)すると、始めるためのノートブックの例のギャラリーを見ることができます。

# 👉 始める前に

AutoMLソフトウェアを作り始める前に、以下の用語を理解しておきましょう。現時点で必要なのは、このチュートリアルで使用しているこれらのツールや用語に関する基本的な理論的知識です。もっと詳しく知りたい方は、このチュートリアルの最後にリンクがありますので、後で調べてみてください。

# **Container** (コンテナーー)

**コンテナーー**は、機械学習**アプリケーションの精度、性能、効率を最大化するために、異なる環境に迅速に展開できるポータブルで一貫性のある環境を提供します。環境には、ランタイム言語（例：Python）、すべてのライブラリ、アプリケーションの依存関係が含まれます。

# **Docker** (ドッカー)

Docker社は、ユーザーがコンテナーを構築、実行、管理するためのソフトウェア（Dockerとも呼ばれる）を提供する企業です。Dockerのコンテナーは最も一般的なものですが、[LxD](https://linuxcontainers.org/lxd/introduction/)や[LXC](https://linuxcontainers.org/)のような、あまり有名ではない*代替手段*もコンテナーソリューションを提供しています。

# GitHub

[GitHub](https://www.github.com/)は、コードをホスト、管理、制御するために使用されるクラウドベースのサービスです。あなたが大規模なチームで働いていて、複数の人(数百人)が同じコードベースに変更を加えていることを想像してみてください。PyCaret自体が、何百人ものコミュニティ開発者が継続的にソースコードに貢献しているオープンソースプロジェクトの一例です。まだGitHubを使ったことがない方は、[sign up](https://github.com/join)で無料のアカウントを取得できます。

# **GitHub Actions** (ギットハブ・アクション)

GitHub Actionsは、ソフトウェア開発のワークフローを自動化するのに役立ちます。コードを保存したり、プルリクエストや課題でコラボレーションしたりするのと同じ場所での作業です。アクションと呼ばれる個々のタスクを記述し、それらを組み合わせてカスタムワークフローを作成することができます。ワークフローとは、GitHub 上の任意のコードプロジェクトをビルド、テスト、パッケージ、リリース、デプロイするために、リポジトリに設定できるカスタムの自動プロセスのことです。

# 👉 始めよう

# 目的

データセットに含まれる他の変数（年齢、性別、BMI、子供、喫煙者、地域）に基づいて患者の料金を予測する、最もパフォーマンスの高い回帰モデルを学習し、選択する。

# 👉 **ステップ1 - app.py** の開発

これはAutoMLのメインファイルであり、Dockerfileのエントリーポイントでもあります（後述のステップ2）。以前にPyCaretを使ったことがある方は、このコードを見れば一目瞭然でしょう。

 <iframe src="https://medium.com/media/4f63b152703a63c4886b9d11e22bad00" frameborder=0></iframe>

最初の5行は、環境からのライブラリと変数のインポートです。次の3行目は、データを*pandas* dataframeとして読み込むためのものです。12行目から15行目までは、環境変数に基づいて関連するモジュールをインポートし、17行目以降は、環境の初期化、ベースモデルの比較、最も性能の良いモデルをデバイスに保存するためのPyCaretの機能についてです。最後の行では、実験ログをcsvファイルでダウンロードしています。

# 👉 Step 2- Dockerfileの作成

Dockerfileとは、数行の命令が書かれた単なるファイルで、プロジェクトフォルダに「Dockerfile」という名前で保存されます（大文字と小文字が区別され、拡張子はありません）。

Dockerファイルを別の角度から考えると、それはあなたが自分のキッチンで考案したレシピのようなものです。そのようなレシピを他の人と共有し、レシピに書かれているのと全く同じ指示に従えば、他の人も同じ料理を同じ品質で再現することができます。同様に、Dockerファイルを他の人と共有すれば、そのDockerファイルに基づいてイメージを作成し、コンテナーを実行することができます。

今回のプロジェクトのDockerファイルは6行で構成されたシンプルなものです。以下をご覧ください。

 <iframe src="https://medium.com/media/b7c21d84b56e85ebdde61bbe7ea6ed55" frameborder="0"> </iframe>。

Dockerfileの1行目はpython:3.7-slimのイメージをインポートしています。次の4行はappフォルダを作成し、**libgomp1 **libraryを更新し、**requirements.txt **ファイルからすべての要件をインストールします。最後に、最後の2行でアプリケーションのエントリーポイントを定義します。これは、コンテナーが起動したときに、先ほどステップ1で見た**app.py**ファイルを実行することを意味します。

# 👉 ステップ3 - action.ymlの作成

Dockerアクションには、メタデータファイルが必要です。メタデータファイル名は、action.ymlまたはaction.yamlのいずれかでなければなりません。メタデータファイルのデータは、アクションの入力、出力、およびメインのエントリーポイントを定義します。アクションファイルはYAML構文を使用します。

 <iframe src="https://medium.com/media/756f08f1f6b5f8be59d91530da2053ea" frameborder=0></iframe>

環境変数dataset、target、usecaseはそれぞれ6行目、9行目、14行目で宣言されています。app.pyの4-6行目を見れば、app.pyファイルでこれらの環境変数をどのように使用したかがわかります。

# 👉 ステップ 4 - アクションを GitHub で公開する

この時点で、プロジェクトのフォルダは以下のようになっているはずです。

![[https://github.com/pycaret/pycaret-git-actions](https://github.com/pycaret/pycaret-git-actions)](https://cdn-images-1.medium.com/max/2082/1*qBWs9Yk2Kgycu1wUtZe2Ow.png)

**'Releases'**をクリックします。

![GitHub Action - Click on Releases](https://cdn-images-1.medium.com/max/2804/1*rrr51HMFW0Sc6gD4A0Agtg.png)

新しいリリースを作成します。

![GitHub Action - Draft a new release](https://cdn-images-1.medium.com/max/3698/1*od3eRb8OaoeRhW4IT5ZduA.png)

詳細（タグ、リリースのタイトル、説明）を入力して、**'Publish release'**をクリックします。

![GitHub Action - Publish release](https://cdn-images-1.medium.com/max/2292/1*fW_n0JkZQEoUk-OBIP-4Sw.png)

公開されたら、リリースをクリックして、**'Marketplace'**をクリックします。

![GitHub Action - Marketplace](https://cdn-images-1.medium.com/max/2814/1*Dfa9llJIIUw501qaAUomRw.png)

**'Use latest version'**をクリックします。

![GitHub Action - use latest version](https://cdn-images-1.medium.com/max/2364/1*9F3GiDDYrIVcwvOmKIcMHA.png)をクリックします。

この情報を保存します。これがソフトウェアのインストールの詳細です。これは、任意のパブリックGitHubリポジトリにこのソフトウェアをインストールするために必要なものです。

![GitHub Action - installation](https://cdn-images-1.medium.com/max/2000/1*UihPzGDhm2smpqOS2YW4Yg.png)

# 👉 Step 5- GitHub リポジトリへのソフトウェアのインストール

先ほど作成したソフトウェアをインストールしてテストするために、新しいリポジトリ[**pycaret-automl-test](https://github.com/pycaret/pycaret-automl-test)**を作成し、分類と回帰のサンプルデータセットをいくつかアップロードしました。

前のステップで公開したソフトウェアをインストールするには、「**Actions**」をクリックしてください。

![[https://github.com/pycaret/pycaret-automl-test/tree/master](https://github.com/pycaret/pycaret-automl-test/tree/master)](https://cdn-images-1.medium.com/max/3776/1*MQKaHVJwqTZQWzwjNn5rcQ.png)

![Get started with GitHub Actions](https://cdn-images-1.medium.com/max/2000/1*h37nTkjLQhrbWRSwIL-VEQ.png)

**set up a workflow yourself**」をクリックして、このスクリプトをエディターにコピーし、**「Start commit」**をクリックします。

 <iframe src="https://medium.com/media/9adb786c134c59506fcabd820e351430" frameborder="0"> </iframe>。

これは、GitHubが実行するための命令ファイルです。9行目からが最初のアクションです。9行目から15行目までは、先に開発したソフトウェアをインストールして実行するためのアクションです。11行目は、ソフトウェアの名前を参照したところです（上記ステップ4の最後の部分を参照）。13行目から15行目は、データセットの名前（csvファイルがリポジトリ上にアップロードされている必要があります）、ターゲット変数の名前、ユースケースタイプなどの環境変数を定義するアクションです。16行目以降は、[this repository](https://github.com/actions/upload-artifact)から、model.pkl、csvファイルのexperiment logs、.logファイルのsystem logsの3つのファイルをアップロードするアクションです。

コミットを開始したら、 **'action'** をクリックします。

![GitHub Action - Workflows](https://cdn-images-1.medium.com/max/2870/1*rYW8L7Yvtj1BIsFL18jquw.png)をクリックします。

ここでは、ビルドのログを監視することができます。また、ワークフローが完了したら、この場所からファイルを収集することもできます。

![GitHub Action - Workflow build logs](https://cdn-images-1.medium.com/max/3062/1*SD4IMJjgg_PB-aAKxYDA0g.png)

![GitHub Action - Run Details](https://cdn-images-1.medium.com/max/3034/1*xmXuNcrm7pJ4F64R7mJXmQ.png)

ファイルをダウンロードして、端末で解凍することができます。

# **ファイル：モデル**)

これは、最終モデルの.pklファイルで、変換パイプライン全体と一緒になっています。このファイルを使用して、predict_model関数を使用して新しいデータセットの予測値を生成することができます。詳しくは[ここをクリック](https://www.pycaret.org/predict-model)をご覧ください。

# ファイル: experiment-logs

このファイルは、モデルに必要なすべての詳細を含む.csvファイルです。app.pyスクリプトで学習された全てのモデル、そのパフォーマンスメトリクス、ハイパーパラメータ、その他の重要なメタデータが含まれています。

![実験ログファイル](https://cdn-images-1.medium.com/max/3830/1*i4fvedl-mtKMtOtWl2pfUQ.png)

# ファイル: システムログ

これは、PyCaretが生成したシステムログファイルです。これは、プロセスの監査に使用できます。重要なメタ情報が含まれており、ソフトウェアのエラーのトラブルシューティングに非常に役立ちます。

![PyCaretが生成したシステムログファイル](https://cdn-images-1.medium.com/max/3838/1*QQ4Um9aRxLhyyLwW-oD4fg.png)

# **開示事項**)

GitHub Actions は、GitHub リポジトリ上でソフトウェア開発ライフサイクルのワークフローを作成することができます。各アカウントには、アカウントプランに応じて、Actionsで使用するためのコンピュート量とストレージ量が含まれており、[Actions documentation](https://docs.github.com/en/github/automating-your-workflow-with-github-actions/about-github-actions#about-github-actions)に記載されています。

アクションおよびアクションサービスのいかなる要素も、本契約、[Acceptable Use Policy](https://docs.github.com/en/github/site-policy/github-acceptable-use-policies)、またはGitHub Actions [service limitations](https://docs.github.com/en/github/automating-your-workflow-with-github-actions/about-github-actions#usage-limits)に違反して使用することはできません。さらに、アクションは以下の目的で使用してはいけません。

* クリプトマイニング。

* サーバーレスコンピューティング。

* サービス、デバイス、データ、アカウント、ネットワークを破壊したり、不正なアクセスを得たり、得ようとするために私たちのサーバーを使用すること（[GitHub Bug Bounty program](https://bounty.github.com/)で許可されたものを除く）。

* 行動」または「行動」の要素を商業目的で提供するスタンドアロンまたは統合されたアプリケーションやサービスの提供、または。

* その他、GitHub Actionsが使用されているリポジトリに関連するソフトウェアプロジェクトの制作、テスト、デプロイ、または公開に関連しない活動。

これらの制限事項への違反およびGitHub Actionsの悪用を防止するために、GitHubはお客様のGitHub Actionsの使用を監視することがあります。GitHub Actionsを悪用した場合には、ジョブの終了やGitHub Actionsの使用を制限することがあります。

# **このチュートリアルで使用するリポジトリ：**。
[**pycaret/pycaret-git-actions**]を使用しています。
*pycaret-git-actions. Contribute to pycaret/pycaret-git-actions development by creating an account on GitHub.*github.com](https://github.com/pycaret/pycaret-git-actions)
[**pycaret/pycaret-automl-test** (英語)
*pycaret-automl-testです。Contribute to pycaret/pycaret-automl-test development by creating an account on GitHub.*github.com](https://github.com/pycaret/pycaret-automl-test)

このPythonによる軽量なワークフロー自動化ライブラリを使って実現できることは無限にあります。もしお役に立てましたら、github repoに⭐️をお願いします。

PyCaretについてもっと知りたい方は、[LinkedIn](https://www.linkedin.com/company/pycaret/)や[Youtube](https://www.youtube.com/channel/UCxA1YTYJ9BEeo50lxyI_B3g)でフォローしてください。

PyCaret 2.0についてもっと知りたい方は、こちらの[お知らせ](https://towardsdatascience.com/announcing-pycaret-2-0-39c11014540e)をご覧ください。以前にPyCaretを使ったことがある方は、現在のリリースの[release notes](https://github.com/pycaret/pycaret/releases/tag/2.0)に興味があるかもしれません。

# あなたはこれにも興味があるかもしれません。

[PyCaret 2.0を使ってPower BIで独自のAutoMLを構築する](https://towardsdatascience.com/build-your-own-automl-in-power-bi-using-pycaret-8291b64181d)
[Deploy Machine Learning Pipeline on Azure using Docker](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-cloud-using-docker-container-bec64458dc01)
[Google Kubernetes Engineへの機械学習パイプラインの導入](https://towardsdatascience.com/deploy-machine-learning-model-on-google-kubernetes-engine-94daac85108b)
[Deploy Machine Learning Pipeline on AWS Fargate](https://towardsdatascience.com/deploy-machine-learning-pipeline-on-aws-fargate-eb6e1c50507)
[はじめての機械学習ウェブアプリの構築とデプロイ](https://towardsdatascience.com/build-and-deploy-your-first-machine-learning-web-app-e020db344a99)
[Deploy PyCaret and Streamlit app using AWS Fargate serverless](https://towardsdatascience.com/deploy-pycaret-and-streamlit-app-using-aws-fargate-serverless-infrastructure-8b7d7c0584c2)
[PyCaretとStreamlitを使った機械学習のWebアプリの構築とデプロイ](https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104)
[StreamlitとPyCaretを使って構築した機械学習アプリをGKEでデプロイ](https://towardsdatascience.com/deploy-machine-learning-app-built-using-streamlit-and-pycaret-on-google-kubernetes-engine-fd7e393d99cb)

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
](https://www.pycaret.org/regression) [Clustering](https://www.pycaret.org/clustering)
[異常検知
](https://www.pycaret.org/anomaly-detection)[自然言語処理](https://www.pycaret.org/nlp)
[アソシエーション・ルール・マイニング](https://www.pycaret.org/association-rules)
