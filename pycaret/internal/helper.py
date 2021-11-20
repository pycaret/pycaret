# report helper


# Specific to model type
def get_plot_list(model_name, model_type):
    plot_list = ''
    if model_type == 'regression':
        plot_list = get_plot_list_regression(model_name)
    elif model_type == 'classification':
        plot_list = get_plot_list_classification(model_name)
    elif model_type == 'clustering':
        plot_list = get_plot_list_clustering(model_name)
    elif model_type == 'anomaly_detection':
        plot_list == get_plot_list_anomaly_detection(model_name)
    elif model_type == 'nlp':
        plot_list = get_plot_list_nlp(model_name)
    return plot_list


def get_plot_list_regression(model_name):
    plot_list_dict = {
        'lr ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'manifold', 'feature', 'feature_all'],
        'lasso ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'ridge ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'en ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'lar ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'llar ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'omp ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'br ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'ard ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'par ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'ransac ': ['residuals', 'error', 'cooks', 'learning', 'manifold'],
        'tr ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'huber ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all'],
        'kr ': ['residuals', 'error', 'cooks', 'learning', 'manifold'],
        'svm ': ['residuals', 'error', 'cooks', 'learning', 'manifold'],
        'knn ': ['residuals', 'error', 'cooks', 'learning', 'manifold'],
        'dt ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'rf ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'et ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'ada ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'gbr ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'mlp ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all', 'tree'],
        'xgboost ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all','tree'],
        'lightgbm ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all','tree'],
        'catboost ': ['residuals', 'error', 'cooks', 'rfe', 'learning', 'vc', 'manifold', 'feature', 'feature_all','tree']
    }
    return plot_list_dict[model_name]


def get_plot_list_classification(model_name):
    plot_list_dict = {
        'lr  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                 'feature', 'feature_all'],
        'knn  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension'],
        'nb  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension'],
        'dt  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                 'feature', 'feature_all', 'tree'],
        'svm  ': ['pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                  'feature', 'feature_all'],
        'rbfsvm  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc',
                     'dimension'],
        'gpc  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension'],
        'mlp  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension'],
        'ridge  ': ['pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                    'feature', 'feature_all'],
        'rf': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
               'feature', 'feature_all', 'tree'],
        'qda  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc'],
        'ada  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                  'feature', 'feature_all'],
        'gbc  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                  'feature', 'feature_all'],
        'lda  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                  'feature', 'feature_all', 'tree'],
        'et ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc', 'dimension',
                'feature', 'feature_all', 'tree'],
        'xgboost ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc',
                     'dimension', 'feature', 'feature_all', 'tree'],
        'lightgbm  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc',
                       'dimension', 'feature', 'feature_all'],
        'catboost  ': ['auc', 'pr', 'confusion_matrix', 'error', 'class_report', 'boundary', 'learning', 'vc',
                       'dimension', 'feature', 'feature_all']
    }
    return plot_list_dict[model_name]


def get_plot_list_clustering(model_name):
    plot_list_dict = {
        'kmeans ': ['cluster', 'tsne', 'elbow', 'silhouette', 'distance', 'distribution'],
        'ap ': ['cluster', 'tsne', 'distance', 'distribution'],
        'meanshift ': ['cluster', 'tsne', 'distance', 'distribution'],
        'sc ': ['cluster', 'tsne', 'elbow', 'distribution'],
        'hclust ': ['cluster', 'tsne', 'elbow', 'distribution'],
        'dbscan ': ['cluster', 'tsne', 'distribution'],
        'optics ': ['cluster', 'tsne', 'distribution'],
        'birch ': ['cluster', 'tsne', 'elbow', 'silhouette', 'distribution'],
        'kmodes ': ['cluster', 'tsne', 'elbow', 'silhouette', 'distribution']
    }
    return plot_list_dict[model_name]


def get_plot_list_anomaly_detection(model_name):
    plot_list_dict = {
        'abod ': ['tsne'],
        'cluster ': ['tsne'],
        'cof ': ['tsne'],
        'histogram ': ['tsne'],
        'knn ': ['tsne'],
        'lof ': ['tsne'],
        'svm ': ['tsne'],
        'pca ': ['tsne'],
        'mcd ': ['tsne'],
        'sod ': ['tsne'],
        'sos ': ['tsne']
    }
    return plot_list_dict[model_name]


def get_plot_list_nlp(model_name):
    plot_list_dict = {
        'lda ': ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'tsne', 'topic_distribution',
                 'wordcloud'],
        'lsi ': ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'tsne', 'topic_distribution',
                 'wordcloud'],
        'hdp ': ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'tsne', 'topic_distribution',
                 'wordcloud'],
        'rp ': ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'tsne', 'topic_distribution',
                'wordcloud'],
        'nmf ': ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'tsne', 'topic_distribution',
                 'wordcloud']
    }
    return plot_list_dict[model_name]


def get_report_name(report_type, model_type):
    report_name = ''
    if model_type == 'regression':
        report_name = get_report_name_regression(report_type)
    elif model_type == 'classification':
        report_name = get_report_name_classification(report_type)
    elif model_type == 'clustering':
        report_name = get_report_name_clustering(report_type)
    elif model_type == 'anomaly_detection':
        report_name == get_report_name_anomaly_detection(report_type)
    elif model_type == 'nlp':
        report_name = get_report_name_nlp(report_type)
    return report_name


# Specific to each model type
def get_report_name_regression(report_type):
    report_dict = {
        'lr': ' Linear Regression',
        'lasso': ' Lasso Regression',
        'ridge': ' Ridge Regression',
        'en': ' Elastic Net',
        'lar': ' Least Angle Regression',
        'llar': ' Lasso Least Angle Regression',
        'omp': ' Orthogonal Matching Pursuit',
        'br': ' Bayesian Ridge',
        'ard': ' Automatic Relevance Determination',
        'par': ' Passive Aggressive Regressor',
        'ransac': ' Random Sample Consensus',
        'tr': ' TheilSen Regressor',
        'huber': ' Huber Regressor',
        'kr': ' Kernel Ridge',
        'svm': ' Support Vector Regression',
        'knn': ' K Neighbors Regressor',
        'dt': ' Decision Tree Regressor',
        'rf': ' Random Forest Regressor',
        'et': ' Extra Trees Regressor',
        'ada': ' AdaBoost Regressor',
        'gbr': ' Gradient Boosting Regressor',
        'mlp': ' MLP Regressor',
        'xgboost': ' Extreme Gradient Boosting',
        'lightgbm': ' Light Gradient Boosting Machine',
        'catboost': ' CatBoost Regressor'
    }
    return report_dict[report_type]


def get_report_name_classification(report_type):
    report_dict = {
        'lr': 'Logistic Regression',
        'knn': 'K Neighbors Classifier',
        'nb': 'Naive Bayes',
        'dt': 'Decision Tree Classifier',
        'svm': 'SVM - Linear Kernel',
        'rbfsvm': 'SVM - Radial Kernel',
        'gpc': 'Gaussian Process Classifier',
        'mlp': 'MLP Classifier',
        'ridge': 'Ridge Classifier',
        'rf': ' Random Forest Classifier',
        'qda': 'Quadratic Discriminant Analysis',
        'ada': 'Ada Boost Classifier',
        'gbc': 'Gradient Boosting Classifier',
        'lda': 'Linear Discriminant Analysis',
        'et': ' Extra Trees Classifier',
        'xgboost': ' Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Classifier'
    }
    return report_dict[report_type]


def get_report_name_clustering(report_type):
    report_dict = {
        'kmeans': ' K -Means Clustering',
        'ap': ' Affinity Propagation',
        'meanshift': ' Mean shift Clustering',
        'sc': ' Spectral Clustering',
        'hclust': ' Agglomerative Clustering',
        'dbscan': ' Density -Based Spatial Clustering',
        'optics': ' OPTICS Clustering',
        'birch': ' Birch Clustering',
        'kmodes': ' K -Modes Clustering'
    }
    return report_dict[report_type]


def get_report_name_anomaly_detection(report_type):
    report_dict = {
        'abod': ' Angle - base Outlier Detection',
        'cluster': ' Clustering -Based Local Outlier',
        'cof': ' Connectivity - Based Outlier Factor',
        'histogram': ' Histogram -based Outlier Detection',
        'knn': ' k - Nearest Neighbors Detector',
        'lof': ' Local Outlier Factor',
        'svm': ' One - class SVM detector',
        'pca': ' Principal Component Analysis',
        'mcd': ' Minimum Covariance Determinant',
        'sod': ' Subspace Outlier Detection',
        'sos': ' Stochastic Outlier Selection'
    }
    return report_dict[report_type]


def get_report_name_nlp(report_type):
    report_dict = {
        'lda ': ' Latent Dirichlet Allocation',
        'lsi ': ' Latent Semantic Indexing',
        'hdp ': ' Hierarchical Dirichlet Process',
        'rp ': ' Random Projections',
        'nmf ': ' Non - Negative Matrix Factorization'
    }
    return report_dict[report_type]


# TODO -- This will be as per model_type
def get_model_definition(model_name, model_type):
    model_definition = ''
    if model_type == 'regression':
        model_definition = get_model_definition_regression(model_name)
    elif model_type == 'classification':
        model_definition = get_model_definition_classification(model_name)
    elif model_type == 'clustering':
        model_definition = get_model_definition_clustering(model_name)
    elif model_type == 'anomaly_detection':
        model_definition == get_model_definition_anomaly_detection(model_name)
    elif model_type == 'nlp':
        model_definition = get_model_definition_nlp(model_type)
    return model_definition


def get_model_definition_regression(model_name):
    model_name_dict = {

    }
    return model_name_dict[model_name]


def get_model_definition_classification(model_name):
    model_name_dict = {

    }
    return model_name_dict[model_name]


def get_model_definition_clustering(model_name):
    model_name_dict = {
        'kmeans': 'K-means clustering is one of the simplest and popular unsupervised machine '
                  'learning algorithms.A cluster refers to a collection of data points aggregated '
                  'together because of certain similarities.Define a target number k, which refers '
                  'to the number of centroids you need in the dataset. A centroid is the imaginary '
                  'or real location representing the center of the cluster.Every data point is '
                  'allocated to each of the clusters through reducing the in-cluster sum of '
                  'squares.In other words, the K-means algorithm identifies k number of centroids, '
                  'and then allocates every data point to the nearest cluster, while keeping the '
                  'centroids as small as possible.The ‘means’ in the K-means refers to averaging of '
                  'the data; that is, finding the centroid.',
        'ap': 'Affinity propagation (AP) is a clustering algorithm based on the concept of "message '
              'passing" between data points.[1] Unlike clustering algorithms such as k-means or '
              'k-medoids, affinity propagation does not require the number of clusters to be '
              'determined or estimated before running the algorithm. Similar to k-medoids, '
              'affinity propagation finds "exemplars," members of the input set that are '
              'representative of clusters.',
        'meanshift': 'Meanshift is falling under the category of a clustering algorithm in contrast '
                     'of Unsupervised learning that assigns the data points to the clusters '
                     'iteratively by shifting points towards the mode (mode is the highest density '
                     'of data points in the region, in the context of the Meanshift). As such, '
                     'it is also known as the Mode-seeking algorithm. Mean-shift algorithm has '
                     'applications in the field of image processing and computer vision.Unlike the '
                     'popular K-Means cluster algorithm, mean-shift does not require specifying the '
                     'number of clusters in advance. The number of clusters is determined by the '
                     'algorithm with respect to the data.Mean-shift builds upon the concept of '
                     'kernel density estimation is sort KDE. Imagine that the above data was '
                     'sampled from a probability distribution. KDE is a method to estimate the '
                     'underlying distribution also called the probability density function for a '
                     'set of data.It works by placing a kernel on each point in the data set. A '
                     'kernel is a fancy mathematical word for a weighting function generally used '
                     'in convolution. There are many different types of kernels, but the most '
                     'popular one is the Gaussian kernel. Adding up all of the individual kernels '
                     'generates a probability surface example density function. Depending on the '
                     'kernel bandwidth parameter used, the resultant density function will vary.',
        'sc': 'In multivariate statistics, spectral clustering techniques make use of the spectrum '
              '(eigenvalues) of the similarity matrix of the data to perform dimensionality '
              'reduction before clustering in fewer dimensions. The similarity matrix is provided '
              'as an input and consists of a quantitative assessment of the relative similarity of '
              'each pair of points in the dataset.In application to image segmentation, '
              'spectral clustering is known as segmentation-based object categorization.',
        'hclust': 'The agglomerative clustering is the most common type of hierarchical clustering '
                  'used to group objects in clusters based on their similarity. It’s also known as '
                  'AGNES (Agglomerative Nesting). The algorithm starts by treating each object as a '
                  'singleton cluster. Next, pairs of clusters are successively merged until all '
                  'clusters have been merged into one big cluster containing all objects. The '
                  'result is a tree-based representation of the objects, '
                  'named dendrogram.Agglomerative clustering works in a “bottom-up” manner. That '
                  'is, each object is initially considered as a single-element cluster (leaf). At '
                  'each step of the algorithm, the two clusters that are the most similar are '
                  'combined into a new bigger cluster (nodes). This procedure is iterated until all '
                  'points are member of just one single big cluster (root) (see figure below).The '
                  'inverse of agglomerative clustering is divisive clustering, which is also known '
                  'as DIANA (Divise Analysis) and it works in a “top-down” manner. It begins with '
                  'the root, in which all objects are included in a single cluster. At each step of '
                  'iteration, the most heterogeneous cluster is divided into two. ',
        'dbscan': '',
        'optics': 'Ordering points to identify the clustering structure (OPTICS) is an algorithm '
                  'for finding density-based clusters in spatial data. It was presented by Mihael '
                  'Ankerst, Markus M. Breunig, Hans-Peter Kriegel and Jörg Sander.Its basic idea is '
                  'similar to DBSCAN, but it addresses one of DBSCANs major weaknesses: the problem '
                  'of detecting meaningful clusters in data of varying density.To do so, '
                  'the points of the database are(linearly) ordered such that spatially closest '
                  'points become neighbors in the ordering.Additionally, a special distance is '
                  'stored for each point that represents the density that must be accepted for a '
                  'cluster so that both points belong to the same cluster.This is represented as a '
                  'dendrogram.',
        'birch': 'BIRCH (balanced iterative reducing and clustering using hierarchies) is an '
                 'unsupervised data mining algorithm used to perform hierarchical clustering over '
                 'particularly large data-sets. With modifications it can also be used to '
                 'accelerate k-means clustering and Gaussian mixture modeling with the '
                 'expectation–maximization algorithm.An advantage of BIRCH is its ability to '
                 'incrementally and dynamically cluster incoming, multi-dimensional metric data '
                 'points in an attempt to produce the best quality clustering for a given set of '
                 'resources (memory and time constraints). In most cases, BIRCH only requires a '
                 'single scan of the database.Its inventors claim BIRCH to be the "first clustering '
                 'algorithm proposed in the database area to handle noise (data points that are not '
                 'part of the underlying pattern) effectively, beating DBSCAN by two months. The '
                 'BIRCH algorithm received the SIGMOD 10 year test of time award in 2006.',
        'kmodes': ''
    }
    return model_name_dict[model_name]


def get_model_definition_anomaly_detection(model_name):
    model_name_dict = {
        'abod': 'Angle-based Outlier Detection (ABOD) evaluates the degree of outlierness on the variance of the '
                'angles (VOA) between a point and all other pairs of points in the data set.The smaller the angle '
                'variance of the point has,the more likely it is an outlier.',
        'cluster': '',
        'cof': '',
        'histogram': '',
        'knn': '',
        'lof': 'The Local Outlier Factor (LOF) algorithm is an unsupervised anomaly detection method which computes '
                'the local density deviation of a given data point with respect to its neighbors. It considers as '
                'outliers the samples that have a substantially lower density than their neighbors. This example '
                'shows how to use LOF for novelty detection. Note that when LOF is used for novelty detection you '
                'MUST not use predict, decision_function and score_samples on the training set as this would lead to '
                'wrong results. You must only use these methods on new unseen data (which are not in the training '
                'set). See User Guide: for details on the difference between outlier detection and novelty detection '
                'and how to use LOF for outlier detection.The number of neighbors considered, (parameter n_neighbors) '
                'is typically set 1) greater than the minimum number of samples a cluster has to contain, '
                'so that other samples can be local outliers relative to this cluster, and 2) smaller than the '
                'maximum number of close by samples that can potentially be local outliers. In practice, '
                'such information is generally not available, and taking n_neighbors=20 appears to work well in '
                'general.',
        'svm': '',
        'pca': 'The principal components of a collection of points in a real coordinate space are a sequence of  '
                'unit vectors, where the i-th vector is the direction of a line that best fits the data while being '
                'orthogonal to the first i-1 vectors. Here, a best-fitting line is defined as one that minimizes the '
                'average squared distance from the points to the line. These directions constitute an orthonormal '
                'basis in which different individual dimensions of the data are linearly uncorrelated. Principal '
                'component analysis (PCA) is the process of computing the principal components and using them to '
                'perform a change of basis on the data, sometimes using only the first few principal components and '
                'ignoring the rest.PCA is used in exploratory data analysis and for making predictive models. It is '
                'commonly used for dimensionality reduction by projecting each data point onto only the first few '
                'principal components to obtain lower-dimensional data while preserving as much of the data variation '
                'as possible. The first principal component can equivalently be defined as a direction that maximizes '
                'the variance of the projected data. The i-th principal component can be taken as a direction '
                'orthogonal to the first i-1 principal components that maximizes the variance of the projected '
                'data.From either objective, it can be shown that the principal components are eigenvectors of the '
                'data covariance matrix. Thus, the principal components are often computed by eigen decomposition of '
                'the data covariance matrix or singular value decomposition of the data matrix. PCA is the simplest '
                'of the true eigenvector-based multivariate analyses and is closely related to factor analysis. '
                'Factor analysis typically incorporates more domain specific assumptions about the underlying '
                'structure and solves eigenvectors of a slightly different matrix. PCA is also related to canonical '
                'correlation analysis (CCA). CCA defines coordinate systems that optimally describe the '
                'cross-covariance between two datasets while PCA defines a new orthogonal coordinate system that '
                'optimally describes variance in a single dataset.Robust and L1-norm-based variants of standard PCA '
                'have also been proposed.',
        'mcd': 'The minimum covariance determinant (MCD) estimator is one of the first affine equivariant and highly '
                'robust estimators of multivariate location and scatter.1,2 Being resistant to outlying observations,'
                'makes the MCD very helpful in outlier detection.Although already introduced in 1984, its main use '
                'has only started since the introduction of the computationally efficient FAST-MCD algorithm of '
                'Rousseeuw and Van Driessen.Since then, the MCD has been applied in numerous fields such as medicine, '
                'finance, image analysis, and chemistry. Moreover, the MCD has also been used to develop many robust '
                'multivariate techniques, such as principal component analysis, factor analysis, and multiple '
                'regression.',
        'sod': 'Rare data in a large-scale database are called outliers that reveal significant information in the '
                'real world. The subspace-based outlier detection is regarded as a feasible approach in very high '
                'dimensional space. However, the outliers found in subspaces are only part of the true outliers in '
                'high dimensional space, indeed. The outliers hidden in normal-clustered points are sometimes '
                'neglected in the projected dimensional subspace. In this paper, we propose a robust subspace method '
                'for detecting such inner outliers in a given dataset, which uses two dimensional-projections: '
                'detecting outliers in subspaces with local density ratio in the first projected dimensions; finding '
                'outliers by comparing neighbors positions in the second projected dimensions. Each points weight is '
                'calculated by summing up all related values got in the two steps projected dimensions, and then the '
                'points scoring the largest weight values are taken as outliers. By taking a series of experiments '
                'with the number of dimensions from 10 to 10000, the results show that our proposed method achieves '
                'high precision in the case of extremely high dimensional space, and works well in low dimensional '
                'space.',
        'sos': 'Stochastic Outlier Selection is an unsupervised outlier-selection algorithm that takes as input '
                'either a feature matrix or a dissimilarity matrix and outputs for each data point an outlier '
                'probability. Intuitively, a data point is considered to be an outlier when the other data points '
                'have insufficient affinity with it.'
    }
    return model_name_dict[model_name]


def get_model_definition_nlp(model_name):
    model_name_dict = {
        'lda ': 'In natural language processing, the latent Dirichlet allocation (LDA) is a generative statistical '
                'model that allows sets of observations to be explained by unobserved groups that explain why some '
                'parts of the data are similar. For example, if observations are words collected into documents, '
                'it posits that each document is a mixture of a small number of topics and that each words presence '
                'is attributable to one of the documents topics. LDA is an example of a topic model and belongs to '
                'the machine learning field and in a wider sense to the artificial intelligence field.',
        'lsi ': 'Latent Semantic Indexing, also known as latent semantic analysis, is a mathematical practice that '
                'helps classify and retrieve information on particular key terms and concepts using singular value '
                'decomposition (SVD).Through SVD, search engines are able to scan through unstructured data and '
                'identify any relationships between these terms and their context to better index these records for '
                'users online.',
        'hdp ': 'In statistics and machine learning, the hierarchical Dirichlet process (HDP) is a nonparametric '
                'Bayesian approach to clustering grouped data.It uses a Dirichlet process for each group of data, '
                'with the Dirichlet processes for all groups sharing a base distribution which is itself drawn from a '
                'Dirichlet process. This method allows groups to share statistical strength via sharing of clusters '
                'across groups. The base distribution being drawn from a Dirichlet process is important, '
                'because draws from a Dirichlet process are atomic probability measures, and the atoms will appear in '
                'all group-level Dirichlet processes. Since each atom corresponds to a cluster, clusters are shared '
                'across all groups.',
        'rp ': 'Random projection is a technique used to reduce the dimensionality of a set of points which lie in '
               'Euclidean space. Random projection methods are known for their power, simplicity, and low error rates '
               'when compared to other methods. According to experimental results, random projection preserves '
               'distances well, but empirical results are sparse.They have been applied to many natural language '
               'tasks under the name random indexing.',
        'nmf ': 'Non-negative matrix factorization (NMF or NNMF), also non-negative matrix approximation is a group '
                'of algorithms in multivariate analysis and linear algebra where a matrix V is factorized into ('
                'usually) two matrices W and H, with the property that all three matrices have no negative elements. '
                'This non-negativity makes the resulting matrices easier to inspect. Also, in applications such as '
                'processing of audio spectrograms or muscular activity, non-negativity is inherent to the data being '
                'considered. Since the problem is not exactly solvable in general, it is commonly approximated '
                'numerically.'
    }
    return model_name_dict[model_name]


# Plot name is same for all models -- completed
def get_plot_name(plot):
    plot_dict = {
        'auc': ' Area Under the Curve',
        'threshold': ' Discrimination Threshold',
        'pr': ' Precision Recall Curve',
        'confusion_matrix': ' Confusion Matrix',
        'error': ' Prediction Error Plot',
        'class_report': ' Classification Report',
        'boundary': ' Decision Boundary',
        'rfe': ' Recursive Feature Selection',
        'learning': ' Learning Curve',
        'manifold': ' Manifold Learning',
        'calibration': ' Calibration Curve',
        'vc': ' Validation Curve',
        'dimension': ' Dimension Learning',
        'feature': ' Feature Importance',
        'feature_all': ' Feature Importance (All)',
        'parameter': ' Model Hyperparameter',
        'lift': ' Lift Curve',
        'gain': ' Gain Chart',
        'tree': ' Decision Tree',
        'ks': ' KS Statistic Plot',
        'residuals': ' Residuals Plot',
        'cooks': ' Cooks Distance Plot',
        'cluster': ' Cluster PCA Plot (2d)',
        'elbow': ' Elbow Plot',
        'silhouette': ' Silhouette Plot',
        'distance': ' Distance Plot',
        'distribution': 'Distribution Plot',
        'tsne': ' t - SNE (3d) Dimension Plot',
        'frequency': 'Word Token Frequency ',
        'bigram': 'Bigram Frequency Plot ',
        'trigram': 'Trigram Frequency Plot ',
        'sentiment': 'Sentiment Polarity Plot ',
        'pos': 'Part of Speech Frequency ',
        'topic_model': 'Topic Model (pyLDAvis) ',
        'topic_distribution': 'Topic Infer Distribution ',
        'wordcloud': 'Word Cloud ',
        'umap': 'UMAP Dimensionality Plot '
    }
    return plot_dict[plot]


# Plot definition is same for all models -- completed
def get_plot_definition(plot):
    plot_definition = {
        'auc': 'Area under the curve is calculated by different methods, of which the antiderivative method of '
               'finding the area is most popular. The area under the curve can be found by knowing the equation of '
               'the curve, the boundaries of the curve, and the axis enclosing the curve. ',
        'threshold': 'The discrimination threshold is the probability or score at which the positive class is chosen '
                     'over the negative class. Generally, this is set to 50% but the threshold can be adjusted to '
                     'increase or decrease the sensitivity to false positives or to other application factors.',
        'pr': 'The precision-recall curve shows the tradeoff between precision and recall for different threshold. A '
              'high area under the curve represents both high recall and high precision, where high precision '
              'relates to a low false positive rate, and high recall relates to a low false negative rate. High '
              'scores for both show that the classifier is returning accurate results (high precision), as well as '
              'returning a majority of all positive results (high recall).',
        'confusion_matrix': 'A confusion matrix is a table that is often used to describe the performance of a '
                            'classification model (or "classifier") on a set of test data for which the true values '
                            'are known.',
        'error': 'The class prediction error chart provides a way to quickly understand how good your classifier is '
                 'at predicting the right classes.',
        'class_report': 'A classification report is a performance evaluation metric in machine learning. It is used '
                        'to show the precision, recall, F1 Score, and support of your trained classification model.',
        'boundary': 'A decision boundary is the region of a problem space in which the output label of a classifier '
                    'is ambiguous. If the decision surface is a hyperplane, then the classification problem is '
                    'linear, and the classes are linearly separable. Decision boundaries are not always clear cut.',
        'rfe': 'Recursive feature elimination (RFE) is a feature selection method that fits a model and removes the '
               'weakest feature (or features) until the specified number of features is reached',
        'learning': 'A learning curve is a plot of model learning performance over experience or time. Learning '
                    'curves are a widely used diagnostic tool in machine learning for algorithms that learn from a '
                    'training dataset incrementally.',
        'manifold': 'A manifold curve is a generalization and abstraction of the notion of a curved surface.',
        'calibration': 'Calibration curves are used to evaluate how calibrated a classifier is i.e., how the '
                       'probabilities of predicting each class label differ. The x-axis represents the average '
                       'predicted probability in each bin. The y-axis is the ratio of positives (the proportion of '
                       'positive predictions).',
        'vc': 'A Validation Curve is an important diagnostic tool that shows the sensitivity between to changes in a '
              'model’s accuracy with change in some parameter of the model. A validation curve is typically drawn '
              'between some parameter of the model and the model’s score. Two curves are present in a validation '
              'curve – one for the training set score and one for the cross-validation score. By default, '
              'the function for validation curve, present in the scikit-learn library performs 3-fold '
              'cross-validation. A validation curve is used to evaluate an existing model based on hyper-parameters '
              'and is not used to tune a model. This is because, if we tune the model according to the validation '
              'score, the model may be biased towards the specific data against which the model is tuned; thereby, '
              'not being a good estimate of the generalization of the model.',
        'dimension': 'dimension',
        'feature': 'Feature importance refers to techniques that assign a score to input features based on how '
                   'useful they are at predicting a target variable.',
        'feature_all': 'this plot shows the relative importance of all the features for a given dataset.',
        'parameter': 'A model hyperparameter is a configuration that is external to the model and whose value cannot '
                     'be estimated from data.',
        'lift': 'The Lift curve shows the curves for analysing the proportion of true positive data instances in '
                'relation to the classifiers threshold or the number of instances that we classify as positive.',
        'gain': 'The cumulative gains curve is an evaluation curve that assesses the performance of the model and '
                'compares the results with the random pick. It shows the percentage of targets reached when '
                'considering a certain percentage of the population with the highest probability to be target '
                'according to the model.',
        'tree': 'A decision tree is a flowchart-like structure in which each internal node represents a test on an '
                'attribute',
        'ks': 'The KS test report the maximum difference between the two cumulative distributions, and calculates a '
              'P value from that and the sample sizes.',
        'residuals_interactive': 'A residual plot is a type of plot that displays the values of a predictor variable '
                                 'in a regression model along the x-axis and the values of the residuals along the '
                                 'y-axis.',
        'residuals': 'A residual plot is a type of plot that displays the values of a predictor variable in a '
                     'regression model along the x-axis and the values of the residuals along the y-axis.',
        'cooks': 'Cooks Distance is an estimate of the influence of a data point. It takes into account both the '
                 'leverage and residual of each observation. Cooks Distance is a summary of how much a regression '
                 'model changes when the ith observation is removed.',
        'cluster': 'Principal Component Analysis (PCA) is a popular technique for deriving a set of low dimensional '
                   'features from a larget set of variables. However, another popular application of PCA is '
                   'visualizing higher dimensional data. In this tutorial, we will go over the basics of PCA and '
                   'apply it to cluster a data set consisting of houses with different features.',
        'elbow': 'The elbow method plots the value of the cost function produced by different values of k. As you '
                 'know, if k increases, average distortion will decrease, each cluster will have fewer constituent '
                 'instances, and the instances will be closer to their respective centroids.',
        'silhouette': 'The silhouette plot displays a measure of how close each point in one cluster is to points in '
                      'the neighboring clusters and thus provides a way to assess parameters like number of clusters '
                      'visually.',
        'distance': 'A distance-time graph shows how far an object has travelled in a given time.It is a simple line '
                    'graph that denotes distance versus time findings on the graph.',
        'frequency': 'It is a distribution because it tells us how the total number of word tokens in the text are '
                     'distributed across the vocabulary items.',
        'distribution': 'This plot details the distribution of words in a text across the vocabulary items.',
        'bigram': 'A bigram or digram is a sequence of two adjacent elements from a string of tokens, which are '
                  'typically letters, syllables, or words. A bigram is an n-gram for n=2. The frequency distribution '
                  'of every bigram in a string is commonly used for simple statistical analysis of text in many '
                  'applications, including in computational linguistics, cryptography, speech recognition, and so on.',
        'trigram': 'Trigrams are a special case of the n-gram, where n is 3. They are often used in natural language '
                   'processing for performing statistical analysis of texts and in cryptography for control and use '
                   'of ciphers and codes.',
        'sentiment': 'The key aspect of sentiment analysis is to analyze a body of text for understanding the '
                     'opinion expressed by it. Typically, we quantify this sentiment with a positive or negative '
                     'value, called polarity. The overall sentiment is often inferred as positive, neutral or '
                     'negative from the sign of the polarity score',
        'pos': 'The rate of occurrence of anything; the relationship between incidence and time period.',
        'tsne': 'T-SNE is a common visualization for understanding high-dimensional data, and right now the '
                'variable tsne is an array where each row represents a set of (x, y, z) coordinates from the obtained '
                'embedding.',
        'topic_model': 'In machine learning and natural language processing, a topic model is a type of statistical '
                       'model for discovering the abstract "topics" that occur in a collection of documents. Topic '
                       'modeling is a frequently used text-mining tool for discovery of hidden semantic structures in '
                       'a text body. ',
        'topic_distribution': 'Each word in the document is attributed to a particular topic with probability given '
                              'by this distribution. Topics themselves are defined as probability distributions over '
                              'the vocabulary.',
        'wordcloud': 'Word Clouds (also known as wordle, word collage or tag cloud) are visual representations of '
                     'words that give greater prominence to words that appear more frequently',
        'umap': 'UMAP is a nonlinear dimensionality reduction method, it is very effective for visualizing clusters '
                'or groups of data points and their relative proximities.'
    }
    return plot_definition[plot]


# Image name is same for all models -- completed
def get_image_name(plot):
    image_name_dict = {
        'residuals ': 'Residuals.png',
        'error ': 'Prediction Error.png',
        'cooks ': 'Cooks Distance.png',
        'rfe ': 'Feature Selection.png',
        'learning ': 'Learning Curve.png',
        'vc ': 'Validation Curve.png',
        'manifold ': 'Manifold Learning.png',
        'feature ': 'Feature Importance.png',
        'feature_all ': 'Feature Importance (All).png',
        'parameter ': 'Hyperparameters.png',
        'tree ': 'Decision Tree.png',
        'plotIdentifier': 'Image Name',
        'auc ': 'AUC.png',
        'threshold ': 'Threshold.png',
        'pr ': 'Precision Recall.png',
        'confusion_matrix ': 'Confusion Matrix.png',
        'class_report ': 'Class Report.png',
        'boundary ': ' Decision Boundary.png',
        'calibration ': 'Calibration Curve.png',
        'dimension ': 'Dimensions.png',
        'lift ': 'Lift Chart.png',
        'gain ': 'Gain Chart.png',
        'ks ': 'KS Statistic Plot.png',
        'cluster': 'cluster',
        'elbow': 'Elbow.png',
        'silhouette': 'Silhouette.png',
        'distance': 'Distance.png',
        'tsne ': 'tsne',
        'umap ': 'umap',
        'frequency': 'Word Frequency.html',
        'distribution': 'Distribution.html',
        'bigram': 'Bigram.html',
        'trigram': 'Trigram.html',
        'sentiment': 'Sentiments.html',
        'pos': 'pos.html',
        'tsne': 'TSNE.html',
        'topic_model': '',
        'topic_distribution': 'Topic Distribution.html',
        'wordcloud': 'Wordcloud.png',
        'umap': 'umap'
    }
    return image_name_dict[plot]
