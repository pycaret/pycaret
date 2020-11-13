# Module: Natural Language Processing
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT
# Release: PyCaret 2.2.0
# Last modified : 25/10/2020


def setup(
    data,
    target=None,
    custom_stopwords=None,
    html=True,
    session_id=None,
    log_experiment=False,
    experiment_name=None,
    log_plots=False,
    log_data=False,
    verbose=True,
):

    """
    This function initializes the training environment and creates the transformation 
    pipeline. Setup function must be called before executing any other function. It takes 
    one mandatory parameter only: ``data``. All the other parameters are optional.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')

    
    data: pandas.Dataframe or list
        pandas.Dataframe with shape (n_samples, n_features) or a list.


    target: str
        When ``data`` is pandas.Dataframe, name of column containing text. 
    

    custom_stopwords: list, default = None
        List of stopwords.


    html: bool, default = True
        When set to False, prevents runtime display of monitor. This must be set to False
        when the environment does not support IPython. For example, command line terminal,
        Databricks Notebook, Spyder and other similar IDEs. 


    session_id: int, default = None
        Controls the randomness of experiment. It is equivalent to 'random_state' in
        scikit-learn. When None, a pseudo random number is generated. This can be used 
        for later reproducibility of the entire experiment.


    log_experiment: bool, default = False
        When set to True, all metrics and parameters are logged on the ``MLFlow`` server.


    experiment_name: str, default = None
        Name of the experiment for logging. Ignored when ``log_experiment`` is not True.


    log_plots: bool or list, default = False
        When set to True, certain plots are logged automatically in the ``MLFlow`` server.


    log_data: bool, default = False
        When set to True, dataset is logged on the ``MLflow`` server as a csv file.
        Ignored when ``log_experiment`` is not True.


    verbose: bool, default = True
        When set to False, Information grid is not printed.


    Returns:
        Global variables that can be changed using the ``set_config`` function.


    Warnings
    --------
    - pycaret.nlp requires following language models: 
      
        ``python -m spacy download en_core_web_sm``
        ``python -m textblob.download_corpora``
        
    """

    # exception checking
    import sys

    from pycaret.utils import __version__

    ver = __version__

    import logging

    # create logger
    global logger

    logger = logging.getLogger("logs")
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.FileHandler("logs.log")
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    logger.info("PyCaret NLP Module")
    logger.info("version " + str(ver))
    logger.info("Initializing setup()")

    # generate USI for mlflow tracking
    import secrets

    global USI
    USI = secrets.token_hex(nbytes=2)
    logger.info("USI: " + str(USI))

    try:
        data_shape = data.shape
    except:
        data_shape = len(data)

    logger.info(
        """setup(data={}, target={}, custom_stopwords={}, html={}, session_id={}, log_experiment={},
                    experiment_name={}, log_plots={}, log_data={}, verbose={})""".format(
            str(data_shape),
            str(target),
            str(custom_stopwords),
            str(html),
            str(session_id),
            str(log_experiment),
            str(experiment_name),
            str(log_plots),
            str(log_data),
            str(verbose),
        )
    )

    # logging environment and libraries
    logger.info("Checking environment")

    from platform import python_version, platform, python_build, machine

    try:
        logger.info("python_version: " + str(python_version()))
    except:
        logger.warning("cannot find platform.python_version")

    try:
        logger.info("python_build: " + str(python_build()))
    except:
        logger.warning("cannot find platform.python_build")

    try:
        logger.info("machine: " + str(machine()))
    except:
        logger.warning("cannot find platform.machine")

    try:
        logger.info("platform: " + str(platform()))
    except:
        logger.warning("cannot find platform.platform")

    try:
        import psutil

        logger.info("Memory: " + str(psutil.virtual_memory()))
        logger.info("Physical Core: " + str(psutil.cpu_count(logical=False)))
        logger.info("Logical Core: " + str(psutil.cpu_count(logical=True)))
    except:
        logger.warning(
            "cannot find psutil installation. memory not traceable. Install psutil using pip to enable memory logging. "
        )

    logger.info("Checking libraries")

    try:
        from pandas import __version__

        logger.info("pd==" + str(__version__))
    except:
        logger.warning("pandas not found")

    try:
        from numpy import __version__

        logger.info("numpy==" + str(__version__))
    except:
        logger.warning("numpy not found")

    try:
        import warnings

        warnings.filterwarnings("ignore")
        from gensim import __version__

        logger.info("gensim==" + str(__version__))
    except:
        logger.warning("gensim not found")

    try:
        from spacy import __version__

        logger.info("spacy==" + str(__version__))
    except:
        logger.warning("spacy not found")

    try:
        from nltk import __version__

        logger.info("nltk==" + str(__version__))
    except:
        logger.warning("nltk not found")

    try:
        from textblob import __version__

        logger.info("textblob==" + str(__version__))
    except:
        logger.warning("textblob not found")

    try:
        from pyLDAvis import __version__

        logger.info("pyLDAvis==" + str(__version__))
    except:
        logger.warning("pyLDAvis not found")

    try:
        from wordcloud import __version__

        logger.info("wordcloud==" + str(__version__))
    except:
        logger.warning("wordcloud not found")

    try:
        from mlflow.version import VERSION
        import warnings

        warnings.filterwarnings("ignore")
        logger.info("mlflow==" + str(VERSION))
    except:
        logger.warning("mlflow not found")

    logger.info("Checking Exceptions")

    # run_time
    import datetime, time

    runtime_start = time.time()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    """
    error handling starts here
    """

    # checking data type
    if hasattr(data, "shape") is False:
        if type(data) is not list:
            sys.exit(
                "(Type Error): data passed must be of type pandas.DataFrame or list"
            )

    # if dataframe is passed then target is mandatory
    if hasattr(data, "shape"):
        if target is None:
            sys.exit(
                "(Type Error): When pandas.Dataframe is passed as data param. Target column containing text must be specified in target param."
            )

    # checking target parameter
    if target is not None:
        if target not in data.columns:
            sys.exit(
                "(Value Error): Target parameter doesnt exist in the data provided."
            )

    # custom stopwords checking
    if custom_stopwords is not None:
        if type(custom_stopwords) is not list:
            sys.exit("(Type Error): custom_stopwords must be of list type.")

    # checking session_id
    if session_id is not None:
        if type(session_id) is not int:
            sys.exit("(Type Error): session_id parameter must be an integer.")

    # check if spacy is loaded
    try:
        import spacy

        sp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except:
        sys.exit(
            "(Type Error): spacy english model is not yet downloaded. See the documentation of setup to see installation guide."
        )

    # html
    if type(html) is not bool:
        sys.exit("(Type Error): html parameter only accepts True or False.")

    # log_experiment
    if type(log_experiment) is not bool:
        sys.exit("(Type Error): log_experiment parameter only accepts True or False.")

    # log_plots
    if type(log_plots) is not bool:
        sys.exit("(Type Error): log_plots parameter only accepts True or False.")

    # log_data
    if type(log_data) is not bool:
        sys.exit("(Type Error): log_data parameter only accepts True or False.")

    # verbose
    if type(verbose) is not bool:
        sys.exit("(Type Error): verbose parameter only accepts True or False.")

    """
    error handling ends here
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    # global html_param
    global html_param

    # create html_param
    html_param = html

    """
    generate monitor starts 
    """

    logger.info("Preparing display monitor")

    # progress bar
    max_steps = 11
    total_steps = 9

    progress = ipw.IntProgress(
        value=0, min=0, max=max_steps, step=1, description="Processing: "
    )
    if verbose:
        if html_param:
            display(progress)

    try:
        max_sub = len(data[target].values.tolist())
    except:
        max_sub = len(data)

    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame(
        [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            [
                "Step",
                ". . . . . . . . . . . . . . . . . .",
                "Step 0 of " + str(total_steps),
            ],
        ],
        columns=["", " ", "   "],
    ).set_index("")

    if verbose:
        if html_param:
            display(monitor, display_id="monitor")

    """
    generate monitor end
    """

    logger.info("Importing libraries")

    # general dependencies
    import numpy as np
    import random
    import spacy
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    import spacy
    import re
    import secrets

    # setting sklearn config to print all parameters including default
    import sklearn

    sklearn.set_config(print_changed_only=False)

    logger.info("Declaring global variables")

    # defining global variables
    global text, id2word, corpus, data_, seed, target_, experiment__, exp_name_log, logging_param, log_plots_param

    # create an empty list for pickling later.
    try:
        experiment__.append("dummy")
        experiment__.pop()

    except:
        experiment__ = []

    # converting to dataframe if list provided
    if type(data) is list:
        logger.info("Converting list into dataframe")
        data = pd.DataFrame(data, columns=["en"])
        target = "en"

    # converting target column into list
    try:
        text = data[target].values.tolist()
        target_ = str(target)
        logger.info("Input provided : dataframe")
    except:
        text = data
        target_ = "en"
        logger.info("Input provided : list")

    # generate seed to be used globally
    if session_id is None:
        seed = random.randint(150, 9000)
    else:
        seed = session_id

    logger.info("session_id set to : " + str(seed))

    logger.info("Copying training dataset")
    # copying dataframe
    if type(data) is list:
        data_ = pd.DataFrame(data)
        data_.columns = ["en"]
    else:
        data_ = data.copy()

    # create logging parameter
    logging_param = log_experiment

    # create exp_name_log param incase logging is False
    exp_name_log = "no_logging"

    # create an empty log_plots_param
    if log_plots:
        log_plots_param = True
    else:
        log_plots_param = False

    progress.value += 1

    """
    DEFINE STOPWORDS
    """
    try:
        logger.info("Importing stopwords from nltk")
        import nltk

        nltk.download("stopwords")
        from nltk.corpus import stopwords

        stop_words = stopwords.words("english")

    except:
        logger.info(
            "Importing stopwords from nltk failed .. loading pre-defined stopwords"
        )

        stop_words = [
            "ourselves",
            "hers",
            "between",
            "yourself",
            "but",
            "again",
            "there",
            "about",
            "once",
            "during",
            "out",
            "very",
            "having",
            "with",
            "they",
            "own",
            "an",
            "be",
            "some",
            "for",
            "do",
            "its",
            "yours",
            "such",
            "into",
            "of",
            "most",
            "itself",
            "other",
            "off",
            "is",
            "s",
            "am",
            "or",
            "who",
            "as",
            "from",
            "him",
            "each",
            "the",
            "themselves",
            "until",
            "below",
            "are",
            "we",
            "these",
            "your",
            "his",
            "through",
            "don",
            "nor",
            "me",
            "were",
            "her",
            "more",
            "himself",
            "this",
            "down",
            "should",
            "our",
            "their",
            "while",
            "above",
            "both",
            "up",
            "to",
            "ours",
            "had",
            "she",
            "all",
            "no",
            "when",
            "at",
            "any",
            "before",
            "them",
            "same",
            "and",
            "been",
            "have",
            "in",
            "will",
            "on",
            "does",
            "yourselves",
            "then",
            "that",
            "because",
            "what",
            "over",
            "why",
            "so",
            "can",
            "did",
            "not",
            "now",
            "under",
            "he",
            "you",
            "herself",
            "has",
            "just",
            "where",
            "too",
            "only",
            "myself",
            "which",
            "those",
            "i",
            "after",
            "few",
            "whom",
            "t",
            "being",
            "if",
            "theirs",
            "my",
            "against",
            "a",
            "by",
            "doing",
            "it",
            "how",
            "further",
            "was",
            "here",
            "than",
        ]

    if custom_stopwords is not None:
        stop_words = stop_words + custom_stopwords

    if custom_stopwords is None:
        logger.info("No custom stopwords defined")

    progress.value += 1

    """
    TEXT PRE-PROCESSING STARTS HERE
    """

    """
    STEP 1 - REMOVE NUMERIC CHARACTERS FROM THE LIST
    """
    logger.info("Removing numeric characters from the text")

    monitor.iloc[1, 1:] = "Removing Numeric Characters"
    monitor.iloc[2, 1:] = "Step 1 of " + str(total_steps)

    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    text_step1 = []

    for i in range(0, len(text)):
        review = re.sub("\d+", "", str(text[i]))
        text_step1.append(review)

    text = text_step1  # re-assigning
    del text_step1

    progress.value += 1

    """
    STEP 2 - REGULAR EXPRESSIONS
    """

    logger.info("Removing special characters from the text")

    monitor.iloc[1, 1:] = "Removing Special Characters"
    monitor.iloc[2, 1:] = "Step 2 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    text_step2 = []

    for i in range(0, len(text)):
        review = re.sub(r"\W", " ", str(text[i]))
        review = review.lower()
        review = re.sub(r"\s+[a-z]\s+", " ", review)
        review = re.sub(r"^[a-z]\s+", " ", review)
        review = re.sub(r"\d+", " ", review)
        review = re.sub(r"\s+", " ", review)
        text_step2.append(review)

    text = text_step2  # re-assigning
    del text_step2

    progress.value += 1

    """
    STEP 3 - WORD TOKENIZATION
    """

    logger.info("Tokenizing Words")

    monitor.iloc[1, 1:] = "Tokenizing Words"
    monitor.iloc[2, 1:] = "Step 3 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    text_step3 = []

    for i in text:
        review = gensim.utils.simple_preprocess(str(i), deacc=True)
        text_step3.append(review)

    text = text_step3
    del text_step3

    progress.value += 1

    """
    STEP 4 - REMOVE STOPWORDS
    """

    logger.info("Removing stopwords")

    monitor.iloc[1, 1:] = "Removing Stopwords"
    monitor.iloc[2, 1:] = "Step 4 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    text_step4 = []

    for i in text:
        ii = []
        for word in i:
            if word not in stop_words:
                ii.append(word)
        text_step4.append(ii)

    text = text_step4
    del text_step4

    progress.value += 1

    """
    STEP 5 - BIGRAM EXTRACTION
    """

    logger.info("Extracting Bigrams")

    monitor.iloc[1, 1:] = "Extracting Bigrams"
    monitor.iloc[2, 1:] = "Step 5 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    bigram = gensim.models.Phrases(text, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)

    text_step5 = []

    for i in text:
        text_step5.append(bigram_mod[i])

    text = text_step5
    del text_step5

    progress.value += 1

    """
    STEP 6 - TRIGRAM EXTRACTION
    """

    logger.info("Extracting Trigrams")

    monitor.iloc[1, 1:] = "Extracting Trigrams"
    monitor.iloc[2, 1:] = "Step 6 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    trigram = gensim.models.Phrases(bigram[text], threshold=100)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    text_step6 = []

    for i in text:
        text_step6.append(trigram_mod[bigram_mod[i]])

    text = text_step6
    del text_step6

    progress.value += 1

    """
    STEP 7 - LEMMATIZATION USING SPACY
    """

    logger.info("Lemmatizing tokens")

    monitor.iloc[1, 1:] = "Lemmatizing"
    monitor.iloc[2, 1:] = "Step 7 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nlp.max_length = (
        3000000  # increasing text length to 3000000 from default of 1000000
    )
    allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]

    text_step7 = []

    for i in text:
        doc = nlp(" ".join(i))
        text_step7.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        )

    text = text_step7
    del text_step7

    progress.value += 1

    """
    STEP 8  - CUSTOM STOPWORD REMOVER
    """

    logger.info("Removing stopwords after lemmatizing")

    monitor.iloc[1, 1:] = "Removing Custom Stopwords"
    monitor.iloc[2, 1:] = "Step 8 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    text_step8 = []

    for i in text:
        ii = []
        for word in i:
            if word not in stop_words:
                ii.append(word)
        text_step8.append(ii)

    text = text_step8
    del text_step8

    progress.value += 1

    """
    STEP 8 - CREATING CORPUS AND DICTIONARY
    """

    logger.info("Creating corpus and dictionary")

    monitor.iloc[1, 1:] = "Compiling Corpus"
    monitor.iloc[2, 1:] = "Step 9 of " + str(total_steps)
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    # creating dictionary
    id2word = corpora.Dictionary(text)

    # creating corpus

    corpus = []
    for i in text:
        d = id2word.doc2bow(i)
        corpus.append(d)

    progress.value += 1

    """
    PROGRESS NOT YET TRACKED - TO BE CODED LATER
    """

    logger.info("Compiling processed text")

    text_join = []

    for i in text:
        word = " ".join(i)
        text_join.append(word)

    data_[target_] = text_join

    """
    Final display Starts
    """

    if custom_stopwords is None:
        csw = False
    else:
        csw = True

    logger.info("Compiling information grid")

    functions = pd.DataFrame(
        [
            ["session_id", seed],
            ["Documents", len(corpus)],
            ["Vocab Size", len(id2word.keys())],
            ["Custom Stopwords", csw],
        ],
        columns=["Description", "Value"],
    )

    functions_ = functions.style.hide_index()

    """
    Final display Ends
    """

    # log into experiment
    experiment__.append(("Info", functions))
    experiment__.append(("Dataset", data_))
    experiment__.append(("Corpus", corpus))
    experiment__.append(("Dictionary", id2word))
    experiment__.append(("Text", text))

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    if logging_param:

        logger.info("Creating MLFlow logs")

        monitor.iloc[1, 1:] = "Creating Logs"
        monitor.iloc[2, 1:] = "Final"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        import mlflow
        from pathlib import Path
        import os

        if experiment_name is None:
            exp_name_ = "nlp-default-name"
        else:
            exp_name_ = experiment_name

        URI = secrets.token_hex(nbytes=4)
        exp_name_log = exp_name_

        try:
            mlflow.create_experiment(exp_name_log)
        except:
            pass

        # mlflow logging
        mlflow.set_experiment(exp_name_log)

        run_name_ = "Session Initialized " + str(USI)
        with mlflow.start_run(run_name=run_name_) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            k = functions.copy()
            k.set_index("Description", drop=True, inplace=True)
            kdict = k.to_dict()
            params = kdict.get("Value")
            mlflow.log_params(params)

            # set tag of compare_models
            mlflow.set_tag("Source", "setup")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log gensim id2word
            id2word.save("id2word")
            mlflow.log_artifact("id2word")
            import os

            os.remove("id2word")

            # Log data
            if log_data:
                data_.to_csv("data.csv")
                mlflow.log_artifact("data.csv")
                os.remove("data.csv")

            # Log plots
            if log_plots:

                logger.info(
                    "SubProcess plot_model() called =================================="
                )

                plot_model(plot="frequency", save=True, system=False)
                mlflow.log_artifact("Word Frequency.html")
                os.remove("Word Frequency.html")

                plot_model(plot="bigram", save=True, system=False)
                mlflow.log_artifact("Bigram.html")
                os.remove("Bigram.html")

                plot_model(plot="trigram", save=True, system=False)
                mlflow.log_artifact("Trigram.html")
                os.remove("Trigram.html")

                plot_model(plot="pos", save=True, system=False)
                mlflow.log_artifact("POS.html")
                os.remove("POS.html")

                logger.info(
                    "SubProcess plot_model() end =================================="
                )

    if verbose:
        clear_output()
        if html_param:
            display(functions_)
        else:
            print(functions_.data)

    logger.info("setup() succesfully completed......................................")

    return (
        text,
        data_,
        corpus,
        id2word,
        seed,
        target_,
        experiment__,
        exp_name_log,
        logging_param,
        log_plots_param,
        USI,
        html_param,
    )


def create_model(
    model=None, multi_core=False, num_topics=None, verbose=True, system=True, **kwargs
):

    """
    
    This function trains a given topic model. All the available models
    can be accessed using the ``models`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')
    >>> lda = create_model('lda')


    model: str, default = None
        Models available in the model library (ID - Name):

        * 'lda' - Latent Dirichlet Allocation         
        * 'lsi' - Latent Semantic Indexing           
        * 'hdp' - Hierarchical Dirichlet Process
        * 'rp' - Random Projections
        * 'nmf' - Non-Negative Matrix Factorization
   

    multi_core: bool, default = False
        True would utilize all CPU cores to parallelize and speed up model training.
        Ignored when ``model`` is not 'lda'.


    num_topics: int, default = 4
        Number of topics to be created. If None, default is set to 4.


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    system: bool, default = True
        Must remain True all times. Only to be changed by internal functions.


    **kwargs: 
        Additional keyword arguments to pass to the estimator.


    Returns:
        Trained Model
     
    """

    # exception checking
    import sys

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing create_model()")
    logger.info(
        """create_model(model={}, multi_core={}, num_topics={}, verbose={}, system={})""".format(
            str(model), str(multi_core), str(num_topics), str(verbose), str(system)
        )
    )

    logger.info("Checking exceptions")

    # run_time
    import datetime, time

    runtime_start = time.time()

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    """
    error handling starts here
    """

    # checking for model parameter
    if model is None:
        sys.exit(
            "(Value Error): Model parameter Missing. Please see docstring for list of available models."
        )

    # checking for allowed models
    allowed_models = ["lda", "lsi", "hdp", "rp", "nmf"]

    if model not in allowed_models:
        sys.exit(
            "(Value Error): Model Not Available. Please see docstring for list of available models."
        )

    # checking multicore type:
    if type(multi_core) is not bool:
        sys.exit(
            "(Type Error): multi_core parameter can only take argument as True or False."
        )

    # checking round parameter
    if num_topics is not None:
        if num_topics <= 1:
            sys.exit("(Type Error): num_topics parameter only accepts integer value.")

    # checking verbose parameter
    if type(verbose) is not bool:
        sys.exit(
            "(Type Error): Verbose parameter can only take argument as True or False."
        )

    """
    error handling ends here
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import numpy as np
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    """
    monitor starts
    """

    logger.info("Preparing display monitor")

    # progress bar and monitor control
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(
        value=0, min=0, max=4, step=1, description="Processing: "
    )
    monitor = pd.DataFrame(
        [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Initializing"],
        ],
        columns=["", " ", "  "],
    ).set_index("")
    if verbose:
        if html_param:
            display(progress)
            display(monitor, display_id="monitor")

    progress.value += 1

    """
    monitor starts
    """

    logger.info("Defining topic model")

    model_name_short = model

    # define topic_model_name
    if model == "lda":
        topic_model_name = "Latent Dirichlet Allocation"
    elif model == "lsi":
        topic_model_name = "Latent Semantic Indexing"
    elif model == "hdp":
        topic_model_name = "Hierarchical Dirichlet Process"
    elif model == "nmf":
        topic_model_name = "Non-Negative Matrix Factorization"
    elif model == "rp":
        topic_model_name = "Random Projections"

    logger.info("Model: " + str(topic_model_name))

    # defining default number of topics
    logger.info("Defining num_topics parameter")
    if num_topics is None:
        n_topics = 4
    else:
        n_topics = num_topics

    logger.info("num_topics set to: " + str(n_topics))

    # monitor update
    monitor.iloc[1, 1:] = "Fitting Topic Model"
    progress.value += 1
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    model_fit_start = time.time()

    if model == "lda":

        if multi_core:
            logger.info("LDA multi_core enabled")

            from gensim.models.ldamulticore import LdaMulticore

            logger.info("LdaMulticore imported successfully")

            model = LdaMulticore(
                corpus=corpus,
                num_topics=n_topics,
                id2word=id2word,
                workers=4,
                random_state=seed,
                chunksize=100,
                passes=10,
                alpha="symmetric",
                per_word_topics=True,
                **kwargs
            )

            logger.info("LdaMulticore trained successfully")

            progress.value += 1

        else:

            from gensim.models.ldamodel import LdaModel

            logger.info("LdaModel imported successfully")

            model = LdaModel(
                corpus=corpus,
                num_topics=n_topics,
                id2word=id2word,
                random_state=seed,
                update_every=1,
                chunksize=100,
                passes=10,
                alpha="auto",
                per_word_topics=True,
                **kwargs
            )

            logger.info("LdaModel trained successfully")

            progress.value += 1

    elif model == "lsi":

        from gensim.models.lsimodel import LsiModel

        logger.info("LsiModel imported successfully")

        model = LsiModel(corpus=corpus, num_topics=n_topics, id2word=id2word, **kwargs)

        logger.info("LsiModel trained successfully")

        progress.value += 1

    elif model == "hdp":

        from gensim.models import HdpModel

        logger.info("HdpModel imported successfully")

        model = HdpModel(
            corpus=corpus,
            id2word=id2word,
            random_state=seed,
            chunksize=100,
            T=n_topics,
            **kwargs
        )

        logger.info("HdpModel trained successfully")

        progress.value += 1

    elif model == "rp":

        from gensim.models import RpModel

        logger.info("RpModel imported successfully")

        model = RpModel(corpus=corpus, id2word=id2word, num_topics=n_topics, **kwargs)

        logger.info("RpModel trained successfully")

        progress.value += 1

    elif model == "nmf":

        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize

        logger.info(
            "CountVectorizer, TfidfTransformer, NMF, normalize imported successfully"
        )

        text_join = []

        for i in text:
            word = " ".join(i)
            text_join.append(word)

        progress.value += 1

        vectorizer = CountVectorizer(analyzer="word", max_features=5000)
        x_counts = vectorizer.fit_transform(text_join)
        logger.info("CountVectorizer() Fit Successfully")
        transformer = TfidfTransformer(smooth_idf=False)
        x_tfidf = transformer.fit_transform(x_counts)
        logger.info("TfidfTransformer() Fit Successfully")
        xtfidf_norm = normalize(x_tfidf, norm="l1", axis=1)
        model = NMF(n_components=n_topics, init="nndsvd", random_state=seed, **kwargs)
        model.fit(xtfidf_norm)
        logger.info("NMF() Trained Successfully")

    model_fit_end = time.time()
    model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

    progress.value += 1

    # end runtime
    runtime_end = time.time()
    runtime = np.array(runtime_end - runtime_start).round(2)

    # mlflow logging
    if logging_param and system:

        logger.info("Creating MLFLow Logs")

        # Creating Logs message monitor
        monitor.iloc[1, 1:] = "Creating Logs"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        # import mlflow
        import mlflow
        from pathlib import Path
        import os

        mlflow.set_experiment(exp_name_log)

        with mlflow.start_run(run_name=topic_model_name) as run:

            # Get active run to log as tag
            RunID = mlflow.active_run().info.run_id

            # Log model parameters
            from copy import deepcopy

            model_copied = deepcopy(model)

            try:
                params = model_copied.get_params()
            except:
                import inspect

                params = inspect.getmembers(model_copied)[2][1]

            for i in list(params):
                v = params.get(i)
                if len(str(v)) > 250:
                    params.pop(i)

            mlflow.log_params(params)

            # set tag of compare_models
            mlflow.set_tag("Source", "create_model")

            import secrets

            URI = secrets.token_hex(nbytes=4)
            mlflow.set_tag("URI", URI)
            mlflow.set_tag("USI", USI)
            mlflow.set_tag("Run Time", runtime)
            mlflow.set_tag("Run ID", RunID)

            # Log model and related artifacts
            if model_name_short == "nmf":
                logger.info(
                    "SubProcess save_model() called =================================="
                )
                save_model(model, "model", verbose=False)
                logger.info(
                    "SubProcess save_model() end =================================="
                )
                mlflow.log_artifact("model.pkl")
                size_bytes = Path("model.pkl").stat().st_size
                os.remove("model.pkl")

            elif model_name_short == "lda":
                model.save("model")
                mlflow.log_artifact("model")
                mlflow.log_artifact("model.expElogbeta.npy")
                mlflow.log_artifact("model.id2word")
                mlflow.log_artifact("model.state")
                size_bytes = (
                    Path("model").stat().st_size
                    + Path("model.id2word").stat().st_size
                    + Path("model.state").stat().st_size
                )
                os.remove("model")
                os.remove("model.expElogbeta.npy")
                os.remove("model.id2word")
                os.remove("model.state")

            elif model_name_short == "lsi":
                model.save("model")
                mlflow.log_artifact("model")
                mlflow.log_artifact("model.projection")
                size_bytes = (
                    Path("model").stat().st_size
                    + Path("model.projection").stat().st_size
                )
                os.remove("model")
                os.remove("model.projection")

            elif model_name_short == "rp":
                model.save("model")
                mlflow.log_artifact("model")
                size_bytes = Path("model").stat().st_size
                os.remove("model")

            elif model_name_short == "hdp":
                model.save("model")
                mlflow.log_artifact("model")
                size_bytes = Path("model").stat().st_size
                os.remove("model")

            size_kb = np.round(size_bytes / 1000, 2)
            mlflow.set_tag("Size KB", size_kb)

            # Log training time in seconds
            mlflow.log_metric("TT", model_fit_time)
            try:
                mlflow.log_metrics(model_results.to_dict().get("Metric"))
            except:
                pass

    # storing into experiment
    if verbose:
        clear_output()

    logger.info(str(model))
    logger.info(
        "create_model() succesfully completed......................................"
    )

    return model


def assign_model(model, verbose=True):

    """
    This function assigns topic labels to the dataset for a given model. 

    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')        
    >>> lda = create_model('lda')
    >>> lda_df = assign_model(lda)
    

    model: trained model object, default = None
        Trained model object


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.


    Returns:
        pandas.DataFrame
      
    """

    # exception checking
    import sys

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing assign_model()")
    logger.info(
        """assign_model(model={}, verbose={})""".format(str(model), str(verbose))
    )

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger.info("Determining model type")

    # determine model type
    if "LdaModel" in str(type(model)):
        mod_type = "lda"

    elif "LdaMulticore" in str(type(model)):
        mod_type = "lda"

    elif "LsiModel" in str(type(model)):
        mod_type = "lsi"

    elif "NMF" in str(type(model)):
        mod_type = "nmf"

    elif "HdpModel" in str(type(model)):
        mod_type = "hdp"

    elif "RpModel" in str(type(model)):
        mod_type = "rp"

    else:
        mod_type = None

    logger.info("model type: " + str(mod_type))

    """
    error handling starts here
    """

    logger.info("Checking exceptions")

    # checking for allowed models
    allowed_models = ["lda", "lsi", "hdp", "rp", "nmf"]

    if mod_type not in allowed_models:
        sys.exit(
            "(Value Error): Model Not Recognized. Please see docstring for list of available models."
        )

    # checking verbose parameter
    if type(verbose) is not bool:
        sys.exit(
            "(Type Error): Verbose parameter can only take argument as True or False."
        )

    """
    error handling ends here
    """

    logger.info("Preloading libraries")
    # pre-load libraries
    import numpy as np
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    logger.info("Preparing display monitor")
    # progress bar and monitor control
    max_progress = len(text) + 5
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(
        value=0, min=0, max=max_progress, step=1, description="Processing: "
    )
    monitor = pd.DataFrame(
        [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Initializing"],
        ],
        columns=["", " ", "  "],
    ).set_index("")
    if verbose:
        if html_param:
            display(progress)
            display(monitor, display_id="monitor")

    progress.value += 1

    monitor.iloc[1, 1:] = "Extracting Topics from Model"

    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    progress.value += 1

    # assignment starts here

    if mod_type == "lda":

        c = model.get_document_topics(corpus, minimum_probability=0)

        ls = []
        for i in range(len(c)):
            ls.append(c[i])
        bb = []
        for i in ls:
            bs = []
            for k in i:
                progress.value += 1
                bs.append(k[1])
            bb.append(bs)

        Dominant_Topic = []
        for i in bb:
            max_ = max(i)
            max_ = i.index(max_)
            Dominant_Topic.append("Topic " + str(max_))

        pdt = []
        for i in range(0, len(bb)):
            l = max(bb[i]) / sum(bb[i])
            pdt.append(round(l, 2))

        col_names = []
        for i in range(len(model.show_topics(num_topics=999999))):
            a = "Topic_" + str(i)
            col_names.append(a)

        progress.value += 1

        bb = pd.DataFrame(bb, columns=col_names)
        bb_ = pd.concat([data_, bb], axis=1)

        dt_ = pd.DataFrame(Dominant_Topic, columns=["Dominant_Topic"])
        bb_ = pd.concat([bb_, dt_], axis=1)

        pdt_ = pd.DataFrame(pdt, columns=["Perc_Dominant_Topic"])
        bb_ = pd.concat([bb_, pdt_], axis=1)

        progress.value += 1

        if verbose:
            clear_output()

    elif mod_type == "lsi":

        col_names = []
        for i in range(0, len(model.print_topics(num_topics=999999))):
            a = "Topic_" + str(i)
            col_names.append(a)

        df_ = pd.DataFrame()
        Dominant_Topic = []

        for i in range(0, len(text)):

            progress.value += 1
            db = id2word.doc2bow(text[i])
            db_ = model[db]
            db_array = np.array(db_)
            db_array_ = db_array[:, 1]

            max_ = max(db_array_)
            max_ = list(db_array_).index(max_)
            Dominant_Topic.append("Topic " + str(max_))

            db_df_ = pd.DataFrame([db_array_])
            df_ = pd.concat([df_, db_df_])

        progress.value += 1

        df_.columns = col_names

        df_["Dominant_Topic"] = Dominant_Topic
        df_ = df_.reset_index(drop=True)
        bb_ = pd.concat([data_, df_], axis=1)
        progress.value += 1

        if verbose:
            clear_output()

    elif mod_type == "hdp" or mod_type == "rp":

        rate = []
        for i in range(0, len(corpus)):
            progress.value += 1
            rate.append(model[corpus[i]])

        topic_num = []
        topic_weight = []
        doc_num = []
        counter = 0
        for i in rate:
            for k in i:
                topic_num.append(k[0])
                topic_weight.append(k[1])
                doc_num.append(counter)
            counter += 1
        progress.value += 1
        df = pd.DataFrame(
            {"Document": doc_num, "Topic": topic_num, "Topic Weight": topic_weight}
        ).sort_values(by="Topic")
        df = df.pivot(index="Document", columns="Topic", values="Topic Weight").fillna(
            0
        )
        df.columns = ["Topic_" + str(i) for i in df.columns]

        Dominant_Topic = []

        for i in range(0, len(df)):
            s = df.iloc[i].max()
            d = list(df.iloc[i]).index(s)
            v = df.columns[d]
            v = v.replace("_", " ")
            Dominant_Topic.append(v)

        df["Dominant_Topic"] = Dominant_Topic
        progress.value += 1

        if verbose:
            clear_output()

        bb_ = pd.concat([data_, df], axis=1)

    elif mod_type == "nmf":

        """
        this section will go away in future release through better handling
        """

        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize

        text_join = []

        for i in text:
            word = " ".join(i)
            text_join.append(word)

        progress.value += 1

        vectorizer = CountVectorizer(analyzer="word", max_features=5000)
        x_counts = vectorizer.fit_transform(text_join)
        transformer = TfidfTransformer(smooth_idf=False)
        x_tfidf = transformer.fit_transform(x_counts)
        xtfidf_norm = normalize(x_tfidf, norm="l1", axis=1)

        """
        section ends
        """

        bb = list(model.fit_transform(xtfidf_norm))

        col_names = []

        for i in range(len(bb[0])):
            a = "Topic_" + str(i)
            col_names.append(a)

        Dominant_Topic = []
        for i in bb:
            progress.value += 1
            max_ = max(i)
            max_ = list(i).index(max_)
            Dominant_Topic.append("Topic " + str(max_))

        pdt = []
        for i in range(0, len(bb)):
            l = max(bb[i]) / sum(bb[i])
            pdt.append(round(l, 2))

        progress.value += 1

        bb = pd.DataFrame(bb, columns=col_names)
        bb_ = pd.concat([data_, bb], axis=1)

        dt_ = pd.DataFrame(Dominant_Topic, columns=["Dominant_Topic"])
        bb_ = pd.concat([bb_, dt_], axis=1)

        pdt_ = pd.DataFrame(pdt, columns=["Perc_Dominant_Topic"])
        bb_ = pd.concat([bb_, pdt_], axis=1)

        progress.value += 1

        if verbose:
            clear_output()

    logger.info(str(bb_.shape))
    logger.info(
        "assign_model() succesfully completed......................................"
    )

    return bb_


def plot_model(model=None, plot="frequency", topic_num=None, save=False, system=True):

    """
    This function takes a trained model object (optional) and returns a plot based 
    on the inferred dataset by internally calling assign_model before generating a
    plot. Where a model parameter is not passed, a plot on the entire dataset will 
    be returned instead of one at the topic level. As such, plot_model can be used 
    with or without model. All plots with a model parameter passed as a trained 
    model object will return a plot based on the first topic i.e.  'Topic 0'. This 
    can be changed using the topic_num param. 


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp = setup(data = kiva, target = 'en')        
    >>> lda = create_model('lda')
    >>> plot_model(lda, plot = 'frequency')


    model: object, default = none
        Trained Model Object


    plot: str, default = 'frequency'
        List of available plots (ID - Name):

        * Word Token Frequency - 'frequency'              
        * Word Distribution Plot - 'distribution'
        * Bigram Frequency Plot - 'bigram' 
        * Trigram Frequency Plot - 'trigram'
        * Sentiment Polarity Plot - 'sentiment'
        * Part of Speech Frequency - 'pos'
        * t-SNE (3d) Dimension Plot - 'tsne'
        * Topic Model (pyLDAvis) - 'topic_model'
        * Topic Infer Distribution - 'topic_distribution'
        * Wordcloud - 'wordcloud'
        * UMAP Dimensionality Plot - 'umap'


    topic_num : str, default = None
        Topic number to be passed as a string. If set to None, default generation will 
        be on 'Topic 0'
    

    save: bool, default = False
        Plot is saved as png file in local directory when save parameter set to True.


    system: bool, default = True
        Must remain True all times. Only to be changed by internal functions.


    Returns:
        None


    Warnings
    --------
    -  'pos' and 'umap' plot not available at model level. Hence the model parameter is 
       ignored. The result will always be based on the entire training corpus.
    
    -  'topic_model' plot is based on pyLDAVis implementation. Hence its not available
       for model = 'lsi', 'rp' and 'nmf'.
         
    
    """

    # exception checking
    import sys

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing plot_model()")
    logger.info(
        """plot_model(model={}, plot={}, topic_num={}, save={}, system={})""".format(
            str(model), str(plot), str(topic_num), str(save), str(system)
        )
    )

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # setting default of topic_num
    if model is not None and topic_num is None:
        topic_num = "Topic 0"
        logger.info("Topic selected. topic_num : " + str(topic_num))

    """
    exception handling starts here
    """

    # determine model type

    if model is not None:

        mod = str(type(model))

        if "LdaModel" in mod:
            mod_type = "lda"

        elif "LdaMulticore" in str(type(model)):
            mod_type = "lda"

        elif "LsiModel" in str(type(model)):
            mod_type = "lsi"

        elif "NMF" in str(type(model)):
            mod_type = "nmf"

        elif "HdpModel" in str(type(model)):
            mod_type = "hdp"

        elif "RpModel" in str(type(model)):
            mod_type = "rp"

    logger.info("Checking exceptions")

    # plot checking
    allowed_plots = [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_model",
        "topic_distribution",
        "wordcloud",
        "umap",
    ]
    if plot not in allowed_plots:
        sys.exit(
            "(Value Error): Plot Not Available. Please see docstring for list of available plots."
        )

    # plots without topic model
    if model is None:
        not_allowed_wm = ["tsne", "topic_model", "topic_distribution"]
        if plot in not_allowed_wm:
            sys.exit(
                "(Type Error): Model parameter Missing. Plot not supported without specific model passed in as Model param."
            )

    # handle topic_model plot error
    if plot == "topic_model":
        not_allowed_tm = ["lsi", "rp", "nmf"]
        if mod_type in not_allowed_tm:
            sys.exit(
                "(Type Error): Model not supported for plot = topic_model. Please see docstring for list of available models supported for topic_model."
            )

    """
    error handling ends here
    """

    logger.info("Importing libraries")
    # import dependencies
    import pandas as pd
    import numpy

    # import cufflinks
    import cufflinks as cf

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    # save parameter

    if save:
        save_param = True
    else:
        save_param = False

    logger.info("save_param set to " + str(save_param))

    logger.info("plot type: " + str(plot))

    if plot == "frequency":

        try:

            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_words(corpus, n=None):
                vec = CountVectorizer()
                logger.info("Fitting CountVectorizer()")
                bag_of_words = vec.fit_transform(corpus)
                sum_words = bag_of_words.sum(axis=0)
                words_freq = [
                    (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
                ]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                return words_freq[:n]

            logger.info("Rendering Visual")

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                common_words = get_top_n_words(data_[target_], n=100)
                df2 = pd.DataFrame(common_words, columns=["Text", "count"])
                df3 = (
                    df2.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title="Top 100 words after removing stop words",
                        asFigure=save_param,
                    )
                )

            else:
                title = (
                    str(topic_num) + ": " + "Top 100 words after removing stop words"
                )
                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]
                common_words = get_top_n_words(filtered_df[target_], n=100)
                df2 = pd.DataFrame(common_words, columns=["Text", "count"])
                df3 = (
                    df2.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title=title,
                        asFigure=save_param,
                    )
                )

            logger.info("Visual Rendered Successfully")

            if save:
                df3.write_html("Word Frequency.html")
                logger.info("Saving 'Word Frequency.html' in current active directory")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "distribution":

        try:

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                b = data_[target_].apply(lambda x: len(str(x).split()))
                b = pd.DataFrame(b)
                logger.info("Rendering Visual")
                b = b[target_].iplot(
                    kind="hist",
                    bins=100,
                    xTitle="word count",
                    linecolor="black",
                    yTitle="count",
                    title="Word Count Distribution",
                    asFigure=save_param,
                )

            else:
                title = str(topic_num) + ": " + "Word Count Distribution"
                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]

                b = filtered_df[target_].apply(lambda x: len(str(x).split()))
                b = pd.DataFrame(b)
                logger.info("Rendering Visual")
                b = b[target_].iplot(
                    kind="hist",
                    bins=100,
                    xTitle="word count",
                    linecolor="black",
                    yTitle="count",
                    title=title,
                    asFigure=save_param,
                )

            logger.info("Visual Rendered Successfully")

            if save:
                b.write_html("Distribution.html")
                logger.info("Saving 'Distribution.html' in current active directory")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "bigram":

        try:

            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_bigram(corpus, n=None):
                logger.info("Fitting CountVectorizer()")
                vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0)
                words_freq = [
                    (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
                ]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                return words_freq[:n]

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                common_words = get_top_n_bigram(data_[target_], 100)
                df3 = pd.DataFrame(common_words, columns=["Text", "count"])
                logger.info("Rendering Visual")
                df3 = (
                    df3.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title="Top 100 bigrams after removing stop words",
                        asFigure=save_param,
                    )
                )

            else:
                title = (
                    str(topic_num) + ": " + "Top 100 bigrams after removing stop words"
                )
                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]
                common_words = get_top_n_bigram(filtered_df[target_], 100)
                df3 = pd.DataFrame(common_words, columns=["Text", "count"])
                logger.info("Rendering Visual")
                df3 = (
                    df3.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title=title,
                        asFigure=save_param,
                    )
                )

            logger.info("Visual Rendered Successfully")

            if save:
                df3.write_html("Bigram.html")
                logger.info("Saving 'Bigram.html' in current active directory")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "trigram":

        try:

            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_trigram(corpus, n=None):
                vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
                logger.info("Fitting CountVectorizer()")
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0)
                words_freq = [
                    (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
                ]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                return words_freq[:n]

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                common_words = get_top_n_trigram(data_[target_], 100)
                df3 = pd.DataFrame(common_words, columns=["Text", "count"])
                logger.info("Rendering Visual")
                df3 = (
                    df3.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title="Top 100 trigrams after removing stop words",
                        asFigure=save_param,
                    )
                )

            else:
                title = (
                    str(topic_num) + ": " + "Top 100 trigrams after removing stop words"
                )
                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]
                common_words = get_top_n_trigram(filtered_df[target_], 100)
                df3 = pd.DataFrame(common_words, columns=["Text", "count"])
                logger.info("Rendering Visual")
                df3 = (
                    df3.groupby("Text")
                    .sum()["count"]
                    .sort_values(ascending=False)
                    .iplot(
                        kind="bar",
                        yTitle="Count",
                        linecolor="black",
                        title=title,
                        asFigure=save_param,
                    )
                )

            logger.info("Visual Rendered Successfully")

            if save:
                df3.write_html("Trigram.html")
                logger.info("Saving 'Trigram.html' in current active directory")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "sentiment":

        try:

            # loadies dependencies
            import plotly.graph_objects as go
            from textblob import TextBlob

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                sentiments = data_[target_].map(
                    lambda text: TextBlob(text).sentiment.polarity
                )
                sentiments = pd.DataFrame(sentiments)
                logger.info("Rendering Visual")
                sentiments = sentiments[target_].iplot(
                    kind="hist",
                    bins=50,
                    xTitle="polarity",
                    linecolor="black",
                    yTitle="count",
                    title="Sentiment Polarity Distribution",
                    asFigure=save_param,
                )

            else:
                title = str(topic_num) + ": " + "Sentiment Polarity Distribution"
                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]
                sentiments = filtered_df[target_].map(
                    lambda text: TextBlob(text).sentiment.polarity
                )
                sentiments = pd.DataFrame(sentiments)
                logger.info("Rendering Visual")
                sentiments = sentiments[target_].iplot(
                    kind="hist",
                    bins=50,
                    xTitle="polarity",
                    linecolor="black",
                    yTitle="count",
                    title=title,
                    asFigure=save_param,
                )

            logger.info("Visual Rendered Successfully")

            if save:
                sentiments.write_html("Sentiments.html")
                logger.info("Saving 'Sentiments.html' in current active directory")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "pos":

        from textblob import TextBlob

        b = list(id2word.token2id.keys())
        logger.info("Fitting TextBlob()")
        blob = TextBlob(str(b))
        pos_df = pd.DataFrame(blob.tags, columns=["word", "pos"])
        pos_df = pos_df.loc[pos_df["pos"] != "POS"]
        pos_df = pos_df.pos.value_counts()[:20]
        logger.info("Rendering Visual")
        pos_df = pos_df.iplot(
            kind="bar",
            xTitle="POS",
            yTitle="count",
            title="Top 20 Part-of-speech tagging for review corpus",
            asFigure=save_param,
        )

        logger.info("Visual Rendered Sucessfully")

        if save:
            pos_df.write_html("POS.html")
            logger.info("Saving 'POS.html' in current active directory")

    elif plot == "tsne":

        logger.info(
            "SubProcess assign_model() called =================================="
        )
        b = assign_model(model, verbose=False)
        logger.info("SubProcess assign_model() end ==================================")
        b.dropna(axis=0, inplace=True)  # droping rows where Dominant_Topic is blank

        c = []
        for i in b.columns:
            if "Topic_" in i:
                a = i
                c.append(a)

        bb = b[c]

        from sklearn.manifold import TSNE

        logger.info("Fitting TSNE()")
        X_embedded = TSNE(n_components=3).fit_transform(bb)

        logger.info("Sorting Dataframe")
        X = pd.DataFrame(X_embedded)
        X["Dominant_Topic"] = b["Dominant_Topic"]
        X.sort_values(by="Dominant_Topic", inplace=True)
        X.dropna(inplace=True)

        logger.info("Rendering Visual")
        import plotly.express as px

        df = X
        fig = px.scatter_3d(
            df,
            x=0,
            y=1,
            z=2,
            color="Dominant_Topic",
            title="3d TSNE Plot for Topic Model",
            opacity=0.7,
            width=900,
            height=800,
        )

        if system:
            fig.show()

        logger.info("Visual Rendered Successfully")

        if save:
            fig.write_html("TSNE.html")
            logger.info("Saving 'TSNE.html' in current active directory")

    elif plot == "topic_model":

        import pyLDAvis
        import pyLDAvis.gensim  # don't skip this

        import warnings

        warnings.filterwarnings("ignore")
        pyLDAvis.enable_notebook()
        logger.info("Preparing pyLDAvis visual")
        vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds="mmds")
        display(vis)
        logger.info("Visual Rendered Successfully")

    elif plot == "topic_distribution":

        try:

            iter1 = len(model.show_topics(999999))

        except:

            try:
                iter1 = model.num_topics

            except:

                iter1 = model.n_components_

        topic_name = []
        keywords = []

        for i in range(0, iter1):

            try:

                s = model.show_topic(i, topn=10)
                topic_name.append("Topic " + str(i))

                kw = []

                for i in s:
                    kw.append(i[0])

                keywords.append(kw)

            except:

                keywords.append("NA")
                topic_name.append("Topic " + str(i))

        keyword = []
        for i in keywords:
            b = ", ".join(i)
            keyword.append(b)

        kw_df = pd.DataFrame({"Topic": topic_name, "Keyword": keyword}).set_index(
            "Topic"
        )
        logger.info(
            "SubProcess assign_model() called =================================="
        )
        ass_df = assign_model(model, verbose=False)
        logger.info("SubProcess assign_model() end ==================================")
        ass_df_pivot = ass_df.pivot_table(
            index="Dominant_Topic", values="Topic_0", aggfunc="count"
        )
        df2 = ass_df_pivot.join(kw_df)
        df2 = df2.reset_index()
        df2.columns = ["Topic", "Documents", "Keyword"]

        """
        sorting column starts
        
        """

        logger.info("Sorting Dataframe")

        topic_list = list(df2["Topic"])

        s = []
        for i in range(0, len(topic_list)):
            a = int(topic_list[i].split()[1])
            s.append(a)

        df2["Topic"] = s
        df2.sort_values(by="Topic", inplace=True)
        df2.sort_values(by="Topic", inplace=True)
        topic_list = list(df2["Topic"])
        topic_list = list(df2["Topic"])
        s = []
        for i in topic_list:
            a = "Topic " + str(i)
            s.append(a)

        df2["Topic"] = s
        df2.reset_index(drop=True, inplace=True)

        """
        sorting column ends
        """

        logger.info("Rendering Visual")

        import plotly.express as px

        fig = px.bar(
            df2,
            x="Topic",
            y="Documents",
            hover_data=["Keyword"],
            title="Document Distribution by Topics",
        )

        if system:
            fig.show()

        logger.info("Visual Rendered Successfully")

        if save:
            fig.write_html("Topic Distribution.html")
            logger.info("Saving 'Topic Distribution.html' in current active directory")

    elif plot == "wordcloud":

        try:

            from wordcloud import WordCloud, STOPWORDS
            import matplotlib.pyplot as plt

            stopwords = set(STOPWORDS)

            if topic_num is None:
                logger.warning("topic_num set to None. Plot generated at corpus level.")
                atext = " ".join(review for review in data_[target_])

            else:

                logger.info(
                    "SubProcess assign_model() called =================================="
                )
                assigned_df = assign_model(model, verbose=False)
                logger.info(
                    "SubProcess assign_model() end =================================="
                )
                filtered_df = assigned_df.loc[
                    assigned_df["Dominant_Topic"] == topic_num
                ]
                atext = " ".join(review for review in filtered_df[target_])

            logger.info("Fitting WordCloud()")
            wordcloud = WordCloud(
                width=800,
                height=800,
                background_color="white",
                stopwords=stopwords,
                min_font_size=10,
            ).generate(atext)

            # plot the WordCloud image
            plt.figure(figsize=(8, 8), facecolor=None)
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.tight_layout(pad=0)

            logger.info("Rendering Visual")

            if save or log_plots_param:
                if system:
                    plt.savefig("Wordcloud.png")
                else:
                    plt.savefig("Wordcloud.png")
                    plt.close()

                logger.info("Saving 'Wordcloud.png' in current active directory")

            else:
                plt.show()

            logger.info("Visual Rendered Successfully")

        except:
            logger.warning(
                "Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )
            sys.exit(
                "(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number."
            )

    elif plot == "umap":

        # warnings
        from matplotlib.axes._axes import _log as matplotlib_axes_logger

        matplotlib_axes_logger.setLevel("ERROR")

        # loading dependencies
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yellowbrick.text import UMAPVisualizer
        import matplotlib.pyplot as plt

        tfidf = TfidfVectorizer()
        logger.info("Fitting TfidfVectorizer()")
        docs = tfidf.fit_transform(data_[target_])

        # Instantiate the clustering model
        clusters = KMeans(n_clusters=5, random_state=seed)
        logger.info("Fitting KMeans()")
        clusters.fit(docs)

        plt.figure(figsize=(10, 6))

        umap = UMAPVisualizer(random_state=seed)
        logger.info("Fitting UMAP()")
        umap.fit(docs, ["c{}".format(c) for c in clusters.labels_])

        logger.info("Rendering Visual")

        if save or log_plots_param:
            if system:
                umap.show(outpath="UMAP.png")
            else:
                umap.show(outpath="UMAP.png", clear_figure=True)

            logger.info("Saving 'UMAP.png' in current active directory")

        else:
            umap.show()

        logger.info("Visual Rendered Successfully")

    logger.info(
        "plot_model() succesfully completed......................................"
    )


def tune_model(
    model=None,
    multi_core=False,
    supervised_target=None,
    estimator=None,
    optimize=None,
    custom_grid=None,
    auto_fe=True,
    fold=10,
    verbose=True,
):

    """
    This function tunes the ``num_topics`` parameter of a given model. 


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')        
    >>> tuned_lda = tune_model(model = 'lda', supervised_target = 'status') 


    model: str, default = None
        Enter ID of the models available in model library (ID - Model):

        * 'lda' - Latent Dirichlet Allocation         
        * 'lsi' - Latent Semantic Indexing           
        * 'hdp' - Hierarchical Dirichlet Process
        * 'rp' - Random Projections
        * 'nmf' - Non-Negative Matrix Factorization


    multi_core: bool, default = False
        True would utilize all CPU cores to parallelize and speed up model 
        training. Ignored when ``model`` is not 'lda'.


    supervised_target: str
        Name of the target column for supervised learning. If None, the model 
        coherence value is used as the objective function.


    estimator: str, default = None
        Classification (ID - Name):
            * 'lr' - Logistic Regression (Default)             
            * 'knn' - K Nearest Neighbour             
            * 'nb' - Naive Bayes                                 
            * 'dt' - Decision Tree Classifier                           
            * 'svm' - SVM - Linear Kernel             	            
            * 'rbfsvm' - SVM - Radial Kernel                            
            * 'gpc' - Gaussian Process Classifier                       
            * 'mlp' - Multi Level Perceptron                            
            * 'ridge' - Ridge Classifier                
            * 'rf' - Random Forest Classifier                           
            * 'qda' - Quadratic Discriminant Analysis                   
            * 'ada' - Ada Boost Classifier                             
            * 'gbc' - Gradient Boosting Classifier                              
            * 'lda' - Linear Discriminant Analysis                      
            * 'et' - Extra Trees Classifier                             
            * 'xgboost' - Extreme Gradient Boosting                     
            * 'lightgbm' - Light Gradient Boosting                       
            * 'catboost' - CatBoost Classifier             

        Regression (ID - Name):
            * 'lr' - Linear Regression (Default)                                
            * 'lasso' - Lasso Regression              
            * 'ridge' - Ridge Regression              
            * 'en' - Elastic Net                   
            * 'lar' - Least Angle Regression                
            * 'llar' - Lasso Least Angle Regression                     
            * 'omp' - Orthogonal Matching Pursuit                        
            * 'br' - Bayesian Ridge                                   
            * 'ard' - Automatic Relevance Determ.                     
            * 'par' - Passive Aggressive Regressor                      
            * 'ransac' - Random Sample Consensus              
            * 'tr' - TheilSen Regressor                               
            * 'huber' - Huber Regressor                                              
            * 'kr' - Kernel Ridge                                                       
            * 'svm' - Support Vector Machine                                   
            * 'knn' - K Neighbors Regressor                                    
            * 'dt' - Decision Tree                                                     
            * 'rf' - Random Forest                                                     
            * 'et' - Extra Trees Regressor                                     
            * 'ada' - AdaBoost Regressor                                               
            * 'gbr' - Gradient Boosting                                            
            * 'mlp' - Multi Level Perceptron                                  
            * 'xgboost' - Extreme Gradient Boosting                                   
            * 'lightgbm' - Light Gradient Boosting                           
            * 'catboost' - CatBoost Regressor               


    optimize: str, default = None
        For Classification tasks:
            Accuracy, AUC, Recall, Precision, F1, Kappa (default = 'Accuracy')
        
        For Regression tasks:
            MAE, MSE, RMSE, R2, RMSLE, MAPE (default = 'R2')


    custom_grid: list, default = None
        By default, a pre-defined number of topics is iterated over to 
        optimize the supervised objective. To overwrite default iteration,
        pass a list of num_topics to iterate over in custom_grid param.


    auto_fe: bool, default = True
        Automatic text feature engineering. When set to True, it will generate 
        text based features such as polarity, subjectivity, wordcounts. Ignored 
        when ``supervised_target`` is None.


    fold: int, default = 10
        Number of folds to be used in Kfold CV. Must be at least 2. 


    verbose: bool, default = True
        Status update is not printed when verbose is set to False.
    

    Returns:
        Trained Model with optimized ``num_topics`` parameter.


    Warnings
    --------
    - Random Projections ('rp') and Non Negative Matrix Factorization ('nmf')
      is not available for unsupervised learning. Error is raised when 'rp' or
      'nmf' is passed without supervised_target.

    - Estimators using kernel based methods such as Kernel Ridge Regressor, 
      Automatic Relevance Determinant, Gaussian Process Classifier, Radial Basis
      Support Vector Machine and Multi Level Perceptron may have longer training 
      times.
     
    
    """

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing tune_model()")
    logger.info(
        """tune_model(model={}, multi_core={}, supervised_target={}, estimator={}, optimize={}, custom_grid={}, auto_fe={}, fold={}, verbose={})""".format(
            str(model),
            str(multi_core),
            str(supervised_target),
            str(estimator),
            str(optimize),
            str(custom_grid),
            str(auto_fe),
            str(fold),
            str(verbose),
        )
    )

    logger.info("Checking exceptions")

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    import sys

    # checking for model parameter
    if model is None:
        sys.exit(
            "(Value Error): Model parameter Missing. Please see docstring for list of available models."
        )

    # checking for allowed models
    allowed_models = ["lda", "lsi", "hdp", "rp", "nmf"]

    if model not in allowed_models:
        sys.exit(
            "(Value Error): Model Not Available. Please see docstring for list of available models."
        )

    # checking multicore type:
    if type(multi_core) is not bool:
        sys.exit(
            "(Type Error): multi_core parameter can only take argument as True or False."
        )

    # check supervised target:
    if supervised_target is not None:
        all_col = list(data_.columns)
        target = target_
        all_col.remove(target)
        if supervised_target not in all_col:
            sys.exit(
                "(Value Error): supervised_target not recognized. It can only be one of the following: "
                + str(all_col)
            )

    # supervised target exception handling
    if supervised_target is None:
        models_not_allowed = ["rp", "nmf"]

        if model in models_not_allowed:
            sys.exit(
                "(Type Error): Model not supported for unsupervised tuning. Either supervised_target param has to be passed or different model has to be used. Please see docstring for available models."
            )

    # checking estimator:
    if estimator is not None:

        available_estimators = [
            "lr",
            "knn",
            "nb",
            "dt",
            "svm",
            "rbfsvm",
            "gpc",
            "mlp",
            "ridge",
            "rf",
            "qda",
            "ada",
            "gbc",
            "lda",
            "et",
            "lasso",
            "ridge",
            "en",
            "lar",
            "llar",
            "omp",
            "br",
            "ard",
            "par",
            "ransac",
            "tr",
            "huber",
            "kr",
            "svm",
            "knn",
            "dt",
            "rf",
            "et",
            "ada",
            "gbr",
            "mlp",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

        if estimator not in available_estimators:
            sys.exit(
                "(Value Error): Estimator Not Available. Please see docstring for list of available estimators."
            )

    # checking optimize parameter
    if optimize is not None:

        available_optimizers = [
            "MAE",
            "MSE",
            "RMSE",
            "R2",
            "ME",
            "Accuracy",
            "AUC",
            "Recall",
            "Precision",
            "F1",
            "Kappa",
        ]

        if optimize not in available_optimizers:
            sys.exit(
                "(Value Error): optimize parameter Not Available. Please see docstring for list of available parameters."
            )

    # checking auto_fe:
    if type(auto_fe) is not bool:
        sys.exit(
            "(Type Error): auto_fe parameter can only take argument as True or False."
        )

    # checking fold parameter
    if type(fold) is not int:
        sys.exit("(Type Error): Fold parameter only accepts integer value.")

    """
    exception handling ends here
    """

    logger.info("Preloading libraries")

    # pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from ipywidgets import Output
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    logger.info("Preparing display monitor")

    # progress bar
    if custom_grid is None:
        max_steps = 25
    else:
        max_steps = 10 + len(custom_grid)

    progress = ipw.IntProgress(
        value=0, min=0, max=max_steps, step=1, description="Processing: "
    )
    if verbose:
        if html_param:
            display(progress)

    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")

    monitor = pd.DataFrame(
        [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            ["Status", ". . . . . . . . . . . . . . . . . .", "Loading Dependencies"],
            ["Step", ". . . . . . . . . . . . . . . . . .", "Initializing"],
        ],
        columns=["", " ", "   "],
    ).set_index("")

    monitor_out = Output()

    if verbose:
        if html_param:
            display(monitor_out)

    if verbose:
        if html_param:
            with monitor_out:
                display(monitor, display_id="monitor")

    logger.info("Importing libraries")

    # General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px

    # setting up cufflinks
    import cufflinks as cf

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    progress.value += 1

    # define the problem
    if supervised_target is None:
        problem = "unsupervised"
        logger.info("Objective : Unsupervised")
    elif data_[supervised_target].value_counts().count() == 2:
        problem = "classification"
        logger.info("Objective : Classification")
    else:
        problem = "regression"
        logger.info("Objective : Regression")

    # define topic_model_name
    logger.info("Defining model name")

    if model == "lda":
        topic_model_name = "Latent Dirichlet Allocation"
    elif model == "lsi":
        topic_model_name = "Latent Semantic Indexing"
    elif model == "hdp":
        topic_model_name = "Hierarchical Dirichlet Process"
    elif model == "nmf":
        topic_model_name = "Non-Negative Matrix Factorization"
    elif model == "rp":
        topic_model_name = "Random Projections"

    logger.info("Topic Model Name: " + str(topic_model_name))

    # defining estimator:
    logger.info("Defining supervised estimator")
    if problem == "classification" and estimator is None:
        estimator = "lr"
    elif problem == "regression" and estimator is None:
        estimator = "lr"
    else:
        estimator = estimator

    logger.info("Estimator: " + str(estimator))

    # defining optimizer:
    logger.info("Defining Optimizer")
    if optimize is None and problem == "classification":
        optimize = "Accuracy"
    elif optimize is None and problem == "regression":
        optimize = "R2"
    else:
        optimize = optimize

    logger.info("Optimize: " + str(optimize))

    progress.value += 1

    # creating sentiments

    if problem == "classification" or problem == "regression":

        logger.info("Problem : Supervised")

        if auto_fe:

            logger.info("auto_fe param set to True")

            monitor.iloc[1, 1:] = "Feature Engineering"
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            from textblob import TextBlob

            monitor.iloc[2, 1:] = "Extracting Polarity"
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            logger.info("Extracting Polarity")
            polarity = data_[target_].map(
                lambda text: TextBlob(text).sentiment.polarity
            )

            monitor.iloc[2, 1:] = "Extracting Subjectivity"
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            logger.info("Extracting Subjectivity")
            subjectivity = data_[target_].map(
                lambda text: TextBlob(text).sentiment.subjectivity
            )

            monitor.iloc[2, 1:] = "Extracting Wordcount"
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            logger.info("Extracting Wordcount")
            word_count = [len(i) for i in text]

            progress.value += 1

    # defining tuning grid
    logger.info("Defining Tuning Grid")

    if custom_grid is not None:
        logger.info("Custom Grid used")
        param_grid = custom_grid

    else:
        logger.info("Pre-defined Grid used")
        param_grid = [2, 4, 8, 16, 32, 64, 100, 200, 300, 400]

    master = []
    master_df = []

    monitor.iloc[1, 1:] = "Creating Topic Model"
    if verbose:
        if html_param:
            update_display(monitor, display_id="monitor")

    for i in param_grid:
        logger.info("Fitting Model with num_topics = " + str(i))
        progress.value += 1
        monitor.iloc[2, 1:] = "Fitting Model With " + str(i) + " Topics"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        # create and assign the model to dataset d
        logger.info(
            "SubProcess create_model() called =================================="
        )
        m = create_model(
            model=model, multi_core=multi_core, num_topics=i, verbose=False
        )
        logger.info("SubProcess create_model() end ==================================")

        logger.info(
            "SubProcess assign_model() called =================================="
        )
        d = assign_model(m, verbose=False)
        logger.info("SubProcess assign_model() end ==================================")

        if problem in ["classification", "regression"] and auto_fe:
            d["Polarity"] = polarity
            d["Subjectivity"] = subjectivity
            d["word_count"] = word_count

        master.append(m)
        master_df.append(d)

        # topic model creation end's here

    if problem == "unsupervised":

        logger.info("Problem : Unsupervised")

        monitor.iloc[1, 1:] = "Evaluating Topic Model"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        from gensim.models import CoherenceModel

        logger.info("CoherenceModel imported successfully")

        coherence = []
        metric = []

        counter = 0

        for i in master:
            logger.info("Evaluating Coherence with num_topics: " + str(i))
            progress.value += 1
            monitor.iloc[2, 1:] = (
                "Evaluating Coherence With " + str(param_grid[counter]) + " Topics"
            )
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            model = CoherenceModel(
                model=i, texts=text, dictionary=id2word, coherence="c_v"
            )
            model_coherence = model.get_coherence()
            coherence.append(model_coherence)
            metric.append("Coherence")
            counter += 1

        monitor.iloc[1, 1:] = "Compiling Results"
        monitor.iloc[1, 1:] = "Finalizing"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        logger.info("Creating metrics dataframe")
        df = pd.DataFrame(
            {"# Topics": param_grid, "Score": coherence, "Metric": metric}
        )
        df.columns = ["# Topics", "Score", "Metric"]

        sorted_df = df.sort_values(by="Score", ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]

        logger.info("Rendering Visual")
        fig = px.line(
            df,
            x="# Topics",
            y="Score",
            line_shape="linear",
            title="Coherence Value and # of Topics",
            color="Metric",
        )

        fig.update_layout(plot_bgcolor="rgb(245,245,245)")

        fig.show()
        logger.info("Visual Rendered Successfully")

        # monitor = ''

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)["# Topics"])[0]
        best_m = round(np.array(sorted_df.head(1)["Score"])[0], 4)
        p = (
            "Best Model: "
            + topic_model_name
            + " |"
            + " # Topics: "
            + str(best_k)
            + " | "
            + "Coherence: "
            + str(best_m)
        )
        print(p)

    elif problem == "classification":

        logger.info("Importing untrained Classifier")

        """

        defining estimator

        """

        monitor.iloc[1, 1:] = "Evaluating Topic Model"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if estimator == "lr":

            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(random_state=seed)
            full_name = "Logistic Regression"

        elif estimator == "knn":

            from sklearn.neighbors import KNeighborsClassifier

            model = KNeighborsClassifier()
            full_name = "K Nearest Neighbours"

        elif estimator == "nb":

            from sklearn.naive_bayes import GaussianNB

            model = GaussianNB()
            full_name = "Naive Bayes"

        elif estimator == "dt":

            from sklearn.tree import DecisionTreeClassifier

            model = DecisionTreeClassifier(random_state=seed)
            full_name = "Decision Tree"

        elif estimator == "svm":

            from sklearn.linear_model import SGDClassifier

            model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
            full_name = "Support Vector Machine"

        elif estimator == "rbfsvm":

            from sklearn.svm import SVC

            model = SVC(
                gamma="auto", C=1, probability=True, kernel="rbf", random_state=seed
            )
            full_name = "RBF SVM"

        elif estimator == "gpc":

            from sklearn.gaussian_process import GaussianProcessClassifier

            model = GaussianProcessClassifier(random_state=seed)
            full_name = "Gaussian Process Classifier"

        elif estimator == "mlp":

            from sklearn.neural_network import MLPClassifier

            model = MLPClassifier(max_iter=500, random_state=seed)
            full_name = "Multi Level Perceptron"

        elif estimator == "ridge":

            from sklearn.linear_model import RidgeClassifier

            model = RidgeClassifier(random_state=seed)
            full_name = "Ridge Classifier"

        elif estimator == "rf":

            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=10, random_state=seed)
            full_name = "Random Forest Classifier"

        elif estimator == "qda":

            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

            model = QuadraticDiscriminantAnalysis()
            full_name = "Quadratic Discriminant Analysis"

        elif estimator == "ada":

            from sklearn.ensemble import AdaBoostClassifier

            model = AdaBoostClassifier(random_state=seed)
            full_name = "AdaBoost Classifier"

        elif estimator == "gbc":

            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(random_state=seed)
            full_name = "Gradient Boosting Classifier"

        elif estimator == "lda":

            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            model = LinearDiscriminantAnalysis()
            full_name = "Linear Discriminant Analysis"

        elif estimator == "et":

            from sklearn.ensemble import ExtraTreesClassifier

            model = ExtraTreesClassifier(random_state=seed)
            full_name = "Extra Trees Classifier"

        elif estimator == "xgboost":

            from xgboost import XGBClassifier

            model = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = "Extreme Gradient Boosting"

        elif estimator == "lightgbm":

            import lightgbm as lgb

            model = lgb.LGBMClassifier(random_state=seed)
            full_name = "Light Gradient Boosting Machine"

        elif estimator == "catboost":
            from catboost import CatBoostClassifier

            model = CatBoostClassifier(
                random_state=seed, silent=True
            )  # Silent is True to suppress CatBoost iteration results
            full_name = "CatBoost Classifier"

        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1

        """
        start model building here

        """

        acc = []
        auc = []
        recall = []
        prec = []
        kappa = []
        f1 = []

        for i in range(0, len(master_df)):
            progress.value += 1
            param_grid_val = param_grid[i]

            logger.info(
                "Training supervised model with num_topics: " + str(param_grid_val)
            )

            monitor.iloc[2, 1:] = (
                "Evaluating Classifier With " + str(param_grid_val) + " Topics"
            )
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            # prepare the dataset for supervised problem
            d = master_df[i]
            d.dropna(axis=0, inplace=True)  # droping rows where Dominant_Topic is blank
            d.drop([target_], inplace=True, axis=1)
            d = pd.get_dummies(d)

            # split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            # fit the model
            logger.info("Fitting Model")
            model.fit(X, y)

            # generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions")
            pred = cross_val_predict(model, X, y, cv=fold, method="predict")

            acc_ = metrics.accuracy_score(y, pred)
            acc.append(acc_)

            recall_ = metrics.recall_score(y, pred)
            recall.append(recall_)

            precision_ = metrics.precision_score(y, pred)
            prec.append(precision_)

            kappa_ = metrics.cohen_kappa_score(y, pred)
            kappa.append(kappa_)

            f1_ = metrics.f1_score(y, pred)
            f1.append(f1_)

            if hasattr(model, "predict_proba"):
                pred_ = cross_val_predict(model, X, y, cv=fold, method="predict_proba")
                pred_prob = pred_[:, 1]
                auc_ = metrics.roc_auc_score(y, pred_prob)
                auc.append(auc_)

            else:
                auc.append(0)

        monitor.iloc[1, 1:] = "Compiling Results"
        monitor.iloc[1, 1:] = "Finalizing"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        logger.info("Creating metrics dataframe")
        df = pd.DataFrame(
            {
                "# Topics": param_grid,
                "Accuracy": acc,
                "AUC": auc,
                "Recall": recall,
                "Precision": prec,
                "F1": f1,
                "Kappa": kappa,
            }
        )

        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        progress.value += 1

        logger.info("Rendering Visual")
        sd = pd.melt(
            df,
            id_vars=["# Topics"],
            value_vars=["Accuracy", "AUC", "Recall", "Precision", "F1", "Kappa"],
            var_name="Metric",
            value_name="Score",
        )

        fig = px.line(
            sd,
            x="# Topics",
            y="Score",
            color="Metric",
            line_shape="linear",
            range_y=[0, 1],
        )
        fig.update_layout(plot_bgcolor="rgb(245,245,245)")
        title = str(full_name) + " Metrics and # of Topics"
        fig.update_layout(
            title={
                "text": title,
                "y": 0.95,
                "x": 0.45,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

        fig.show()
        logger.info("Visual Rendered Successfully")

        # monitor = ''

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)["# Topics"])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0], 4)
        p = (
            "Best Model: "
            + topic_model_name
            + " |"
            + " # Topics: "
            + str(best_k)
            + " | "
            + str(optimize)
            + " : "
            + str(best_m)
        )
        print(p)

    elif problem == "regression":

        logger.info("Importing untrained Regressor")

        """

        defining estimator

        """

        monitor.iloc[1, 1:] = "Evaluating Topic Model"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        if estimator == "lr":

            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            full_name = "Linear Regression"

        elif estimator == "lasso":

            from sklearn.linear_model import Lasso

            model = Lasso(random_state=seed)
            full_name = "Lasso Regression"

        elif estimator == "ridge":

            from sklearn.linear_model import Ridge

            model = Ridge(random_state=seed)
            full_name = "Ridge Regression"

        elif estimator == "en":

            from sklearn.linear_model import ElasticNet

            model = ElasticNet(random_state=seed)
            full_name = "Elastic Net"

        elif estimator == "lar":

            from sklearn.linear_model import Lars

            model = Lars()
            full_name = "Least Angle Regression"

        elif estimator == "llar":

            from sklearn.linear_model import LassoLars

            model = LassoLars()
            full_name = "Lasso Least Angle Regression"

        elif estimator == "omp":

            from sklearn.linear_model import OrthogonalMatchingPursuit

            model = OrthogonalMatchingPursuit()
            full_name = "Orthogonal Matching Pursuit"

        elif estimator == "br":
            from sklearn.linear_model import BayesianRidge

            model = BayesianRidge()
            full_name = "Bayesian Ridge Regression"

        elif estimator == "ard":

            from sklearn.linear_model import ARDRegression

            model = ARDRegression()
            full_name = "Automatic Relevance Determination"

        elif estimator == "par":

            from sklearn.linear_model import PassiveAggressiveRegressor

            model = PassiveAggressiveRegressor(random_state=seed)
            full_name = "Passive Aggressive Regressor"

        elif estimator == "ransac":

            from sklearn.linear_model import RANSACRegressor

            model = RANSACRegressor(random_state=seed)
            full_name = "Random Sample Consensus"

        elif estimator == "tr":

            from sklearn.linear_model import TheilSenRegressor

            model = TheilSenRegressor(random_state=seed)
            full_name = "TheilSen Regressor"

        elif estimator == "huber":

            from sklearn.linear_model import HuberRegressor

            model = HuberRegressor()
            full_name = "Huber Regressor"

        elif estimator == "kr":

            from sklearn.kernel_ridge import KernelRidge

            model = KernelRidge()
            full_name = "Kernel Ridge"

        elif estimator == "svm":

            from sklearn.svm import SVR

            model = SVR()
            full_name = "Support Vector Regression"

        elif estimator == "knn":

            from sklearn.neighbors import KNeighborsRegressor

            model = KNeighborsRegressor()
            full_name = "Nearest Neighbors Regression"

        elif estimator == "dt":

            from sklearn.tree import DecisionTreeRegressor

            model = DecisionTreeRegressor(random_state=seed)
            full_name = "Decision Tree Regressor"

        elif estimator == "rf":

            from sklearn.ensemble import RandomForestRegressor

            model = RandomForestRegressor(random_state=seed)
            full_name = "Random Forest Regressor"

        elif estimator == "et":

            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(random_state=seed)
            full_name = "Extra Trees Regressor"

        elif estimator == "ada":

            from sklearn.ensemble import AdaBoostRegressor

            model = AdaBoostRegressor(random_state=seed)
            full_name = "AdaBoost Regressor"

        elif estimator == "gbr":

            from sklearn.ensemble import GradientBoostingRegressor

            model = GradientBoostingRegressor(random_state=seed)
            full_name = "Gradient Boosting Regressor"

        elif estimator == "mlp":

            from sklearn.neural_network import MLPRegressor

            model = MLPRegressor(random_state=seed)
            full_name = "MLP Regressor"

        elif estimator == "xgboost":

            from xgboost import XGBRegressor

            model = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = "Extreme Gradient Boosting Regressor"

        elif estimator == "lightgbm":

            import lightgbm as lgb

            model = lgb.LGBMRegressor(random_state=seed)
            full_name = "Light Gradient Boosting Machine"

        elif estimator == "catboost":
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(random_state=seed, silent=True)
            full_name = "CatBoost Regressor"

        logger.info(str(full_name) + " Imported Successfully")

        progress.value += 1

        """
        start model building here

        """

        score = []
        metric = []

        for i in range(0, len(master_df)):
            progress.value += 1
            param_grid_val = param_grid[i]

            logger.info(
                "Training supervised model with num_topics: " + str(param_grid_val)
            )

            monitor.iloc[2, 1:] = (
                "Evaluating Regressor With " + str(param_grid_val) + " Topics"
            )
            if verbose:
                if html_param:
                    update_display(monitor, display_id="monitor")

            # prepare the dataset for supervised problem
            d = master_df[i]
            d.dropna(axis=0, inplace=True)  # droping rows where Dominant_Topic is blank
            d.drop([target_], inplace=True, axis=1)
            d = pd.get_dummies(d)

            # split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            # fit the model
            logger.info("Fitting Model")
            model.fit(X, y)

            # generate the prediction and evaluate metric
            logger.info("Generating Cross Val Predictions")
            pred = cross_val_predict(model, X, y, cv=fold, method="predict")

            if optimize == "R2":
                r2_ = metrics.r2_score(y, pred)
                score.append(r2_)

            elif optimize == "MAE":
                mae_ = metrics.mean_absolute_error(y, pred)
                score.append(mae_)

            elif optimize == "MSE":
                mse_ = metrics.mean_squared_error(y, pred)
                score.append(mse_)

            elif optimize == "RMSE":
                mse_ = metrics.mean_squared_error(y, pred)
                rmse_ = np.sqrt(mse_)
                score.append(rmse_)

            elif optimize == "ME":
                max_error_ = metrics.max_error(y, pred)
                score.append(max_error_)

            metric.append(str(optimize))

        monitor.iloc[1, 1:] = "Compiling Results"
        monitor.iloc[1, 1:] = "Finalizing"
        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        logger.info("Creating metrics dataframe")
        df = pd.DataFrame({"# Topics": param_grid, "Score": score, "Metric": metric})
        df.columns = ["# Topics", optimize, "Metric"]

        # sorting to return best model
        if optimize == "R2":
            sorted_df = df.sort_values(by=optimize, ascending=False)
        else:
            sorted_df = df.sort_values(by=optimize, ascending=True)

        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]

        logger.info("Rendering Visual")

        fig = px.line(
            df,
            x="# Topics",
            y=optimize,
            line_shape="linear",
            title=str(full_name) + " Metrics and # of Topics",
            color="Metric",
        )

        fig.update_layout(plot_bgcolor="rgb(245,245,245)")

        progress.value += 1

        # monitor = ''

        if verbose:
            if html_param:
                update_display(monitor, display_id="monitor")

        monitor_out.clear_output()
        progress.close()

        fig.show()
        logger.info("Visual Rendered Successfully")

        best_k = np.array(sorted_df.head(1)["# Topics"])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0], 4)
        p = (
            "Best Model: "
            + topic_model_name
            + " |"
            + " # Topics: "
            + str(best_k)
            + " | "
            + str(optimize)
            + " : "
            + str(best_m)
        )
        print(p)

    logger.info(str(best_model))
    logger.info(
        "tune_model() succesfully completed......................................"
    )

    return best_model


def evaluate_model(model):

    """
    This function displays a user interface for analyzing performance of a trained
    model. It calls the ``plot_model`` function internally. 
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> experiment_name = setup(data = kiva, target = 'en')     
    >>> lda = create_model('lda')
    >>> evaluate_model(lda)
    

    model: object, default = none
        A trained model object should be passed. 


    Returns:
        None
           
    """

    from ipywidgets import widgets
    from ipywidgets.widgets import interact, fixed, interact_manual
    import numpy as np

    """
    generate sorted list
    
    """

    try:
        n_topic_assigned = len(model.show_topics())
    except:
        try:
            n_topic_assigned = model.num_topics
        except:
            n_topic_assigned = model.n_components

    final_list = []
    for i in range(0, n_topic_assigned):
        final_list.append("Topic " + str(i))

    a = widgets.ToggleButtons(
        options=[
            ("Frequency Plot", "frequency"),
            ("Bigrams", "bigram"),
            ("Trigrams", "trigram"),
            ("Sentiment Polarity", "sentiment"),
            ("Word Cloud", "wordcloud"),
        ],
        description="Plot Type:",
        disabled=False,
        button_style="",  # 'success', 'info', 'warning', 'danger' or ''
        icons=[""],
    )

    b = widgets.Dropdown(options=final_list, description="Topic #:", disabled=False)

    d = interact_manual(
        plot_model,
        model=fixed(model),
        plot=a,
        topic_num=b,
        save=fixed(False),
        system=fixed(True),
    )


def save_model(model, model_name, verbose=True):

    """
    This function saves the trained model object into the current active 
    directory as a pickle file for later use. 
    

    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> experiment_name = setup(data = kiva, target = 'en')
    >>> lda = create_model('lda')
    >>> save_model(lda, 'saved_lda_model')
    

    model: object
        A trained model object should be passed.
    

    model_name: str
        Name of pickle file to be passed as a string.


    verbose: bool, default = True
        When set to False, success message is not printed.


    Returns:
        Tuple of the model object and the filename.

    """
    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing save_model()")
    logger.info(
        """save_model(model={}, model_name={}, verbose={})""".format(
            str(model), str(model_name), str(verbose)
        )
    )

    import joblib

    model_name = model_name + ".pkl"
    joblib.dump(model, model_name)
    if verbose:
        print("Model Succesfully Saved")

    logger.info(str(model))
    logger.info(
        "save_model() succesfully completed......................................"
    )

    return (model, model_name)


def load_model(model_name, verbose=True):

    """
    This function loads a previously saved model.
    
    
    Example
    -------
    >>> from pycaret.nlp import load_model
    >>> saved_lda = load_model('saved_lda_model')
    

    model_name: str
        Name of pickle file to be passed as a string.


    verbose: bool, default = True
        When set to False, success message is not printed.


    Returns:
        Trained Model
         
    """

    import joblib

    model_name = model_name + ".pkl"
    if verbose:
        print("Model Sucessfully Loaded")
    return joblib.load(model_name)


def models():

    """
    Returns table of models available in model library.


    Example
    -------
    >>> from pycaret.nlp import models
    >>> all_models = models()


    Returns:
        pandas.DataFrame

    """

    import pandas as pd

    model_id = ["lda", "lsi", "hdp", "rp", "nmf"]

    model_name = [
        "Latent Dirichlet Allocation",
        "Latent Semantic Indexing",
        "Hierarchical Dirichlet Process",
        "Random Projections",
        "Non-Negative Matrix Factorization",
    ]

    model_ref = [
        "gensim/models/ldamodel",
        "gensim/models/lsimodel",
        "gensim/models/hdpmodel",
        "gensim/models/rpmodel",
        "sklearn.decomposition.NMF",
    ]

    df = pd.DataFrame({"ID": model_id, "Name": model_name, "Reference": model_ref})

    df.set_index("ID", inplace=True)

    return df


def get_logs(experiment_name=None, save=False):

    """
    Returns a table of experiment logs. Only works when ``log_experiment``
    is True when initializing the ``setup`` function.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en', log_experiment = True) 
    >>> lda = create_model('lda')
    >>> exp_logs = get_logs()


    experiment_name: str, default = None
        When None current active run is used.


    save: bool, default = False
        When set to True, csv file is saved in current working directory.


    Returns:
        pandas.DataFrame

    """

    import sys

    if experiment_name is None:
        exp_name_log_ = exp_name_log
    else:
        exp_name_log_ = experiment_name

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()

    if client.get_experiment_by_name(exp_name_log_) is None:
        sys.exit(
            "No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name."
        )

    exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id
    runs = mlflow.search_runs(exp_id)

    if save:
        file_name = str(exp_name_log_) + "_logs.csv"
        runs.to_csv(file_name, index=False)
    return runs


def get_config(variable):

    """
    This function retrieves the global variables created when initializing the 
    ``setup`` function. Following variables are accessible:

    - text: Tokenized words as a list with length = # documents
    - data_: pandas.DataFrame containing text after all processing
    - corpus: List containing tuples of id to word mapping
    - id2word: gensim.corpora.dictionary.Dictionary  
    - seed: random state set through session_id
    - target_: Name of column containing text. 'en' by default.
    - html_param: html_param configured through setup
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup

    
    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')
    >>> text = get_config('text') 


    Returns:
        Global variable

    """

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing get_config()")
    logger.info("""get_config(variable={})""".format(str(variable)))

    if variable == "text":
        global_var = text

    if variable == "data_":
        global_var = data_

    if variable == "corpus":
        global_var = corpus

    if variable == "id2word":
        global_var = id2word

    if variable == "seed":
        global_var = seed

    if variable == "target_":
        global_var = target_

    if variable == "html_param":
        global_var = html_param

    if variable == "exp_name_log":
        global_var = exp_name_log

    if variable == "logging_param":
        global_var = logging_param

    if variable == "log_plots_param":
        global_var = log_plots_param

    if variable == "USI":
        global_var = USI

    logger.info("Global variable: " + str(variable) + " returned")
    logger.info(
        "get_config() succesfully completed......................................"
    )

    return global_var


def set_config(variable, value):

    """
    This function resets the global variables. Following variables are 
    accessible:

    - text: Tokenized words as a list with length = # documents
    - data_: pandas.DataFrame containing text after all processing
    - corpus: List containing tuples of id to word mapping
    - id2word: gensim.corpora.dictionary.Dictionary 
    - seed: random state set through session_id
    - target_: Name of column containing text. 'en' by default.
    - html_param: html_param configured through setup
    - exp_name_log: Name of experiment set through setup
    - logging_param: log_experiment param set through setup
    - log_plots_param: log_plots param set through setup
    - USI: Unique session ID parameter set through setup


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> kiva = get_data('kiva')
    >>> from pycaret.nlp import *
    >>> exp_name = setup(data = kiva, target = 'en')
    >>> set_config('seed', 123)


    Returns:
        None

    """

    import logging

    try:
        hasattr(logger, "name")
    except:
        logger = logging.getLogger("logs")
        logger.setLevel(logging.DEBUG)

        # create console handler and set level to debug
        if logger.hasHandlers():
            logger.handlers.clear()

        ch = logging.FileHandler("logs.log")
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        logger.addHandler(ch)

    logger.info("Initializing set_config()")
    logger.info(
        """set_config(variable={}, value={})""".format(str(variable), str(value))
    )

    if variable == "text":
        global text
        text = value

    if variable == "data_":
        global data_
        data_ = value

    if variable == "corpus":
        global corpus
        corpus = value

    if variable == "id2word":
        global id2word
        id2word = value

    if variable == "seed":
        global seed
        seed = value

    if variable == "html_param":
        global html_param
        html_param = value

    if variable == "exp_name_log":
        global exp_name_log
        exp_name_log = value

    if variable == "logging_param":
        global logging_param
        logging_param = value

    if variable == "log_plots_param":
        global log_plots_param
        log_plots_param = value

    if variable == "USI":
        global USI
        USI = value

    logger.info("Global variable:  " + str(variable) + " updated")
    logger.info(
        "set_config() succesfully completed......................................"
    )


def get_topics(data, text, model=None, num_topics=4):

    """
    Callable from any external environment without requiring setup initialization.    
    """

    if model is None:
        model = "lda"

    s = setup(data=data, target=text)
    c = create_model(model=model, num_topics=num_topics, verbose=False)
    dataset = assign_model(c, verbose=False)
    return dataset
