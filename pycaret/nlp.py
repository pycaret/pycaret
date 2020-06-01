# Module: Natural Language Processing
# Author: Moez Ali <moez.ali@queensu.ca>
# License: MIT


def setup(data, 
          target=None,
          custom_stopwords=None,
          session_id = None):
    
    """
        
    Description:
    ------------
    This function initializes the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes one mandatory parameter:
    dataframe {array-like, sparse matrix} or object of type list. If a dataframe is 
    passed, target column containing text must be specified. When data passed is of 
    type list, no target parameter is required. All other parameters are optional. 
    This module only supports English Language at this time.

        Example
        -------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')

        'kiva' is a pandas Dataframe.
        
    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features) where n_samples 
    is the number of samples and n_features is the number of features or object of type
    list with n length.

    target: string
    If data is of type DataFrame, name of column containing text values must be passed as 
    string. 
    
    custom_stopwords: list, default = None
    list containing custom stopwords.

    session_id: int, default = None
    If None, a random seed is generated and returned in the Information grid. The 
    unique number is then distributed as a seed in all functions used during the 
    experiment. This can be used for later reproducibility of the entire experiment.


    Returns:
    --------

    info grid:    Information grid is printed.
    -----------      

    environment:  This function returns various outputs that are stored in variable
    -----------   as tuple. They are used by other functions in pycaret.

    Warnings:
    ---------
    - Some functionalities in pycaret.nlp requires you to have english language model. 
      The language model is not downloaded automatically when you install pycaret. 
      You will have to download two models using your Anaconda Prompt or python 
      command line interface. To download the model, please type the following in 
      your command line:
      
         python -m spacy download en_core_web_sm
         python -m textblob.download_corpora
    
      Once downloaded, please restart your kernel and re-run the setup.
        
          
    
    """
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    
    """
    error handling starts here
    """
    
    #checking data type
    if hasattr(data,'shape') is False:
        if type(data) is not list:   
            sys.exit('(Type Error): data passed must be of type pandas.DataFrame or list')  

    #if dataframe is passed then target is mandatory
    if hasattr(data,'shape'):
        if target is None:   
            sys.exit('(Type Error): When DataFrame is passed as data param. Target column containing text must be specified in target param.')  
            
    #checking target parameter
    if target is not None:
        if target not in data.columns:
            sys.exit('(Value Error): Target parameter doesnt exist in the data provided.')   

    #custom stopwords checking
    if custom_stopwords is not None:
        if type(custom_stopwords) is not list:
            sys.exit('(Type Error): custom_stopwords must be of list type.')  
    
    #checking session_id
    if session_id is not None:
        if type(session_id) is not int:
            sys.exit('(Type Error): session_id parameter must be an integer.')  
            
    #chcek if spacy is loaded 
    try:
        import spacy
        sp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    except:
        sys.exit('(Type Error): spacy english model is not yet downloaded. See the documentation of setup to see installation guide.')
    
    
    """
    error handling ends here
    """
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    '''
    generate monitor starts 
    '''
    
    #progress bar
    max_steps = 11
    total_steps = 9
        
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    display(progress)
    
    try:
        max_sub = len(data[target].values.tolist())
    except:
        max_sub = len(data)
        
    #sub_progress = ipw.IntProgress(value=0, min=0, max=max_sub, step=1, bar_style='', description='Sub Process: ')
    #display(sub_progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies' ],
                             ['Step' , '. . . . . . . . . . . . . . . . . .',  'Step 0 of ' + str(total_steps)] ],
                              columns=['', ' ', '   ']).set_index('')
    
    display(monitor, display_id = 'monitor')
    
    '''
    generate monitor end
    '''
    
    #general dependencies
    import numpy as np
    import random
    import spacy
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    import spacy
    import re

    
    #defining global variables
    global text, id2word, corpus, data_, seed, target_, experiment__
    
    #create an empty list for pickling later.
    try:
        experiment__.append('dummy')
        experiment__.pop()
    
    except:
        experiment__ = []
    
    #converting to dataframe if list provided
    if type(data) is list:
        data = pd.DataFrame(data, columns=['en'])
        target = 'en'

    #converting target column into list
    try:
        text = data[target].values.tolist()
        target_ = str(target)
    except:
        text = data
        target_ = 'en'
    
    #generate seed to be used globally
    if session_id is None:
        seed = random.randint(150,9000)
    else:
        seed = session_id
    
    #copying dataframe
    if type(data) is list:
        data_ = pd.DataFrame(data)
        data_.columns = ['en']
    else: 
        data_ = data.copy()
    
    progress.value += 1

    
    """
    DEFINE STOPWORDS
    """
    try:
        import nltk
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        
    except:
        stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 
                      'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 
                      'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 
                      'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 
                      'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 
                      'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 
                      'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 
                      'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 
                      'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 
                      'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

    
    if custom_stopwords is not None:
        stop_words = stop_words + custom_stopwords
        
    progress.value += 1
    
    
    """
    TEXT PRE-PROCESSING STARTS HERE
    """
    
    """
    STEP 1 - REMOVE NUMERIC CHARACTERS FROM THE LIST
    """
    
    monitor.iloc[1,1:] = 'Removing Numeric Characters'
    monitor.iloc[2,1:] = 'Step 1 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
            
    text_step1 = []
    
    for i in range(0,len(text)):
        review = re.sub("\d+", "", str(text[i]))
        text_step1.append(review)
        
        #sub_progress.value += 1
        
    #sub_progress.value = 0

    text = text_step1 #re-assigning
    del(text_step1)
    
    progress.value += 1
    
    """
    STEP 2 - REGULAR EXPRESSIONS
    """    
    
    monitor.iloc[1,1:] = 'Removing Special Characters'
    monitor.iloc[2,1:] = 'Step 2 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    text_step2 = []
    
    for i in range(0,len(text)):
        review = re.sub(r'\W', ' ', str(text[i]))
        review = review.lower()
        review = re.sub(r'\s+[a-z]\s+', ' ', review)
        review = re.sub(r'^[a-z]\s+', ' ', review)
        review = re.sub(r'\d+', ' ', review)
        review = re.sub(r'\s+', ' ', review)
        text_step2.append(review)
        
        #sub_progress.value += 1
        
    #sub_progress.value = 0
    
    text = text_step2 #re-assigning
    del(text_step2)
    
    progress.value += 1
    
    """
    STEP 3 - WORD TOKENIZATION
    """ 
    
    monitor.iloc[1,1:] = 'Tokenizing Words'
    monitor.iloc[2,1:] = 'Step 3 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    text_step3 = []
    
    for i in text:
        review = gensim.utils.simple_preprocess(str(i), deacc=True)
        text_step3.append(review)
        
        #sub_progress.value += 1
        
    #sub_progress.value = 0
    
    text = text_step3
    del(text_step3)
    
    progress.value += 1
    
    """
    STEP 4 - REMOVE STOPWORDS
    """
    
    monitor.iloc[1,1:] = 'Removing Stopwords'
    monitor.iloc[2,1:] = 'Step 4 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    text_step4 = []
    
    for i in text:
        ii = []
        for word in i:
            if word not in stop_words:
                ii.append(word)
        text_step4.append(ii)
        
        #sub_progress.value += 1
        
    text = text_step4
    del(text_step4)
        
    #sub_progress.value = 0
            
    progress.value += 1
    
    """
    STEP 5 - BIGRAM EXTRACTION
    """    
    
    monitor.iloc[1,1:] = 'Extracting Bigrams'
    monitor.iloc[2,1:] = 'Step 5 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    bigram = gensim.models.Phrases(text, min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    text_step5 = []
    
    for i in text:
        text_step5.append(bigram_mod[i])
        #sub_progress.value += 1
        
    text = text_step5
    del(text_step5)
        
    #sub_progress.value = 0
    
    progress.value += 1
    
    """
    STEP 6 - TRIGRAM EXTRACTION
    """ 
    
    monitor.iloc[1,1:] = 'Extracting Trigrams'
    monitor.iloc[2,1:] = 'Step 6 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    trigram = gensim.models.Phrases(bigram[text], threshold=100)  
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    text_step6 = []

    for i in text:
        text_step6.append(trigram_mod[bigram_mod[i]])
        #sub_progress.value += 1
    
    #sub_progress.value = 0
    
    text = text_step6
    del(text_step6)
    
    progress.value += 1
    
    """
    STEP 7 - LEMMATIZATION USING SPACY
    """     
    
    monitor.iloc[1,1:] = 'Lemmatizing'
    monitor.iloc[2,1:] = 'Step 7 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']
    
    text_step7 = []
    
    for i in text:
        doc = nlp(" ".join(i))
        text_step7.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
        #sub_progress.value += 1
        
    #sub_progress.value = 0
    text = text_step7
    del(text_step7)
    
    progress.value += 1
    
    
    """
    STEP 8  - CUSTOM STOPWORD REMOVER
    """
    
    monitor.iloc[1,1:] = 'Removing Custom Stopwords'
    monitor.iloc[2,1:] = 'Step 8 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    text_step8 = []
    
    for i in text:
        ii = []
        for word in i:
            if word not in stop_words:
                ii.append(word)
        text_step8.append(ii)
        
        #sub_progress.value += 1
        
    text = text_step8
    del(text_step8)
        
    #sub_progress.value = 0
            
    progress.value += 1
    
    
    """
    STEP 8 - CREATING CORPUS AND DICTIONARY
    """  
    
    monitor.iloc[1,1:] = 'Compiling Corpus'
    monitor.iloc[2,1:] = 'Step 9 of '+ str(total_steps)
    update_display(monitor, display_id = 'monitor')
    
    #creating dictionary
    id2word = corpora.Dictionary(text)
    
    #creating corpus
    
    corpus = []
    for i in text:
        d = id2word.doc2bow(i)
        corpus.append(d)
        
        #sub_progress.value += 1
        
    #sub_progress.value = 0
        
    progress.value += 1
    
    """
    PROGRESS NOT YET TRACKED - TO BE CODED LATER
    """
    
    text_join = []

    for i in text:
        word = ' '.join(i)
        text_join.append(word)
        
    data_[target_] = text_join
    
    '''
    Final display Starts
    '''
    clear_output()
    
    if custom_stopwords is None:
        csw = False
    else:
        csw = True
        
    functions = pd.DataFrame ( [ ['session_id', seed ],
                                 ['# Documents', len(corpus) ], 
                                 ['Vocab Size',len(id2word.keys()) ],
                                 ['Custom Stopwords',csw ],
                               ], columns = ['Description', 'Value'] )

    functions_ = functions.style.hide_index()
    display(functions_)

    '''
    Final display Ends
    '''   

    #log into experiment
    experiment__.append(('Info', functions))
    experiment__.append(('Dataset', data_))
    experiment__.append(('Corpus', corpus))
    experiment__.append(('Dictionary', id2word))
    experiment__.append(('Text', text))

    return text, data_, corpus, id2word, seed, target_, experiment__




def create_model(model=None,
                 multi_core=False,
                 num_topics = None,
                 verbose=True):
    
    """  
     
    Description:
    ------------
    This function creates a model on the dataset passed as a data param during 
    the setup stage. setup() function must be called before using create_model().

    This function returns a trained model object. 

        Example
        -------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')
        
        lda = create_model('lda')

        This will return trained Latent Dirichlet Allocation model.

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated string of the model class. List of models supported:

    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    Latent Dirichlet Allocation        'lda'                gensim/models/ldamodel.html
    Latent Semantic Indexing           'lsi'                gensim/models/lsimodel.html
    Hierarchical Dirichlet Process     'hdp'                gensim/models/hdpmodel.html
    Random Projections                 'rp'                 gensim/models/rpmodel.html
    Non-Negative Matrix Factorization  'nmf'                sklearn.decomposition.NMF.html
   
    multi_core: Boolean, default = False
    True would utilize all CPU cores to parallelize and speed up model training. Only
    available for 'lda'. For all other models, the multi_core parameter is ignored.

    num_topics: integer, default = 4
    Number of topics to be created. If None, default is set to 4.

    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    model:   trained model object
    ------

    Warnings:
    ---------
    None
      
    
     
    """
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    """
    error handling starts here
    """
    
    #checking for model parameter
    if model is None:
        sys.exit('(Value Error): Model parameter Missing. Please see docstring for list of available models.')
        
    #checking for allowed models
    allowed_models = ['lda', 'lsi', 'hdp', 'rp', 'nmf']
    
    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')
        
    #checking multicore type:
    if type(multi_core) is not bool:
        sys.exit('(Type Error): multi_core parameter can only take argument as True or False.')
        
    #checking round parameter
    if num_topics is not None:
        if type(num_topics) is not int:
            sys.exit('(Type Error): num_topics parameter only accepts integer value.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.') 
        
    """
    error handling ends here
    """
    
    
    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time
    
    """
    monitor starts
    """
    
    #progress bar and monitor control    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=4, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    """
    monitor starts
    """
        
    #define topic_model_name
    if model == 'lda':
        topic_model_name = 'Latent Dirichlet Allocation'
    elif model == 'lsi':
        topic_model_name = 'Latent Semantic Indexing'
    elif model == 'hdp':
        topic_model_name = 'Hierarchical Dirichlet Process'
    elif model == 'nmf':
        topic_model_name = 'Non-Negative Matrix Factorization'
    elif model == 'rp':
        topic_model_name = 'Random Projections'
        
        
    #defining default number of topics
    if num_topics is None:
        n_topics = 4
    else:
        n_topics = num_topics
    

    #monitor update
    monitor.iloc[1,1:] = 'Fitting Topic Model'
    progress.value += 1
    if verbose:
        update_display(monitor, display_id = 'monitor')
    
    if model == 'lda':
        
        if multi_core:
            
            from gensim.models.ldamulticore import LdaMulticore

            model = LdaMulticore(corpus=corpus,
                                num_topics=n_topics,
                                id2word=id2word,
                                workers=4,
                                random_state=seed,
                                chunksize=100,
                                passes=10,
                                alpha= 'symmetric',
                                per_word_topics=True)
            
            progress.value += 1
        
        else:
            
            from gensim.models.ldamodel import LdaModel
        
            model = LdaModel(corpus=corpus,
                            num_topics=n_topics,
                            id2word=id2word,
                            random_state=seed,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='auto',
                            per_word_topics=True)
            
            progress.value += 1
            
    elif model == 'lsi':
        
        from gensim.models.lsimodel import LsiModel
        
        model = LsiModel(corpus=corpus, 
                         num_topics=n_topics, 
                         id2word=id2word)
        
        progress.value += 1

    elif model == 'hdp':
        
        from gensim.models import HdpModel
        
        model = HdpModel(corpus=corpus, 
                         id2word=id2word, 
                         random_state=seed, 
                         chunksize=100,
                         T=n_topics)
        
        progress.value += 1
        
    elif model == 'rp':
        
        from gensim.models import RpModel
        
        model = RpModel(corpus=corpus, 
                        id2word=id2word, 
                        num_topics=n_topics)
        
        progress.value += 1
        
    elif model == 'nmf':
        
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize
        
        text_join = []

        for i in text:
            word = ' '.join(i)
            text_join.append(word)
        
        progress.value += 1
        
        vectorizer = CountVectorizer(analyzer='word', max_features=5000)
        x_counts = vectorizer.fit_transform(text_join)
        transformer = TfidfTransformer(smooth_idf=False);
        x_tfidf = transformer.fit_transform(x_counts);
        xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
        model = NMF(n_components=n_topics, init='nndsvd', random_state=seed);
        model.fit(xtfidf_norm)
        
    progress.value += 1
    
    #storing into experiment
    if verbose:
        clear_output()
        tup = (topic_model_name,model)
        experiment__.append(tup)  
    
    return model



def assign_model(model,
                 verbose=True):
    
    
    """  
     
    Description:
    ------------
    This function assigns each of the data point in the dataset passed during setup
    stage to one of the topic using trained model object passed as model param.
    create_model() function must be called before using assign_model().
    
    This function returns dataframe with topic weights, dominant topic and % of the
    dominant topic (where applicable).

        Example
        -------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')        
        lda = create_model('lda')
        
        lda_df = assign_model(lda)
        
        This will return a dataframe with inferred topics using trained model.

    Parameters
    ----------
    model : trained model object, default = None

    verbose: Boolean, default = True
    Status update is not printed when verbose is set to False.

    Returns:
    --------

    dataframe:   Returns dataframe with inferred topics using trained model object.
    ---------

    Warnings:
    ---------
    None
      
    
    """
    
    #determine model type
    if 'LdaModel' in str(type(model)):
        mod_type = 'lda'
        
    elif 'LdaMulticore' in str(type(model)):
        mod_type = 'lda'
        
    elif 'LsiModel' in str(type(model)):
        mod_type = 'lsi'
        
    elif 'NMF' in str(type(model)):
        mod_type = 'nmf'
    
    elif 'HdpModel' in str(type(model)):
        mod_type = 'hdp'
        
    elif 'RpModel' in str(type(model)):
        mod_type = 'rp'
        
    else:
        mod_type = None
        
        
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
    
    
    """
    error handling starts here
    """
    
    #checking for allowed models
    allowed_models = ['lda', 'lsi', 'hdp', 'rp', 'nmf']
    
    if mod_type not in allowed_models:
        sys.exit('(Value Error): Model Not Recognized. Please see docstring for list of available models.')
        
    #checking verbose parameter
    if type(verbose) is not bool:
        sys.exit('(Type Error): Verbose parameter can only take argument as True or False.')     
    
    
    """
    error handling ends here
    """
    
    #pre-load libraries
    import numpy as np
    import pandas as pd
    import ipywidgets as ipw
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    #progress bar and monitor control 
    max_progress = len(text) + 5
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
    progress = ipw.IntProgress(value=0, min=0, max=max_progress, step=1 , description='Processing: ')
    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                              ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Initializing'] ],
                              columns=['', ' ', '  ']).set_index('')
    if verbose:
        display(progress)
        display(monitor, display_id = 'monitor')
        
    progress.value += 1
    
    monitor.iloc[1,1:] = 'Extracting Topics from Model'
    
    if verbose:
        update_display(monitor, display_id = 'monitor')
    
    progress.value += 1
    
    #assignment starts here
    
    if mod_type == 'lda':
        
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
            Dominant_Topic.append('Topic ' + str(max_))

        pdt = []
        for i in range(0,len(bb)):
            l = max(bb[i]) / sum(bb[i])
            pdt.append(round(l,2))

        col_names = []
        for i in range(len(model.show_topics(num_topics=999999))):
            a = 'Topic_' + str(i)
            col_names.append(a)
            
        progress.value += 1

        bb = pd.DataFrame(bb,columns=col_names)
        bb_ = pd.concat([data_,bb], axis=1)

        dt_ = pd.DataFrame(Dominant_Topic, columns=['Dominant_Topic'])
        bb_ = pd.concat([bb_,dt_], axis=1)

        pdt_ = pd.DataFrame(pdt, columns=['Perc_Dominant_Topic'])
        bb_ = pd.concat([bb_,pdt_], axis=1)
        
        progress.value += 1
        
        if verbose:
            clear_output()
            
        #return bb_
    
    elif mod_type == 'lsi':
        
        col_names = []
        for i in range(0,len(model.print_topics(num_topics=999999))):
            a = 'Topic_' + str(i)
            col_names.append(a)
        
        df_ = pd.DataFrame()
        Dominant_Topic = []
        
        for i in range(0,len(text)):
            
            progress.value += 1
            db = id2word.doc2bow(text[i])
            db_ = model[db]
            db_array = np.array(db_)
            db_array_ = db_array[:,1]
            
            max_ = max(db_array_)
            max_ = list(db_array_).index(max_)
            Dominant_Topic.append('Topic ' + str(max_))
            
            db_df_ = pd.DataFrame([db_array_])
            df_ = pd.concat([df_,db_df_])
        
        progress.value += 1
        
        df_.columns = col_names
        
        df_['Dominant_Topic'] = Dominant_Topic
        df_ = df_.reset_index(drop=True)
        bb_ = pd.concat([data_,df_], axis=1)
        progress.value += 1
        
        if verbose:
            clear_output()
            
        #return bb_
    
    elif mod_type == 'hdp' or mod_type == 'rp':
        
        rate = []
        for i in range(0,len(corpus)):
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
        df = pd.DataFrame({'Document': doc_num, 'Topic' : topic_num, 'Topic Weight' : topic_weight}).sort_values(by='Topic')
        df = df.pivot(index='Document', columns='Topic', values='Topic Weight').fillna(0)
        df.columns = ['Topic_' + str(i) for i in df.columns]

        Dominant_Topic = []

        for i in range(0,len(df)):
            s = df.iloc[i].max()
            d = list(df.iloc[i]).index(s)
            v = df.columns[d]
            v = v.replace("_", ' ')
            Dominant_Topic.append(v)

        df['Dominant_Topic'] = Dominant_Topic
        progress.value += 1
        
        if verbose:
            clear_output()
            
        bb_ = pd.concat([data_,df], axis=1)
        
        #return bb_
    
    elif mod_type == 'nmf':
        
        """
        this section will go away in future release through better handling
        """
        
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.decomposition import NMF
        from sklearn.preprocessing import normalize
        
        text_join = []

        for i in text:
            word = ' '.join(i)
            text_join.append(word)
        
        progress.value += 1
        
        vectorizer = CountVectorizer(analyzer='word', max_features=5000)
        x_counts = vectorizer.fit_transform(text_join)
        transformer = TfidfTransformer(smooth_idf=False);
        x_tfidf = transformer.fit_transform(x_counts);
        xtfidf_norm = normalize(x_tfidf, norm='l1', axis=1)
        
        """
        section ends
        """
        
        
        bb = list(model.fit_transform(xtfidf_norm))
        
        col_names = []
               
        for i in range(len(bb[0])):
            a = 'Topic_' + str(i)
            col_names.append(a)
        
        Dominant_Topic = []
        for i in bb:
            progress.value += 1
            max_ = max(i)
            max_ = list(i).index(max_)
            Dominant_Topic.append('Topic ' + str(max_))
            
        pdt = []
        for i in range(0,len(bb)):
            l = max(bb[i]) / sum(bb[i])
            pdt.append(round(l,2))
        
        progress.value += 1
        
        bb = pd.DataFrame(bb, columns=col_names)
        bb_ = pd.concat([data_,bb], axis=1)
        
        dt_ = pd.DataFrame(Dominant_Topic, columns=['Dominant_Topic'])
        bb_ = pd.concat([bb_,dt_], axis=1)

        pdt_ = pd.DataFrame(pdt, columns=['Perc_Dominant_Topic'])
        bb_ = pd.concat([bb_,pdt_], axis=1)
        
        progress.value += 1
        
        if verbose:
            clear_output()
        
    #storing into experiment
    if verbose:
        clear_output()
        mod__ = str(mod_type) + ' Topic Assignment'
        tup = (mod__,bb_)
        experiment__.append(tup) 
        #return bb_
    
    return bb_



def plot_model(model = None,
               plot = 'frequency',
               topic_num = None):
    
    
    """
          
    Description:
    ------------
    This function takes a trained model object (optional) and returns a plot based 
    on the inferred dataset by internally calling assign_model before generating a
    plot. Where a model parameter is not passed, a plot on the entire dataset will 
    be returned instead of one at the topic level. As such, plot_model can be used 
    with or without model. All plots with a model parameter passed as a trained 
    model object will return a plot based on the first topic i.e.  'Topic 0'. This 
    can be changed using the topic_num param. 

        Example:
        --------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')        
        lda = create_model('lda')
        
        plot_model(lda, plot = 'frequency')

        This will return a frequency plot on a trained Latent Dirichlet Allocation 
        model for all documents in 'Topic 0'. The topic number can be changed as 
        follows:
        
        plot_model(lda, plot = 'frequency', topic_num = 'Topic 1')
        
        This will now return a frequency plot on a trained LDA model for all 
        documents inferred in 'Topic 1'.
        
        Alternatively, if following is used:
        
        plot_model(plot = 'frequency')
        
        This will return frequency plot on the entire training corpus compiled 
        during setup stage.

    Parameters
    ----------
    model : object, default = none
    A trained model object can be passed. Model must be created using create_model().

    plot : string, default = 'frequency'
    Enter abbreviation for type of plot. The current list of plots supported are:

    Name                           Abbreviated String     
    ---------                      ------------------     
    Word Token Frequency           'frequency'              
    Word Distribution Plot         'distribution'
    Bigram Frequency Plot          'bigram' 
    Trigram Frequency Plot         'trigram'
    Sentiment Polarity Plot        'sentiment'
    Part of Speech Frequency       'pos'
    t-SNE (3d) Dimension Plot      'tsne'
    Topic Model (pyLDAvis)         'topic_model'
    Topic Infer Distribution       'topic_distribution'
    Wordcloud                      'wordcloud'
    UMAP Dimensionality Plot       'umap'

    topic_num : string, default = None
    Topic number to be passed as a string. If set to None, default generation will 
    be on 'Topic 0'
    
    Returns:
    --------

    Visual Plot:  Prints the visual plot. 
    ------------

    Warnings:
    ---------
    -  'pos' and 'umap' plot not available at model level. Hence the model parameter is 
       ignored. The result will always be based on the entire training corpus.
    
    -  'topic_model' plot is based on pyLDAVis implementation. Hence its not available
       for model = 'lsi', 'rp' and 'nmf'.
         
                

    """  
    
    #exception checking   
    import sys
    
    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 
        
    #setting default of topic_num
    if model is not None and topic_num is None:
        topic_num = 'Topic 0'
        
    """
    exception handling starts here
    """
    
    #determine model type
    
    if model is not None: 
        
        mod = str(type(model))
        
        if 'LdaModel' in mod:
            mod_type = 'lda'

        elif 'LdaMulticore' in str(type(model)):
            mod_type = 'lda'

        elif 'LsiModel' in str(type(model)):
            mod_type = 'lsi'

        elif 'NMF' in str(type(model)):
            mod_type = 'nmf'

        elif 'HdpModel' in str(type(model)):
            mod_type = 'hdp'

        elif 'RpModel' in str(type(model)):
            mod_type = 'rp'
            
    
    #plot checking
    allowed_plots = ['frequency', 'distribution', 'bigram', 'trigram', 'sentiment', 'pos', 'tsne', 'topic_model', 
                     'topic_distribution', 'wordcloud', 'umap']  
    if plot not in allowed_plots:
        sys.exit('(Value Error): Plot Not Available. Please see docstring for list of available plots.')
     
    #plots without topic model
    if model is None:
        not_allowed_wm = ['tsne', 'topic_model', 'topic_distribution']
        if plot in not_allowed_wm:
            sys.exit('(Type Error): Model parameter Missing. Plot not supported without specific model passed in as Model param.')
            
    #handle topic_model plot error
    if plot == 'topic_model':
        not_allowed_tm = ['lsi', 'rp', 'nmf']
        if mod_type in not_allowed_tm:
            sys.exit('(Type Error): Model not supported for plot = topic_model. Please see docstring for list of available models supported for topic_model.')
        

    
    """
    error handling ends here
    """
    

    
    #import dependencies
    import pandas as pd
    import numpy
    
    #import cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    
    if plot == 'frequency':
        
        try:   
            
            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_words(corpus, n=None):
                vec = CountVectorizer()
                bag_of_words = vec.fit_transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]

            if topic_num is None:

                common_words = get_top_n_words(data_[target_], n=100)
                df2 = pd.DataFrame(common_words, columns = ['Text' , 'count'])

                df2.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title='Top 100 words after removing stop words')

            else: 

                title = str(topic_num) + ': ' + 'Top 100 words after removing stop words'

                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num]

                common_words = get_top_n_words(filtered_df[target_], n=100)
                df2 = pd.DataFrame(common_words, columns = ['Text' , 'count'])

                df2.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title=title)
                
        except:
            
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
                
    
    elif plot == 'distribution':
        
        try:
        
            if topic_num is None:

                b = data_[target_].apply(lambda x: len(str(x).split()))
                b = pd.DataFrame(b)
                b[target_].iplot(
                kind='hist',
                bins=100,
                xTitle='word count',
                linecolor='black',
                yTitle='count',
                title='Word Count Distribution')

            else:

                title = str(topic_num) + ': ' + 'Word Count Distribution'
                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num]

                b = filtered_df[target_].apply(lambda x: len(str(x).split()))
                b = pd.DataFrame(b)
                b[target_].iplot(
                kind='hist',
                bins=100,
                xTitle='word count',
                linecolor='black',
                yTitle='count',
                title= title)            

        except:
            
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
                
                
    elif plot == 'bigram':
        
        try:
            
            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_bigram(corpus, n=None):
                vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]

            if topic_num is None:

                common_words = get_top_n_bigram(data_[target_], 100)
                df3 = pd.DataFrame(common_words, columns = ['Text' , 'count'])
                df3.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title='Top 100 bigrams after removing stop words')

            else:

                title = str(topic_num) + ': ' + 'Top 100 bigrams after removing stop words'
                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num]

                common_words = get_top_n_bigram(filtered_df[target_], 100)
                df3 = pd.DataFrame(common_words, columns = ['Text' , 'count'])
                df3.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title=title)            
    
        except:
            
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
            
    elif plot == 'trigram':
        
        try:
            
            from sklearn.feature_extraction.text import CountVectorizer

            def get_top_n_trigram(corpus, n=None):
                vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
                bag_of_words = vec.transform(corpus)
                sum_words = bag_of_words.sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
                return words_freq[:n]

            if topic_num is None:

                common_words = get_top_n_trigram(data_[target_], 100)
                df3 = pd.DataFrame(common_words, columns = ['Text' , 'count'])
                df3.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title='Top 100 trigrams after removing stop words')

            else:

                title = str(topic_num) + ': ' + 'Top 100 trigrams after removing stop words'
                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num]            
                common_words = get_top_n_trigram(filtered_df[target_], 100)
                df3 = pd.DataFrame(common_words, columns = ['Text' , 'count'])
                df3.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
                kind='bar', yTitle='Count', linecolor='black', title=title)
                
        except:
            
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
            
   
    elif plot == 'sentiment':
        
        try:
            
            #loadies dependencies
            import plotly.graph_objects as go
            from textblob import TextBlob

            if topic_num is None:

                sentiments = data_[target_].map(lambda text: TextBlob(text).sentiment.polarity)
                sentiments = pd.DataFrame(sentiments)
                sentiments[target_].iplot(
                kind='hist',
                bins=50,
                xTitle='polarity',
                linecolor='black',
                yTitle='count',
                title='Sentiment Polarity Distribution')

            else: 
                title = str(topic_num) + ': ' + 'Sentiment Polarity Distribution'
                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num] 
                sentiments = filtered_df[target_].map(lambda text: TextBlob(text).sentiment.polarity)
                sentiments = pd.DataFrame(sentiments)
                sentiments[target_].iplot(
                kind='hist',
                bins=50,
                xTitle='polarity',
                linecolor='black',
                yTitle='count',
                title=title)            
         
        except:
            
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
            
            
    elif plot == 'pos':
        
        from textblob import TextBlob
        
        b = list(id2word.token2id.keys())
        blob = TextBlob(str(b))
        pos_df = pd.DataFrame(blob.tags, columns = ['word' , 'pos'])
        pos_df = pos_df.loc[pos_df['pos'] != 'POS']
        pos_df = pos_df.pos.value_counts()[:20]
        pos_df.iplot(
        kind='bar',
        xTitle='POS',
        yTitle='count', 
        title='Top 20 Part-of-speech tagging for review corpus')
        
        
    elif plot == 'tsne':
        
        b = assign_model(model, verbose = False)
        b.dropna(axis=0, inplace=True) #droping rows where Dominant_Topic is blank

        c = []
        for i in b.columns:
            if 'Topic_' in i:
                a = i
                c.append(a)

        bb = b[c]

        from sklearn.manifold import TSNE
        X_embedded = TSNE(n_components=3).fit_transform(bb)

        X = pd.DataFrame(X_embedded)
        X['Dominant_Topic'] = b['Dominant_Topic']
        X.sort_values(by='Dominant_Topic', inplace=True)
        X.dropna(inplace=True)

        import plotly.express as px
        df = X
        fig = px.scatter_3d(df, x=0, y=1, z=2,
                      color='Dominant_Topic', title='3d TSNE Plot for Topic Model', opacity=0.7, width=900, height=800)
        fig.show()
        
    
    elif plot == 'topic_model':
        
        import pyLDAvis
        import pyLDAvis.gensim  # don't skip this

        import warnings
        warnings.filterwarnings('ignore') 
        pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(model, corpus, id2word, mds='mmds')
        display(vis)

    elif plot == 'topic_distribution':
        
        try:
            
            iter1 = len(model.show_topics(999999))
        
        except:
            
            try:
                iter1 = model.num_topics
            
            except:
                
                iter1 = model.n_components_

        topic_name = []
        keywords = []

        for i in range(0,iter1):
            
            try:

                s = model.show_topic(i,topn=10)
                topic_name.append('Topic ' + str(i))

                kw = []

                for i in s:
                    kw.append(i[0])

                keywords.append(kw)
                
            except:
                
                keywords.append('NA')
                topic_name.append('Topic ' + str(i))

        keyword = []
        for i in keywords:
            b = ", ".join(i)
            keyword.append(b)


        kw_df = pd.DataFrame({'Topic': topic_name, 'Keyword' : keyword}).set_index('Topic')
        ass_df = assign_model(model, verbose = False)
        ass_df_pivot = ass_df.pivot_table(index='Dominant_Topic', values='Topic_0', aggfunc='count')
        df2 = ass_df_pivot.join(kw_df)
        df2 = df2.reset_index()
        df2.columns = ['Topic', 'Documents', 'Keyword']
        
        """
        sorting column starts
        
        """
        
        topic_list = list(df2['Topic'])

        s = []
        for i in range(0,len(topic_list)):
            a = int(topic_list[i].split()[1])
            s.append(a)

        df2['Topic'] = s
        df2.sort_values(by='Topic', inplace=True)
        df2.sort_values(by='Topic', inplace=True)
        topic_list = list(df2['Topic'])
        topic_list = list(df2['Topic'])
        s = []
        for i in topic_list:
            a = 'Topic ' + str(i)
            s.append(a)

        df2['Topic'] = s
        df2.reset_index(drop=True, inplace=True)
        
        """
        sorting column ends
        """

        import plotly.express as px
        fig = px.bar(df2, x='Topic', y='Documents', hover_data = ['Keyword'], title='Document Distribution by Topics')
        fig.show()
    
        
    elif plot == 'wordcloud':
        
        try:
            
            from wordcloud import WordCloud, STOPWORDS 
            import matplotlib.pyplot as plt 

            stopwords = set(STOPWORDS) 

            if topic_num is None:

                atext = " ".join(review for review in data_[target_])

            else:

                assigned_df = assign_model(model, verbose = False)
                filtered_df = assigned_df.loc[assigned_df['Dominant_Topic'] == topic_num] 
                atext = " ".join(review for review in filtered_df[target_])

            wordcloud = WordCloud(width = 800, height = 800, 
                            background_color ='white', 
                            stopwords = stopwords, 
                            min_font_size = 10).generate(atext) 

            # plot the WordCloud image                        
            plt.figure(figsize = (8, 8), facecolor = None) 
            plt.imshow(wordcloud) 
            plt.axis("off") 
            plt.tight_layout(pad = 0) 

            plt.show() 
            
        except:
            sys.exit('(Value Error): Invalid topic_num param or empty Vocab. Try changing Topic Number.')
        
    elif plot == 'umap':
        
        #warnings
        from matplotlib.axes._axes import _log as matplotlib_axes_logger
        matplotlib_axes_logger.setLevel('ERROR')
        
        #loading dependencies
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from yellowbrick.text import UMAPVisualizer
        import matplotlib.pyplot as plt
            
        tfidf = TfidfVectorizer()
        docs = tfidf.fit_transform(data_[target_])
                
        # Instantiate the clustering model
        clusters = KMeans(n_clusters=5, random_state=seed)
        clusters.fit(docs)
        
        plt.figure(figsize=(10,6))
        
        umap = UMAPVisualizer(random_state=seed)
        umap.fit(docs, ["c{}".format(c) for c in clusters.labels_])
        umap.show()



def tune_model(model=None,
               multi_core=False,
               supervised_target=None,
               estimator=None,
               optimize=None,
               auto_fe = True,
               fold=10):


    """

    Description:
    ------------
    This function tunes the num_topics model parameter using a predefined grid with
    the objective of optimizing a supervised learning metric as defined in the optimize
    param. You can choose the supervised estimator from a large library available in 
    pycaret. By default, supervised estimator is Linear. 

    This function returns the tuned model object.

        Example
        -------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')        
        
        tuned_lda = tune_model(model = 'lda', supervised_target = 'status') 

        This will return trained Latent Dirichlet Allocation model. 

    Parameters
    ----------
    model : string, default = None

    Enter abbreviated name of the model. List of available models supported: 

    Model                              Abbreviated String   Original Implementation 
    ---------                          ------------------   -----------------------
    Latent Dirichlet Allocation        'lda'                gensim/models/ldamodel.html
    Latent Semantic Indexing           'lsi'                gensim/models/lsimodel.html
    Hierarchical Dirichlet Process     'hdp'                gensim/models/hdpmodel.html
    Random Projections                 'rp'                 gensim/models/rpmodel.html
    Non-Negative Matrix Factorization  'nmf'                sklearn.decomposition.NMF.html

    multi_core: Boolean, default = False
    True would utilize all CPU cores to parallelize and speed up model training. Only
    available for 'lda'. For all other models, multi_core parameter is ignored.

    supervised_target: string
    Name of the target column for supervised learning. If None, the mdel coherence value
    is used as the objective function.

    estimator: string, default = None

    Estimator                     Abbreviated String     Task 
    ---------                     ------------------     ---------------
    Logistic Regression           'lr'                   Classification
    K Nearest Neighbour           'knn'                  Classification
    Naives Bayes                  'nb'                   Classification
    Decision Tree                 'dt'                   Classification
    SVM (Linear)                  'svm'                  Classification
    SVM (RBF)                     'rbfsvm'               Classification
    Gaussian Process              'gpc'                  Classification
    Multi Level Perceptron        'mlp'                  Classification
    Ridge Classifier              'ridge'                Classification
    Random Forest                 'rf'                   Classification
    Quadratic Disc. Analysis      'qda'                  Classification
    AdaBoost                      'ada'                  Classification
    Gradient Boosting             'gbc'                  Classification
    Linear Disc. Analysis         'lda'                  Classification
    Extra Trees Classifier        'et'                   Classification
    Extreme Gradient Boosting     'xgboost'              Classification
    Light Gradient Boosting       'lightgbm'             Classification
    CatBoost Classifier           'catboost'             Classification
    Linear Regression             'lr'                   Regression
    Lasso Regression              'lasso'                Regression
    Ridge Regression              'ridge'                Regression
    Elastic Net                   'en'                   Regression
    Least Angle Regression        'lar'                  Regression
    Lasso Least Angle Regression  'llar'                 Regression
    Orthogonal Matching Pursuit   'omp'                  Regression
    Bayesian Ridge                'br'                   Regression
    Automatic Relevance Determ.   'ard'                  Regression
    Passive Aggressive Regressor  'par'                  Regression
    Random Sample Consensus       'ransac'               Regression
    TheilSen Regressor            'tr'                   Regression
    Huber Regressor               'huber'                Regression
    Kernel Ridge                  'kr'                   Regression
    Support Vector Machine        'svm'                  Regression
    K Neighbors Regressor         'knn'                  Regression
    Decision Tree                 'dt'                   Regression
    Random Forest                 'rf'                   Regression
    Extra Trees Regressor         'et'                   Regression
    AdaBoost Regressor            'ada'                  Regression
    Gradient Boosting             'gbr'                  Regression
    Multi Level Perceptron        'mlp'                  Regression
    Extreme Gradient Boosting     'xgboost'              Regression
    Light Gradient Boosting       'lightgbm'             Regression
    CatBoost Regressor            'catboost'             Regression

    If set to None, Linear model is used by default for both classification
    and regression tasks.
    
    optimize: string, default = None
    
    For Classification tasks:
    Accuracy, AUC, Recall, Precision, F1, Kappa
    
    For Regression tasks:
    MAE, MSE, RMSE, R2, ME
    
    If set to None, default is 'Accuracy' for classification and 'R2' for 
    regression tasks.

    auto_fe: boolean, default = True
    Automatic text feature engineering. Only used when supervised_target is
    passed. When set to true, it will generate text based features such as 
    polarity, subjectivity, wordcounts to be used in supervised learning.
    Ignored when supervised_target is set to None.

    fold: integer, default = 10
    Number of folds to be used in Kfold CV. Must be at least 2. 

    Returns:
    --------

    visual plot:  Visual plot with k number of topics on x-axis with metric to
    -----------   optimize on y-axis. Coherence is used when learning is 
                  unsupervised. Also, prints the best model metric.

    model:        trained model object with best K number of topics.
    -----------

    Warnings:
    ---------
    - Random Projections ('rp') and Non Negative Matrix Factorization ('nmf')
      is not available for unsupervised learning. Error is raised when 'rp' or
      'nmf' is passed without supervised_target.

    - Estimators using kernel based methods such as Kernel Ridge Regressor, 
      Automatic Relevance Determinant, Gaussian Process Classifier, Radial Basis
      Support Vector Machine and Multi Level Perceptron may have longer training 
      times.
     
    
    """



    """
    exception handling starts here
    """

    #ignore warnings
    import warnings
    warnings.filterwarnings('ignore') 

    import sys

    #checking for model parameter
    if model is None:
        sys.exit('(Value Error): Model parameter Missing. Please see docstring for list of available models.')

    #checking for allowed models
    allowed_models = ['lda', 'lsi', 'hdp', 'rp', 'nmf']

    if model not in allowed_models:
        sys.exit('(Value Error): Model Not Available. Please see docstring for list of available models.')

    #checking multicore type:
    if type(multi_core) is not bool:
        sys.exit('(Type Error): multi_core parameter can only take argument as True or False.')

    #check supervised target:
    if supervised_target is not None:
        all_col = list(data_.columns)
        target = target_
        all_col.remove(target)
        if supervised_target not in all_col:
            sys.exit('(Value Error): supervised_target not recognized. It can only be one of the following: ' + str(all_col))

    #supervised target exception handling
    if supervised_target is None:
        models_not_allowed = ['rp', 'nmf']

        if model in models_not_allowed:
            sys.exit('(Type Error): Model not supported for unsupervised tuning. Either supervised_target param has to be passed or different model has to be used. Please see docstring for available models.')



    #checking estimator:
    if estimator is not None:

        available_estimators = ['lr', 'knn', 'nb', 'dt', 'svm', 'rbfsvm', 'gpc', 'mlp', 'ridge', 'rf', 'qda', 'ada', 
                            'gbc', 'lda', 'et', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm', 'catboost']

        if estimator not in available_estimators:
            sys.exit('(Value Error): Estimator Not Available. Please see docstring for list of available estimators.')


    #checking optimize parameter
    if optimize is not None:

        available_optimizers = ['MAE', 'MSE', 'RMSE', 'R2', 'ME', 'Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa']

        if optimize not in available_optimizers:
            sys.exit('(Value Error): optimize parameter Not Available. Please see docstring for list of available parameters.')

    #checking auto_fe:
    if type(auto_fe) is not bool:
        sys.exit('(Type Error): auto_fe parameter can only take argument as True or False.')


    #checking fold parameter
    if type(fold) is not int:
        sys.exit('(Type Error): Fold parameter only accepts integer value.')


    """
    exception handling ends here
    """

    #pre-load libraries
    import pandas as pd
    import ipywidgets as ipw
    from ipywidgets import Output
    from IPython.display import display, HTML, clear_output, update_display
    import datetime, time

    #progress bar
    max_steps = 25
    progress = ipw.IntProgress(value=0, min=0, max=max_steps, step=1 , description='Processing: ')
    display(progress)
    
    timestampStr = datetime.datetime.now().strftime("%H:%M:%S")

    monitor = pd.DataFrame( [ ['Initiated' , '. . . . . . . . . . . . . . . . . .', timestampStr ], 
                             ['Status' , '. . . . . . . . . . . . . . . . . .' , 'Loading Dependencies'],
                             ['Step' , '. . . . . . . . . . . . . . . . . .',  'Initializing' ] ],
                              columns=['', ' ', '   ']).set_index('')
    
    monitor_out = Output()
    display(monitor_out)
    
    with monitor_out:
        display(monitor, display_id = 'monitor')

    #General Dependencies
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn import metrics
    import numpy as np
    import plotly.express as px

    #setting up cufflinks
    import cufflinks as cf
    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    progress.value += 1 

    #define the problem
    if supervised_target is None:
        problem ='unsupervised'
    elif data_[supervised_target].value_counts().count() == 2: 
        problem = 'classification'
    else:
        problem = 'regression'

    #define topic_model_name
    if model == 'lda':
        topic_model_name = 'Latent Dirichlet Allocation'
    elif model == 'lsi':
        topic_model_name = 'Latent Semantic Indexing'
    elif model == 'hdp':
        topic_model_name = 'Hierarchical Dirichlet Process'
    elif model == 'nmf':
        topic_model_name = 'Non-Negative Matrix Factorization'
    elif model == 'rp':
        topic_model_name = 'Random Projections'

    #defining estimator:
    if problem == 'classification' and estimator is None:
        estimator = 'lr'
    elif problem == 'regression' and estimator is None:
        estimator = 'lr'        
    else:
        estimator = estimator

    #defining optimizer:
    if optimize is None and problem == 'classification':
        optimize = 'Accuracy'
    elif optimize is None and problem == 'regression':
        optimize = 'R2'
    else:
        optimize=optimize

    progress.value += 1 

    #creating sentiments

    if problem == 'classification' or problem == 'regression':

        if auto_fe:

            monitor.iloc[1,1:] = 'Feature Engineering'
            update_display(monitor, display_id = 'monitor')

            from textblob import TextBlob

            monitor.iloc[2,1:] = 'Extracting Polarity'
            update_display(monitor, display_id = 'monitor')

            polarity = data_[target_].map(lambda text: TextBlob(text).sentiment.polarity)

            monitor.iloc[2,1:] = 'Extracting Subjectivity'
            update_display(monitor, display_id = 'monitor')

            subjectivity = data_[target_].map(lambda text: TextBlob(text).sentiment.subjectivity)

            monitor.iloc[2,1:] = 'Extracting Wordcount'
            update_display(monitor, display_id = 'monitor')

            word_count = [len(i) for i in text]

            progress.value += 1 

    #defining tuning grid
    param_grid = [2,4,8,16,32,64,100,200,300,400] 

    master = []; master_df = []

    monitor.iloc[1,1:] = 'Creating Topic Model'
    update_display(monitor, display_id = 'monitor')

    for i in param_grid:
        progress.value += 1                      
        monitor.iloc[2,1:] = 'Fitting Model With ' + str(i) + ' Topics'
        update_display(monitor, display_id = 'monitor')

        #create and assign the model to dataset d
        m = create_model(model=model, multi_core=multi_core, num_topics=i, verbose=False)
        d = assign_model(m, verbose=False)

        if problem in ['classification', 'regression'] and auto_fe:
            d['Polarity'] = polarity
            d['Subjectivity'] = subjectivity
            d['word_count'] = word_count

        master.append(m)
        master_df.append(d)

        #topic model creation end's here

    if problem == 'unsupervised':

        monitor.iloc[1,1:] = 'Evaluating Topic Model'
        update_display(monitor, display_id = 'monitor')

        from gensim.models import CoherenceModel

        coherence = []
        metric = []

        counter = 0

        for i in master:
            progress.value += 1 
            monitor.iloc[2,1:] = 'Evaluating Coherence With ' + str(param_grid[counter]) + ' Topics'
            update_display(monitor, display_id = 'monitor')

            model = CoherenceModel(model=i, texts=text, dictionary=id2word, coherence='c_v')
            model_coherence = model.get_coherence()
            coherence.append(model_coherence)
            metric.append('Coherence')
            counter += 1

        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        update_display(monitor, display_id = 'monitor')

        df = pd.DataFrame({'# Topics': param_grid, 'Score' : coherence, 'Metric': metric})
        df.columns = ['# Topics', 'Score', 'Metric']

        sorted_df = df.sort_values(by='Score', ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]

        fig = px.line(df, x='# Topics', y='Score', line_shape='linear', 
                      title= 'Coherence Value and # of Topics', color='Metric')

        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        
        fig.show()
        
        monitor = '' 
        update_display(monitor, display_id = 'monitor')
        
        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)['# Topics'])[0]
        best_m = round(np.array(sorted_df.head(1)['Score'])[0],4)
        p = 'Best Model: ' + topic_model_name + ' |' + ' # Topics: ' + str(best_k) + ' | ' + 'Coherence: ' + str(best_m)
        print(p)


    elif problem == 'classification':

        """

        defining estimator

        """

        monitor.iloc[1,1:] = 'Evaluating Topic Model'
        update_display(monitor, display_id = 'monitor')

        if estimator == 'lr':

            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=seed)
            full_name = 'Logistic Regression'

        elif estimator == 'knn':

            from sklearn.neighbors import KNeighborsClassifier
            model = KNeighborsClassifier()
            full_name = 'K Nearest Neighbours'

        elif estimator == 'nb':

            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
            full_name = 'Naive Bayes'

        elif estimator == 'dt':

            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier(random_state=seed)
            full_name = 'Decision Tree'

        elif estimator == 'svm':

            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
            full_name = 'Support Vector Machine'

        elif estimator == 'rbfsvm':

            from sklearn.svm import SVC
            model = SVC(gamma='auto', C=1, probability=True, kernel='rbf', random_state=seed)
            full_name = 'RBF SVM'

        elif estimator == 'gpc':

            from sklearn.gaussian_process import GaussianProcessClassifier
            model = GaussianProcessClassifier(random_state=seed)
            full_name = 'Gaussian Process Classifier'

        elif estimator == 'mlp':

            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(max_iter=500, random_state=seed)
            full_name = 'Multi Level Perceptron'    

        elif estimator == 'ridge':

            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier(random_state=seed)
            full_name = 'Ridge Classifier'        

        elif estimator == 'rf':

            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=seed)
            full_name = 'Random Forest Classifier'    

        elif estimator == 'qda':

            from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
            model = QuadraticDiscriminantAnalysis()
            full_name = 'Quadratic Discriminant Analysis' 

        elif estimator == 'ada':

            from sklearn.ensemble import AdaBoostClassifier
            model = AdaBoostClassifier(random_state=seed)
            full_name = 'AdaBoost Classifier'        

        elif estimator == 'gbc':

            from sklearn.ensemble import GradientBoostingClassifier    
            model = GradientBoostingClassifier(random_state=seed)
            full_name = 'Gradient Boosting Classifier'    

        elif estimator == 'lda':

            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
            full_name = 'Linear Discriminant Analysis'

        elif estimator == 'et':

            from sklearn.ensemble import ExtraTreesClassifier 
            model = ExtraTreesClassifier(random_state=seed)
            full_name = 'Extra Trees Classifier'

        elif estimator == 'xgboost':

            from xgboost import XGBClassifier
            model = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = 'Extreme Gradient Boosting'

        elif estimator == 'lightgbm':

            import lightgbm as lgb
            model = lgb.LGBMClassifier(random_state=seed)
            full_name = 'Light Gradient Boosting Machine'

        elif estimator == 'catboost':
            from catboost import CatBoostClassifier
            model = CatBoostClassifier(random_state=seed, silent=True) # Silent is True to suppress CatBoost iteration results 
            full_name = 'CatBoost Classifier'

        progress.value += 1 

        """
        start model building here

        """

        acc = [];  auc = []; recall = []; prec = []; kappa = []; f1 = []

        for i in range(0,len(master_df)):
            progress.value += 1 
            param_grid_val = param_grid[i]

            monitor.iloc[2,1:] = 'Evaluating Classifier With ' + str(param_grid_val) + ' Topics'
            update_display(monitor, display_id = 'monitor')                

            #prepare the dataset for supervised problem
            d = master_df[i]
            d.dropna(axis=0, inplace=True) #droping rows where Dominant_Topic is blank
            d.drop([target_], inplace=True, axis=1)
            d = pd.get_dummies(d)

            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            model.fit(X,y)

            #generate the prediction and evaluate metric
            pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

            acc_ = metrics.accuracy_score(y,pred)
            acc.append(acc_)

            recall_ = metrics.recall_score(y,pred)
            recall.append(recall_)

            precision_ = metrics.precision_score(y,pred)
            prec.append(precision_)

            kappa_ = metrics.cohen_kappa_score(y,pred)
            kappa.append(kappa_)

            f1_ = metrics.f1_score(y,pred)
            f1.append(f1_)

            if hasattr(model,'predict_proba'):
                pred_ = cross_val_predict(model,X,y,cv=fold, method = 'predict_proba')
                pred_prob = pred_[:,1]
                auc_ = metrics.roc_auc_score(y,pred_prob)
                auc.append(auc_)

            else:
                auc.append(0)


        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        update_display(monitor, display_id = 'monitor')

        df = pd.DataFrame({'# Topics': param_grid, 'Accuracy' : acc, 'AUC' : auc, 'Recall' : recall, 
                   'Precision' : prec, 'F1' : f1, 'Kappa' : kappa})

        sorted_df = df.sort_values(by=optimize, ascending=False)
        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]
        progress.value += 1 
        sd = pd.melt(df, id_vars=['# Topics'], value_vars=['Accuracy', 'AUC', 'Recall', 'Precision', 'F1', 'Kappa'], 
                     var_name='Metric', value_name='Score')

        fig = px.line(sd, x='# Topics', y='Score', color='Metric', line_shape='linear', range_y = [0,1])
        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        title= str(full_name) + ' Metrics and # of Topics'
        fig.update_layout(title={'text': title, 'y':0.95,'x':0.45,'xanchor': 'center','yanchor': 'top'})

        fig.show()
        
        monitor = ''
        update_display(monitor, display_id = 'monitor')
        
        monitor_out.clear_output()
        progress.close()

        best_k = np.array(sorted_df.head(1)['# Topics'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + topic_model_name + ' |' + ' # Topics: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)

    elif problem == 'regression':

        """

        defining estimator

        """

        monitor.iloc[1,1:] = 'Evaluating Topic Model'
        update_display(monitor, display_id = 'monitor')

        if estimator == 'lr':

            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            full_name = 'Linear Regression'

        elif estimator == 'lasso':

            from sklearn.linear_model import Lasso
            model = Lasso(random_state=seed)
            full_name = 'Lasso Regression'

        elif estimator == 'ridge':

            from sklearn.linear_model import Ridge
            model = Ridge(random_state=seed)
            full_name = 'Ridge Regression'

        elif estimator == 'en':

            from sklearn.linear_model import ElasticNet
            model = ElasticNet(random_state=seed)
            full_name = 'Elastic Net'

        elif estimator == 'lar':

            from sklearn.linear_model import Lars
            model = Lars()
            full_name = 'Least Angle Regression'

        elif estimator == 'llar':

            from sklearn.linear_model import LassoLars
            model = LassoLars()
            full_name = 'Lasso Least Angle Regression'

        elif estimator == 'omp':

            from sklearn.linear_model import OrthogonalMatchingPursuit
            model = OrthogonalMatchingPursuit()
            full_name = 'Orthogonal Matching Pursuit'

        elif estimator == 'br':
            from sklearn.linear_model import BayesianRidge
            model = BayesianRidge()
            full_name = 'Bayesian Ridge Regression' 

        elif estimator == 'ard':

            from sklearn.linear_model import ARDRegression
            model = ARDRegression()
            full_name = 'Automatic Relevance Determination'        

        elif estimator == 'par':

            from sklearn.linear_model import PassiveAggressiveRegressor
            model = PassiveAggressiveRegressor(random_state=seed)
            full_name = 'Passive Aggressive Regressor'    

        elif estimator == 'ransac':

            from sklearn.linear_model import RANSACRegressor
            model = RANSACRegressor(random_state=seed)
            full_name = 'Random Sample Consensus'   

        elif estimator == 'tr':

            from sklearn.linear_model import TheilSenRegressor
            model = TheilSenRegressor(random_state=seed)
            full_name = 'TheilSen Regressor'     

        elif estimator == 'huber':

            from sklearn.linear_model import HuberRegressor
            model = HuberRegressor()
            full_name = 'Huber Regressor'   

        elif estimator == 'kr':

            from sklearn.kernel_ridge import KernelRidge
            model = KernelRidge()
            full_name = 'Kernel Ridge'

        elif estimator == 'svm':

            from sklearn.svm import SVR
            model = SVR()
            full_name = 'Support Vector Regression'  

        elif estimator == 'knn':

            from sklearn.neighbors import KNeighborsRegressor
            model = KNeighborsRegressor()
            full_name = 'Nearest Neighbors Regression' 

        elif estimator == 'dt':

            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor(random_state=seed)
            full_name = 'Decision Tree Regressor'

        elif estimator == 'rf':

            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(random_state=seed)
            full_name = 'Random Forest Regressor'

        elif estimator == 'et':

            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(random_state=seed)
            full_name = 'Extra Trees Regressor'    

        elif estimator == 'ada':

            from sklearn.ensemble import AdaBoostRegressor
            model = AdaBoostRegressor(random_state=seed)
            full_name = 'AdaBoost Regressor'   

        elif estimator == 'gbr':

            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(random_state=seed)
            full_name = 'Gradient Boosting Regressor'       

        elif estimator == 'mlp':

            from sklearn.neural_network import MLPRegressor
            model = MLPRegressor(random_state=seed)
            full_name = 'MLP Regressor'

        elif estimator == 'xgboost':

            from xgboost import XGBRegressor
            model = XGBRegressor(random_state=seed, n_jobs=-1, verbosity=0)
            full_name = 'Extreme Gradient Boosting Regressor'

        elif estimator == 'lightgbm':

            import lightgbm as lgb
            model = lgb.LGBMRegressor(random_state=seed)
            full_name = 'Light Gradient Boosting Machine'
            
        elif estimator == 'catboost':
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(random_state=seed, silent = True)
            full_name = 'CatBoost Regressor'

        progress.value += 1 

        """
        start model building here

        """

        score = []
        metric = []

        for i in range(0,len(master_df)):
            progress.value += 1 
            param_grid_val = param_grid[i]

            monitor.iloc[2,1:] = 'Evaluating Regressor With ' + str(param_grid_val) + ' Topics'
            update_display(monitor, display_id = 'monitor')    

            #prepare the dataset for supervised problem
            d = master_df[i]
            d.dropna(axis=0, inplace=True) #droping rows where Dominant_Topic is blank
            d.drop([target_], inplace=True, axis=1)
            d = pd.get_dummies(d)

            #split the dataset
            X = d.drop(supervised_target, axis=1)
            y = d[supervised_target]

            #fit the model
            model.fit(X,y)

            #generate the prediction and evaluate metric
            pred = cross_val_predict(model,X,y,cv=fold, method = 'predict')

            if optimize == 'R2':
                r2_ = metrics.r2_score(y,pred)
                score.append(r2_)

            elif optimize == 'MAE':          
                mae_ = metrics.mean_absolute_error(y,pred)
                score.append(mae_)

            elif optimize == 'MSE':
                mse_ = metrics.mean_squared_error(y,pred)
                score.append(mse_)

            elif optimize == 'RMSE':
                mse_ = metrics.mean_squared_error(y,pred)        
                rmse_ = np.sqrt(mse_)
                score.append(rmse_)

            elif optimize == 'ME':
                max_error_ = metrics.max_error(y,pred)
                score.append(max_error_)

            metric.append(str(optimize))

        monitor.iloc[1,1:] = 'Compiling Results'
        monitor.iloc[1,1:] = 'Finalizing'
        update_display(monitor, display_id = 'monitor')                    

        df = pd.DataFrame({'# Topics': param_grid, 'Score' : score, 'Metric': metric})
        df.columns = ['# Topics', optimize, 'Metric']

        #sorting to return best model
        if optimize == 'R2':
            sorted_df = df.sort_values(by=optimize, ascending=False)
        else: 
            sorted_df = df.sort_values(by=optimize, ascending=True)

        ival = sorted_df.index[0]

        best_model = master[ival]
        best_model_df = master_df[ival]

        fig = px.line(df, x='# Topics', y=optimize, line_shape='linear', 
                      title= str(full_name) + ' Metrics and # of Topics', color='Metric')

        fig.update_layout(plot_bgcolor='rgb(245,245,245)')
        progress.value += 1
        
        monitor = ''
        update_display(monitor, display_id = 'monitor')
        
        monitor_out.clear_output()
        progress.close()

        fig.show()
        best_k = np.array(sorted_df.head(1)['# Topics'])[0]
        best_m = round(np.array(sorted_df.head(1)[optimize])[0],4)
        p = 'Best Model: ' + topic_model_name + ' |' + ' # Topics: ' + str(best_k) + ' | ' + str(optimize) + ' : ' + str(best_m)
        print(p)

    #storing into experiment
    tup = ('Best Model',best_model)
    experiment__.append(tup)    

    return best_model




def evaluate_model(model):
    
    """
          
    Description:
    ------------
    This function displays the user interface for all the available plots 
    for a given model. It internally uses the plot_model() function. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')     
        lda = create_model('lda')
        
        evaluate_model(lda)
        
        This will display the User Interface for all of the plots for 
        given model. 

    Parameters
    ----------
    model : object, default = none
    A trained model object should be passed. 

    Returns:
    --------

    User Interface:  Displays the user interface for plotting.
    --------------

    Warnings:
    ---------
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
    for i in range(0,n_topic_assigned):
        final_list.append('Topic ' +str(i))

    a = widgets.ToggleButtons(
                            options=[('Frequency Plot', 'frequency'),
                                     ('Bigrams', 'bigram'), 
                                     ('Trigrams', 'trigram'), 
                                     ('Sentiment Polarity', 'sentiment'),
                                     ('Word Cloud', 'wordcloud'),
                                    ],
        
                            description='Plot Type:',

                            disabled=False,

                            button_style='', # 'success', 'info', 'warning', 'danger' or ''
        
                            icons=['']
    )
    
    b = widgets.Dropdown(options=final_list, description='Topic #:', disabled=False)
    
    d = interact_manual(plot_model, model = fixed(model), plot = a, topic_num=b)



def save_model(model, model_name):
    
    """
          
    Description:
    ------------
    This function saves the trained model object into the current active 
    directory as a pickle file for later use. 
    
        Example:
        --------
        from pycaret.datasets import get_data
        kiva = get_data('kiva')
        experiment_name = setup(data = kiva, target = 'en')
        lda = create_model('lda')
        
        save_model(lda, 'lda_model_23122019')
        
        This will save the model as a binary pickle file in the current 
        directory. 

    Parameters
    ----------
    model : object, default = none
    A trained model object should be passed.
    
    model_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None    
       
         
    """
    
    import joblib
    model_name = model_name + '.pkl'
    joblib.dump(model, model_name)
    print('Model Succesfully Saved')



def load_model(model_name):
    
    """
          
    Description:
    ------------
    This function loads a previously saved model from the current active directory 
    into the current python environment. Load object must be a pickle file.
    
        Example:
        --------
        saved_lda = load_model('lda_model_23122019')
        
        This will call the trained model in saved_lr variable using model_name param.
        The file must be in current directory.

    Parameters
    ----------
    model_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None    
       
         
    """
        
        
    import joblib
    model_name = model_name + '.pkl'
    print('Model Sucessfully Loaded')
    return joblib.load(model_name)



def save_experiment(experiment_name=None):
    
        
    """
          
    Description:
    ------------
    This function saves the entire experiment into the current active directory. 
    All outputs using pycaret are internally saved into a binary list which is
    pickilized when save_experiment() is used. 
    
        Example:
        --------
        save_experiment()
        
        This will save the entire experiment into the current active directory. 
        By default, the name of the experiment will use the session_id generated 
        during setup(). To use a custom name, a string must be passed to the 
        experiment_name param. For example:
        
        save_experiment('experiment_23122019')

    Parameters
    ----------
    experiment_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Success Message

    Warnings:
    ---------
    None
      
          
    """
    
    #general dependencies
    import joblib
    global experiment__
    
    #defining experiment name
    if experiment_name is None:
        experiment_name = 'experiment_' + str(seed)
        
    else:
        experiment_name = experiment_name  
        
    experiment_name = experiment_name + '.pkl'
    joblib.dump(experiment__, experiment_name)
    
    print('Experiment Succesfully Saved')



def load_experiment(experiment_name):
    
    """
          
    Description:
    ------------
    This function loads a previously saved experiment from the current active 
    directory into current python environment. Load object must be a pickle file.
    
        Example:
        --------
        saved_experiment = load_experiment('experiment_23122019')
        
        This will load the entire experiment pipeline into the object 
        saved_experiment. The experiment file must be in current directory.
        
    Parameters
    ----------
    experiment_name : string, default = none
    Name of pickle file to be passed as a string.

    Returns:
    --------    
    Information Grid containing details of saved objects in experiment pipeline.
    
    Warnings:
    ---------
    None  
      
          
    """
    
    #general dependencies
    import joblib
    import pandas as pd
    
    experiment_name = experiment_name + '.pkl'
    temp = joblib.load(experiment_name)
    
    name = []
    exp = []

    for i in temp:
        name.append(i[0])
        exp.append(i[-1])

    ind = pd.DataFrame(name, columns=['Object'])
    display(ind)

    return exp



def get_topics(data, text, model=None, num_topics=4):
    
    """
    Magic function to get topic model in Power Query / Power BI.
    """
    
    if model is None:
        model = 'lda'
        
    s = setup(data=data, target=text)
    c = create_model(model=model, num_topics=num_topics, verbose=False)
    dataset = assign_model(c, verbose=False)
    return dataset


