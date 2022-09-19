import warnings

from pycaret.utils._dependencies import _check_soft_dependencies

warnings.warn(
    "PyCaret ARules module is deprecated and not representative "
    "of the current state of the library. "
    "It may be removed and/or reworked in a future update.",
    DeprecationWarning,
)


def setup(data, transaction_id, item_id, ignore_items=None, session_id=None):

    """
    This function initializes the environment in pycaret. setup() must called before
    executing any other function in pycaret. It takes three mandatory parameters:
    (i) data, (ii) transaction_id param identifying basket and (iii) item_id param
    used to create rules. These three params are normally found in any transactional
    dataset. pycaret will internally convert the pandas.DataFrame into a sparse matrix
    which is required for association rules mining.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> data = get_data('france')
    >>> from pycaret.arules import *
    >>> exp = setup(data = data, transaction_id = 'InvoiceNo', item_id = 'Description')


    data: pandas.DataFrame
        Shape (n_samples, n_features) where n_samples is the number of samples and
        n_features is the number of features.


    transaction_id: str
        Name of column representing transaction id. This will be used to pivot the matrix.


    item_id: str
        Name of column used for creation of rules. Normally, this will be the variable of
        interest.


    ignore_items: list, default = None
        List of strings to be ignored when considering rule mining.


    session_id: int, default = None
        If None, a random seed is generated and returned in the Information grid. The
        unique number is then distributed as a seed in all functions used during the
        experiment. This can be used for later reproducibility of the entire experiment.


    Returns:
        Global variables.

    """

    # exception checking
    import sys

    # checking data type
    if hasattr(data, "shape") is False:
        sys.exit("(Type Error): data passed must be of type pandas.DataFrame")

    # ignore warnings
    import warnings

    warnings.filterwarnings("ignore")

    # load dependencies
    import random

    import numpy as np
    import pandas as pd
    from IPython.display import HTML, clear_output, display, update_display

    global X, txid, iid, ignore_list, seed, experiment__

    # create an empty list for pickling later.
    experiment__ = []

    # storing items in variable
    X = data
    txid = transaction_id
    iid = item_id
    ignore_list = ignore_items

    # generate seed to be used globally
    if session_id is None:
        seed = random.randint(150, 9000)
    else:
        seed = session_id

    # display info grid

    # transactions

    tx_unique = len(data[transaction_id].unique())
    item_unique = len(data[item_id].unique())
    if ignore_items is None:
        ignore_flag = "None"
    else:
        ignore_flag = ignore_items

    functions = pd.DataFrame(
        [
            ["session_id", seed],
            ["# Transactions", tx_unique],
            ["# Items", item_unique],
            ["Ignore Items", ignore_flag],
        ],
        columns=["Description", "Value"],
    )

    functions_ = functions.style.hide_index()
    display(functions_)

    return X, txid, iid, ignore_list, seed, experiment__


def create_model(
    metric="confidence",
    threshold=0.5,
    min_support=0.05,
    round=4,
    low_memory=False,
    max_len=None,
):

    """
    This function creates an association rules model using data and identifiers
    passed at setup stage. This function internally transforms the data for
    association rule mining.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> data = get_data('france')
    >>> from pycaret.arules import *
    >>> exp_name = setup(data = data, transaction_id = 'InvoiceNo', item_id = 'Description')
    >>> model1 = create_model(metric = 'confidence')


    metric: str, default = 'confidence'
        Metric to evaluate if a rule is of interest. Default is set to confidence.
        Other available metrics include 'support', 'lift', 'leverage', 'conviction'.
        These metrics are computed as follows:

        * support(A->C) = support(A+C) [aka 'support'], range: [0, 1]
        * confidence(A->C) = support(A+C) / support(A), range: [0, 1]
        * lift(A->C) = confidence(A->C) / support(C), range: [0, inf]
        * leverage(A->C) = support(A->C) - support(A)*support(C), range: [-1, 1]
        * conviction = [1 - support(C)] / [1 - confidence(A->C)], range: [0, inf]


    threshold: float, default = 0.5
        Minimal threshold for the evaluation metric, via the `metric` parameter,
        to decide whether a candidate rule is of interest.


    min_support: float, default = 0.05
        A float between 0 and 1 for minumum support of the itemsets returned.
        The support is computed as the fraction `transactions_where_item(s)_occur /
        total_transactions`.


    round: int, default = 4
        Number of decimal places metrics in score grid will be rounded to.

    low_memory: bool, default = False
        If `True`, uses an iterator for apriori to search for combinations above
      `min_support`.
      Note that while `low_memory=True` should only be used for large dataset
      if memory resources are limited, because this implementation is approx.
      3-6x slower than the default.

    max_len: int, default = None
        Maximum length of the itemsets generated in apriori. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.


    Returns:
        pandas.DataFrame


    Warnings
    --------
    - Setting low values for min_support may increase training time.

    """

    # loading dependencies
    import pandas as pd
    from IPython.display import HTML, clear_output, display, update_display

    _check_soft_dependencies("mlxtend", extra="models", severity="error")
    from mlxtend.frequent_patterns import apriori, association_rules

    # reshaping the dataframe
    basket = (
        X.groupby([txid, iid])[iid]
        .count()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index(txid)
    )
    if ignore_list is not None:
        basket = basket.drop(ignore_list, axis=1)

    def encode_units(x):

        if x <= 0:
            return 0
        if x >= 1:
            return 1

    basket = basket.applymap(encode_units)

    frequent_itemsets = apriori(
        basket,
        min_support=min_support,
        use_colnames=True,
        low_memory=low_memory,
        max_len=max_len,
    )
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=threshold)
    rules = rules.sort_values(by=[metric], ascending=False).reset_index(drop=True)
    rules = rules.round(round)

    # storing into experiment
    tup = ("Basket", basket)
    experiment__.append(tup)

    tup = ("Frequent Itemsets", frequent_itemsets)
    experiment__.append(tup)

    tup = ("Rules", rules)
    experiment__.append(tup)

    return rules


def plot_model(model, plot="2d", scale=1, display_format=None):

    """
    This function takes a model dataframe returned by create_model() function.
    '2d' and '3d' plots are available.


    Example
    -------
    >>> from pycaret.datasets import get_data
    >>> data = get_data('france')
    >>> from pycaret.arules import *
    >>> exp_name = setup(data = data, transaction_id = 'InvoiceNo', item_id = 'Description')
    >>> rule1 = create_model(metric='confidence', threshold=0.7, min_support=0.05)
    >>> plot_model(rule1, plot='2d')


    model: pandas.DataFrame, default = none
        pandas.DataFrame returned by trained model using create_model().


    plot: str, default = '2d'
        Enter abbreviation of type of plot. The current list of plots supported are
        (Name - Abbreviated String):

        * Support, Confidence and Lift (2d) - '2d'
        * Support, Confidence and Lift (3d) - '3d'


    scale: float, default = 1
        The resolution scale of the figure.

    display_format: str, default = None
        To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.

    Returns:
        None

    """

    # error handling
    import pandas as pd

    # check if model is a pandas dataframe
    if not isinstance(model, pd.DataFrame):
        raise TypeError("Model needs to be a pandas.DataFrame object.")

    # check plot parameter
    plot_types = ["2d", "3d"]

    if plot not in plot_types:
        raise ValueError("Plots can only be '2d' or '3d'.")

    # checking display_format parameter
    plot_formats = [None, "streamlit"]

    if display_format not in plot_formats:
        raise ValueError("display_format can only be None or 'streamlit'.")

    if display_format == "streamlit":
        _check_soft_dependencies("streamlit", extra=None, severity="error")
        import streamlit as st

    """
    error handling ends here
    """

    import cufflinks as cf
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from IPython.display import HTML, clear_output, display, update_display

    cf.go_offline()
    cf.set_config_file(offline=False, world_readable=True)

    # copy dataframe
    data_ = model.copy()

    data_["antecedents"] = [list(i) for i in data_["antecedents"]]

    antecedents_short = []

    for i in data_["antecedents"]:
        a = i[:10]
        antecedents_short.append(a)

    data_["antecedents_short"] = antecedents_short

    data_["consequents"] = [list(i) for i in data_["consequents"]]

    if plot == "2d":

        fig = px.scatter(
            data_,
            x="support",
            y="confidence",
            text="antecedents_short",
            log_x=True,
            size_max=600,
            color="lift",
            hover_data=["antecedents", "consequents"],
            opacity=0.5,
        )

        fig.update_traces(textposition="top center")
        fig.update_layout(plot_bgcolor="rgb(240,240,240)")

        fig.update_layout(
            height=800 * scale, title_text="2D Plot of Support, Confidence and Lift"
        )

    if plot == "3d":

        fig = px.scatter_3d(
            data_,
            x="support",
            y="confidence",
            z="lift",
            color="antecedent support",
            title="3d Plot for Rule Mining",
            opacity=0.7,
            width=900 * scale,
            height=800 * scale,
            hover_data=["antecedents", "consequents"],
        )

    if display_format == "streamlit":
        st.write(fig)
    else:
        fig.show()


def get_rules(
    data,
    transaction_id,
    item_id,
    ignore_items=None,
    metric="confidence",
    threshold=0.5,
    min_support=0.05,
):

    """
    Magic function to get Association Rules in Power Query / Power BI.
    """

    setup(
        data=data,
        transaction_id=transaction_id,
        item_id=item_id,
        ignore_items=ignore_items,
    )
    dataset = create_model(
        metric=metric, threshold=threshold, min_support=min_support, round=4
    )

    return dataset
