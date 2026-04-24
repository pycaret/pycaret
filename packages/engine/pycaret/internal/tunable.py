# Module: internal.tunable
# Author: Antoni Baum (Yard1) <antoni.baum@protonmail.com>
# License: MIT

# Provides a VotingClassifier which weights can be tuned.

import inspect

from sklearn.base import clone
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

try:
    from collections.abc import Iterable
except Exception:
    from collections import Iterable


class TunableMixin:
    def get_base_sklearn_type(self):
        return next(x for x in self.__class__.__bases__ if "sklearn." in str(x))

    def get_base_sklearn_params(self):
        sklearn_base = self.get_base_sklearn_type()
        sklearn_signature = inspect.signature(sklearn_base.__init__).parameters
        return {k: v for k, v in self.get_params().items() if k in sklearn_signature}

    def get_base_sklearn_object(self):
        """Returns a pure scikit-learn parent of the class. Will be unfitted."""
        sklearn_base = self.get_base_sklearn_type()
        params = self.get_base_sklearn_params()
        sklearn_object = sklearn_base(**params)
        return clone(sklearn_object)


class TunableMLPClassifier(MLPClassifier, TunableMixin):
    """
    A MLPClassifier with hidden layer sizes being kwargs instead of a list/tuple, allowing
    for tuning.

    The kwargs need to be in format ``hidden_layer_size_n``, where n is an integer corresponding
    to the index of the layer.

    If ``hidden_layer_sizes`` parameter is changed with ``set_params()``, ``hidden_layer_size_n``
    parameters will change as well, and vice versa.

    scikit-learn description below:

    Multi-layer Perceptron classifier.

    This model optimizes the log-loss function using LBFGS or stochastic
    gradient descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed
          by Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate at each
          time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when ``solver='sgd'``.

    learning_rate_init : double, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : boolean, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least tol for
        ``n_iter_no_change`` consecutive epochs. The split is stratified,
        except in a multilabel setting.
        Only effective when solver='sgd' or 'adam'

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

        .. versionadded:: 0.20

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of loss function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of loss function calls.
        Note that number of loss function calls will be greater than or equal
        to the number of iterations for the `MLPClassifier`.

        .. versionadded:: 0.22

    **kwargs:
        Hidden layer sizes in format ``hidden_layer_size_n`` where ``n``
        is an integer corresponding to the index of the estimator.
        If value is lesser or equal to zero, the hidden layer will be removed.
        Will overwrite ``hidden_layer_sizes``.

    Attributes
    ----------
    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.

    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : string
        Name of the output activation function.


    Examples
    --------
    >>> from sklearn.neural_network import MLPClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.038..., 0.961...]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    >>> clf.score(X_test, y_test)
    0.8...

    Notes
    -----
    MLPClassifier trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense numpy arrays or
    sparse scipy arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
        self,
        hidden_layer_sizes=None,
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        **kwargs,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self._hidden_layer_size_kwargs_to_hidden_layer_sizes(kwargs)
        super().__init__(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _hidden_layer_size_kwargs_to_hidden_layer_sizes(self, kwargs):
        if not self.hidden_layer_sizes:
            self.hidden_layer_sizes = [100]
        if not isinstance(self.hidden_layer_sizes, Iterable):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        if not isinstance(self.hidden_layer_sizes, list):
            self.hidden_layer_sizes = list(self.hidden_layer_sizes)
        reset_layers = False
        for k, v in kwargs.items():
            if k.startswith("hidden_layer_size_") and not (
                k in self.__dict__ and self.__dict__[k] == v
            ):
                try:
                    hidden_layer_size = k.split("_")
                    hidden_layer_size = int(hidden_layer_size[3])
                    if v <= 0:
                        self.hidden_layer_sizes.pop(hidden_layer_size)
                    else:
                        if hidden_layer_size < len(self.hidden_layer_sizes):
                            self.hidden_layer_sizes[hidden_layer_size] = v
                        else:
                            self.hidden_layer_sizes = (
                                self.hidden_layer_sizes
                                + [1]
                                * (hidden_layer_size - len(self.hidden_layer_sizes))
                                + [v]
                            )
                    reset_layers = True
                except Exception:
                    pass
        if reset_layers:
            self._hidden_layer_sizes_to_hidden_layer_size_kwargs()

    def _hidden_layer_sizes_to_hidden_layer_size_kwargs(self):
        to_delete = []
        for k, v in self.__dict__.items():
            if k.startswith("hidden_layer_size_") and int(k.split("_")[3]) >= len(
                self.hidden_layer_sizes
            ):
                to_delete.append(k)
        for k in to_delete:
            delattr(self, k)
        for i, w in enumerate(self.hidden_layer_sizes):
            if not (
                f"hidden_layer_size_{i}" in self.__dict__
                and self.__dict__[f"hidden_layer_size_{i}"] == w
            ):
                setattr(self, f"hidden_layer_size_{i}", w)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        self._hidden_layer_size_kwargs_to_hidden_layer_sizes(params)
        super().set_params(
            **{
                k: v
                for k, v in params.items()
                if not k.startswith("hidden_layer_size_")
            }
        )
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        r = super().get_params(deep=deep)
        if self.hidden_layer_sizes:
            for i, w in enumerate(self.hidden_layer_sizes):
                if f"hidden_layer_size_{i}" not in r:
                    r[f"hidden_layer_size_{i}"] = w
        return r

    def fit(self, X, y, **fit_params):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """
        return super().fit(X, y)

    def _partial_fit(self, X, y, *args, classes=None, **fit_params):
        return super()._partial_fit(X, y, classes=classes)


class TunableMLPRegressor(MLPRegressor, TunableMixin):
    """
    A MLPRegressor with hidden layer sizes being kwargs instead of a list/tuple, allowing
    for tuning.

    The kwargs need to be in format ``hidden_layer_size_n``, where n is an integer corresponding
    to the index of the layer.

    If ``hidden_layer_sizes`` parameter is changed with ``set_params()``, ``hidden_layer_size_n``
    parameters will change as well, and vice versa.

    scikit-learn description below:

    Multi-layer Perceptron regressor.

    This model optimizes the squared-loss using LBFGS or stochastic gradient
    descent.

    .. versionadded:: 0.18

    Parameters
    ----------
    hidden_layer_sizes : tuple, length = n_layers - 2, default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'logistic', the logistic sigmoid function,
          returns f(x) = 1 / (1 + exp(-x)).

        - 'tanh', the hyperbolic tan function,
          returns f(x) = tanh(x).

        - 'relu', the rectified linear unit function,
          returns f(x) = max(0, x)

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization.

        - 'lbfgs' is an optimizer in the family of quasi-Newton methods.

        - 'sgd' refers to stochastic gradient descent.

        - 'adam' refers to a stochastic gradient-based optimizer proposed by
          Kingma, Diederik, and Jimmy Ba

        Note: The default solver 'adam' works pretty well on relatively
        large datasets (with thousands of training samples or more) in terms of
        both training time and validation score.
        For small datasets, however, 'lbfgs' can converge faster and perform
        better.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers.
        If the solver is 'lbfgs', the classifier will not use minibatch.
        When set to "auto", `batch_size=min(200, n_samples)`

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.

        - 'constant' is a constant learning rate given by
          'learning_rate_init'.

        - 'invscaling' gradually decreases the learning rate ``learning_rate_``
          at each time step 't' using an inverse scaling exponent of 'power_t'.
          effective_learning_rate = learning_rate_init / pow(t, power_t)

        - 'adaptive' keeps the learning rate constant to
          'learning_rate_init' as long as training loss keeps decreasing.
          Each time two consecutive epochs fail to decrease training loss by at
          least tol, or fail to increase validation score by at least tol if
          'early_stopping' is on, the current learning rate is divided by 5.

        Only used when solver='sgd'.

    learning_rate_init : double, default=0.001
        The initial learning rate used. It controls the step-size
        in updating the weights. Only used when solver='sgd' or 'adam'.

    power_t : double, default=0.5
        The exponent for inverse scaling learning rate.
        It is used in updating effective learning rate when the learning_rate
        is set to 'invscaling'. Only used when solver='sgd'.

    max_iter : int, default=200
        Maximum number of iterations. The solver iterates until convergence
        (determined by 'tol') or this number of iterations. For stochastic
        solvers ('sgd', 'adam'), note that this determines the number of epochs
        (how many times each data point will be used), not the number of
        gradient steps.

    shuffle : bool, default=True
        Whether to shuffle samples in each iteration. Only used when
        solver='sgd' or 'adam'.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization, train-test split if early stopping is used, and batch
        sampling when solver='sgd' or 'adam'.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        unless ``learning_rate`` is set to 'adaptive', convergence is
        considered to be reached and training stops.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous
        call to fit as initialization, otherwise, just erase the
        previous solution. See :term:`the Glossary <warm_start>`.

    momentum : float, default=0.9
        Momentum for gradient descent update.  Should be between 0 and 1. Only
        used when solver='sgd'.

    nesterovs_momentum : boolean, default=True
        Whether to use Nesterov's momentum. Only used when solver='sgd' and
        momentum > 0.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If set to true, it will automatically set
        aside 10% of training data as validation and terminate training when
        validation score is not improving by at least ``tol`` for
        ``n_iter_no_change`` consecutive epochs.
        Only effective when solver='sgd' or 'adam'

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1.
        Only used if early_stopping is True

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam,
        should be in [0, 1). Only used when solver='adam'

    epsilon : float, default=1e-8
        Value for numerical stability in adam. Only used when solver='adam'

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet ``tol`` improvement.
        Only effective when solver='sgd' or 'adam'

        .. versionadded:: 0.20

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.
        The solver iterates until convergence (determined by 'tol'), number
        of iterations reaches max_iter, or this number of function calls.
        Note that number of function calls will be greater than or equal to
        the number of iterations for the MLPRegressor.

        .. versionadded:: 0.22

    Attributes
    ----------
    loss_ : float
        The current loss computed with the loss function.

    coefs_ : list, length n_layers - 1
        The ith element in the list represents the weight matrix corresponding
        to layer i.

    intercepts_ : list, length n_layers - 1
        The ith element in the list represents the bias vector corresponding to
        layer i + 1.

    n_iter_ : int,
        The number of iterations the solver has ran.

    n_layers_ : int
        Number of layers.

    n_outputs_ : int
        Number of outputs.

    out_activation_ : string
        Name of the output activation function.

    Examples
    --------
    >>> from sklearn.neural_network import MLPRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=200, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=1)
    >>> regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
    >>> regr.predict(X_test[:2])
    array([-0.9..., -7.1...])
    >>> regr.score(X_test, y_test)
    0.4...

    Notes
    -----
    MLPRegressor trains iteratively since at each time step
    the partial derivatives of the loss function with respect to the model
    parameters are computed to update the parameters.

    It can also have a regularization term added to the loss function
    that shrinks model parameters to prevent overfitting.

    This implementation works with data represented as dense and sparse numpy
    arrays of floating point values.

    References
    ----------
    Hinton, Geoffrey E.
        "Connectionist learning procedures." Artificial intelligence 40.1
        (1989): 185-234.

    Glorot, Xavier, and Yoshua Bengio. "Understanding the difficulty of
        training deep feedforward neural networks." International Conference
        on Artificial Intelligence and Statistics. 2010.

    He, Kaiming, et al. "Delving deep into rectifiers: Surpassing human-level
        performance on imagenet classification." arXiv preprint
        arXiv:1502.01852 (2015).

    Kingma, Diederik, and Jimmy Ba. "Adam: A method for stochastic
        optimization." arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
        self,
        hidden_layer_sizes=None,
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        **kwargs,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self._hidden_layer_size_kwargs_to_hidden_layer_sizes(kwargs)
        super().__init__(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )

    def _hidden_layer_size_kwargs_to_hidden_layer_sizes(self, kwargs):
        if not self.hidden_layer_sizes:
            self.hidden_layer_sizes = [100]
        if not isinstance(self.hidden_layer_sizes, Iterable):
            self.hidden_layer_sizes = [self.hidden_layer_sizes]
        if not isinstance(self.hidden_layer_sizes, list):
            self.hidden_layer_sizes = list(self.hidden_layer_sizes)
        reset_layers = False
        for k, v in kwargs.items():
            if k.startswith("hidden_layer_size_") and not (
                k in self.__dict__ and self.__dict__[k] == v
            ):
                try:
                    hidden_layer_size = k.split("_")
                    hidden_layer_size = int(hidden_layer_size[3])
                    if v <= 0:
                        self.hidden_layer_sizes.pop(hidden_layer_size)
                    else:
                        if hidden_layer_size < len(self.hidden_layer_sizes):
                            self.hidden_layer_sizes[hidden_layer_size] = v
                        else:
                            self.hidden_layer_sizes = (
                                self.hidden_layer_sizes
                                + [1]
                                * (hidden_layer_size - len(self.hidden_layer_sizes))
                                + [v]
                            )
                    reset_layers = True
                except Exception:
                    pass
        if reset_layers:
            self._hidden_layer_sizes_to_hidden_layer_size_kwargs()

    def _hidden_layer_sizes_to_hidden_layer_size_kwargs(self):
        to_delete = []
        for k, v in self.__dict__.items():
            if k.startswith("hidden_layer_size_") and int(k.split("_")[3]) >= len(
                self.hidden_layer_sizes
            ):
                to_delete.append(k)
        for k in to_delete:
            delattr(self, k)
        for i, w in enumerate(self.hidden_layer_sizes):
            if not (
                f"hidden_layer_size_{i}" in self.__dict__
                and self.__dict__[f"hidden_layer_size_{i}"] == w
            ):
                setattr(self, f"hidden_layer_size_{i}", w)

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        self._hidden_layer_size_kwargs_to_hidden_layer_sizes(params)
        super().set_params(
            **{
                k: v
                for k, v in params.items()
                if not k.startswith("hidden_layer_size_")
            }
        )
        return self

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        r = super().get_params(deep=deep)
        if self.hidden_layer_sizes:
            for i, w in enumerate(self.hidden_layer_sizes):
                if f"hidden_layer_size_{i}" not in r:
                    r[f"hidden_layer_size_{i}"] = w
        return r

    def fit(self, X, y, **fit_params):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : returns a trained MLP model.
        """
        return super().fit(X, y)

    def _partial_fit(self, X, y, *args, **fit_params):
        return super()._partial_fit(X, y)


class TunableVotingClassifier(VotingClassifier, TunableMixin):
    """
    A VotingClassifier with weights being kwargs instead of a list, allowing
    for tuning.

    The kwargs need to be in format ``weight_n``, where n is an integer corresponding
    to the index of the estimator.

    If ``weights`` parameter is changed with ``set_params()``, ``weight_n`` parameters
    will change as well, and vice versa.

    scikit-learn description below:

    Soft Voting/Majority Rule classifier for unfitted estimators.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'``
        using ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted.

        .. deprecated:: 0.22
           Using ``None`` to drop an estimator is deprecated in 0.22 and
           support will be dropped in 0.24. Use the string ``'drop'`` instead.

    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

    **kwargs:
        Weights in format ``weight_n`` where ``n`` is an integer corresponding
        to the index of the estimator. Will overwrite ``weights``.

    Attributes
    ----------
    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    classes_ : array-like of shape (n_predictions,)
        The classes labels.

    See Also
    --------
    VotingRegressor: Prediction voting regressor.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = VotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    """

    def __init__(
        self,
        estimators,
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
        **kwargs,
    ):
        self.weights = weights
        self._weight_kwargs_to_weights(kwargs, estimators=estimators)
        super().__init__(
            estimators=estimators,
            voting=voting,
            weights=self.weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose,
        )

    def _weight_kwargs_to_weights(self, kwargs, estimators=None):
        if estimators is None:
            estimators = self.estimators
        if not self.weights:
            self.weights = [1 for x in estimators]
        if len(self.weights) < len(estimators):
            self.weights += [1] * (len(self.weights) - len(estimators))
        for k, v in kwargs.items():
            if k.startswith("weight_"):
                try:
                    weight = k.split("_")
                    weight = int(weight[1])
                    self.weights[weight] = v
                except Exception:
                    pass
        self._weights_to_weight_kwargs()

    def _weights_to_weight_kwargs(self):
        for i, w in enumerate(self.weights):
            if not (
                f"weight_{i}" in self.__dict__ and self.__dict__[f"weight_{i}"] == w
            ):
                setattr(self, f"weight_{i}", w)

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the stacking estimator, the individual estimator of
            the stacking estimators can also be set, or can be removed by
            setting them to 'drop'.
        """
        super()._set_params("estimators", **params)
        self._weight_kwargs_to_weights(params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well.
        """
        r = super()._get_params("estimators", deep=deep)
        if self.weights:
            for i, w in enumerate(self.weights):
                if f"weight_{i}" not in r:
                    r[f"weight_{i}"] = w
        return r


class TunableVotingRegressor(VotingRegressor, TunableMixin):
    """
    A VotingRegressor with weights being kwargs instead of a list, allowing
    for tuning.

    The kwargs need to be in format ``weight_n``, where n is an integer corresponding
    to the index of the estimator.

    If ``weights`` parameter is changed with ``set_params()``, ``weight_n`` parameters
    will change as well, and vice versa.

    scikit-learn description below:

    Prediction voting regressor for unfitted estimators.

    .. versionadded:: 0.21

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    Read more in the :ref:`User Guide <voting_regressor>`.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        ``set_params``.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted.

        .. deprecated:: 0.22
           Using ``None`` to drop an estimator is deprecated in 0.22 and
           support will be dropped in 0.24. Use the string ``'drop'`` instead.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : Bunch
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    See Also
    --------
    VotingClassifier: Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2)])
    >>> print(er.fit(X, y).predict(X))
    [ 3.3  5.7 11.8 19.7 28.  40.3]
    """

    def __init__(
        self,
        estimators,
        *,
        weights=None,
        n_jobs=None,
        verbose=False,
        **kwargs,
    ):
        self.weights = weights
        self._weight_kwargs_to_weights(kwargs, estimators=estimators)
        super().__init__(
            estimators=estimators,
            weights=self.weights,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def _weight_kwargs_to_weights(self, kwargs, estimators=None):
        if estimators is None:
            estimators = self.estimators
        if not self.weights:
            self.weights = [1 for x in estimators]
        if len(self.weights) < len(estimators):
            self.weights += [1] * (len(self.weights) - len(estimators))
        for k, v in kwargs.items():
            if k.startswith("weight_"):
                try:
                    weight = k.split("_")
                    weight = int(weight[1])
                    self.weights[weight] = v
                except Exception:
                    pass
        self._weights_to_weight_kwargs()

    def _weights_to_weight_kwargs(self):
        for i, w in enumerate(self.weights):
            if not (
                f"weight_{i}" in self.__dict__ and self.__dict__[f"weight_{i}"] == w
            ):
                setattr(self, f"weight_{i}", w)

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the stacking estimator, the individual estimator of
            the stacking estimators can also be set, or can be removed by
            setting them to 'drop'.
        """
        super()._set_params("estimators", **params)
        self._weight_kwargs_to_weights(params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various classifiers and the parameters
            of the classifiers as well.
        """
        r = super()._get_params("estimators", deep=deep)
        if self.weights:
            for i, w in enumerate(self.weights):
                if f"weight_{i}" not in r:
                    r[f"weight_{i}"] = w
        return r
