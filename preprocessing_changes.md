Refactor preprocessing module
-----------------------------

Ideas:

- Drop preprocessing.py
- The pipeline is created in the setup method in tabular_experiment.py
- Pipeline consists ONLY of sklearn, imbalanced-learn, category-encoders
  estimators. No custom estimators since we want the pipeline to work
  without pycaret.


Changes:

- Added category_encoders library to requirements

- Data attributes train, test, X, y, X_train, etc... are now properties of the
  experiment. The dataset is stored only once in self.data. Train and test indices
  are stored in self.idx = [len(train), len(test)].
  
- The `pca_components` parameter is now equal for all methods. Accepts total
  or fraction of components.
- `remove_low_variance` is deprecated. Use `low_variance_threshold` instead. Now
  works for all numerical columns (is done after encoding).
- Remove `remove_perfect_collinearity` parameter. Same can be achieved
  setting multicollinearity_threshold=1.
- Remove `trigonometry_features` parameter.
- Removed `polynomial_threshold` parameter.
- Removed `create_cluster` parameter.
- Removed `cluster_iter` parameter.
- Changed the way encoding works. Now defaults to LeaveOneOut and another
  estimator can be selected.
- Reorder of the parameters to a logical order (following the pipeline)


Preprocessing steps:

- create features form datetime columns
- encoding: DONE
- simple imputation: DONE
- iterative imputation: DONE
- transformation: DONE
- normalization: DONE
- low variance: DONE
- remove multicollinearity: DONE
- remove outliers: DONE
- polynomial features: DONE
- fix imbalance: DONE
- pca: DONE
