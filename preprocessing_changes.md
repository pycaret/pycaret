Refactor preprocessing module
-----------------------------

Ideas:

- Drop preprocessing.py
- The pipeline is created in the setup method in tabular_experiment.py
- Pipeline consists ONLY of sklearn, imbalanced-learn, category-encoder
  estimators. No custom estimators since we want the pipeline to work
  without pycaret.


Changes:

- Data attributes train, test, X, y, X_train, etc... are now properties of the
  experiment. The dataset is stored only once in self.data. Train and test indices
  are stored in self.idx = [len(train), len(test)].
  
- The `pca_components` parameter is now equal for all methods. Accepts total
  or fraction of components.
- `remove_low_variance` is deprecated. Use `low_variance_threshold` instead. Now
  works for all numerical columns (is done after encoding).
- Remove `trigonometry_features` parameter.
- Removed `polynomial_threshold` parameter.



Preprocessing steps:

- simple imputation: DONE
- normalization: DONE
- low variance: DONE
- pca: DONE
- polynomial features: DONE
- fix imbalance: DONE

