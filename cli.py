#dataset and target
dataset = 'juice'
target = 'Purchase'

#checking version
from pycaret.utils import version
version()

import time
t0 = time.time()

#loading dataset
from pycaret.datasets import get_data
data = get_data(dataset, verbose=False)

#init regression
from pycaret.classification import setup
exp1 = setup(data, target = target, silent=True, html=False, verbose=False)

#RECEIPE #1 - SELECT TOP 5 MODELS
from pycaret.classification import compare_models
top5 = compare_models(n_select=5, whitelist = ['dt', 'lr', 'rf', 'lightgbm', 'xgboost'])

#RECEIPE #2 - TUNE TOP 5 MODELS
from pycaret.classification import tune_model
tuned_top5 = [tune_model(i) for i in top5]
print(len(tuned_top5))

#RECIPE #3 
from pycaret.classification import blend_models
blender = blend_models(top5, verbose=False) 
print(blender)

from pycaret.classification import pull
pull()

#FINALIZE BEST MODEL
from pycaret.classification import automl
best_model = automl(optimize='MCC', use_holdout=True)
print(best_model)

t1 = time.time()
tt = round(t1-t0,4)

from pycaret.classification import plot_model
plot_model(best_model, plot = 'confusion_matrix')

from pycaret.classification import create_model
xgboost = create_model('xgboost', verbose=False)

from pycaret.classification import interpret_model
interpret_model(xgboost)

print("Succesfully Completed in {} Seconds".format(tt))