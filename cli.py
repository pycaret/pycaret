from pycaret.datasets import get_data
data = get_data('boston')

from pycaret.regression import *
reg1 = setup(data, target = 'medv', session_id = 123, silent = True, html = False, verbose = False, n_jobs=1)

available_estimators = ['lr', 'lasso', 'ridge', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 
                            'ransac', 'tr', 'huber', 'kr', 'svm', 'knn', 'dt', 'rf', 'et', 'ada', 'gbr', 
                            'mlp', 'xgboost', 'lightgbm', 'catboost']

#train all models
models = []

for i in available_estimators:
    m  = create_model(i)
    models.append(m)
    print(str(i) + ' Succesfully Trained!')

#tune trained models
tuned_models = []

for i in models:
    t = tune_model(i)
    tuned_models.append(t)
    print(str(i) + ' Succesfully Tuned!')

#ensemble tuned models - bagging method
bagged_models = []

for i in tuned_models:
    try:
        e = ensemble_model(i, method = 'Bagging')
        bagged_models.append(e)
        print(str(e) + ' Succesfully Bagged!')
    except:
        pass

#ensemble tuned models - boosting method
boosted_models = []

for i in tuned_models:
    try:
        e = ensemble_model(i, method = 'Boosting')
        boosted_models.append(e)
        print(str(e) + ' Succesfully Boosted!')
    except:
        pass

#print results
print('Total Models Trained ' + str(len(models)))
print('Total Models Tuned ' + str(len(tuned_models)))
print('Total Models Bagged ' + str(len(bagged_models)))
print('Total Models Boosted ' + str(len(boosted_models)))
print(bagged_models)
print(boosted_models)