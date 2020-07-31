#import libraries
import pandas as pd
import sys

fix_loc = 'C:/Users/moezs/pycaret-dev/pycaret/datasets/'
file_name = sys.argv[1]
import_loc = str(fix_loc) + str(file_name) + '.csv' 
data = pd.read_csv(import_loc) 
exp_name = sys.argv[2]
gt = sys.argv[3]

from pycaret.clustering import *
clu1 = setup(data, normalize = True, silent=True, html=False, 
             verbose=False, logging=True, experiment_name=exp_name, log_plots=True, log_profile=False)

allowed_models = ['kmeans', 'ap', 'meanshift', 'sc', 'hclust', 'dbscan', 'optics', 'birch', 'kmodes']

no_num_required = ['ap', 'meanshift', 'dbscan', 'optics']

num_clust_required = ['kmeans', 'sc', 'hclust', 'birch', 'kmodes']

for i in allowed_models:
    create_model(i, verbose=False, ground_truth=gt)

print('9 Models with num_clusters = None Created')

for i in num_clust_required:
    create_model(i, verbose=False, num_clusters=3, ground_truth=gt)

print('5 Models with num_clusters = 3 Created')

for i in num_clust_required:
    create_model(i, verbose=False, num_clusters=4, ground_truth=gt)

print('5 Models with num_clusters = 4 Created')

for i in num_clust_required:
    create_model(i, verbose=False, num_clusters=6, ground_truth=gt)

print('5 Models with num_clusters = 6 Created')

for i in num_clust_required:
    create_model(i, verbose=False, num_clusters=8, ground_truth=gt)

print('5 Models with num_clusters = 8 Created')