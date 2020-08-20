# Module: Preprocess
# Author: Fahad Akbar <m.akbar@queensu.ca>
# License: MIT




import pandas as pd
import numpy as np
import ipywidgets as wg 
from IPython.display import display
from ipywidgets import Layout
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import KBinsDiscretizer
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.pca import PCA as PCA_od
from sklearn import cluster
from scipy import stats
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from lightgbm import LGBMClassifier as lgbmc
from lightgbm import LGBMRegressor as lgbmr
import sys 
from sklearn.pipeline import Pipeline
from sklearn import metrics
import datefinder
from datetime import datetime
import calendar
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

#ignore warnings
import warnings
warnings.filterwarnings('ignore') 

#_____________________________________________________________________________________________________________________________

class DataTypes_Auto_infer(BaseEstimator,TransformerMixin):
  '''
    - This will try to infer data types automatically, option to override learent data types is also available.
    - This alos automatically delets duplicate columns (values or same colume name), removes rows where target variable is null and 
      remove columns and rows where all the records are null
  '''

  def __init__(self,target,ml_usecase,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=True): # nothing to define
    '''
    User to define the target (y) variable
      args:
        target: string, name of the target variable
        ml_usecase: string , 'regresson' or 'classification . For now, only supports two  class classification
        - this is useful in case target variable is an object / string . it will replace the strings with integers
        categorical_features: list of categorical features, default None, when None best guess will be used to identify categorical features
        numerical_features: list of numerical features, default None, when None best guess will be used to identify numerical features
        time_features: list of date/time features, default None, when None best guess will be used to identify date/time features    
  '''
    self.target = target
    self.ml_usecase= ml_usecase
    self.categorical_features =categorical_features
    self.numerical_features = numerical_features
    self.time_features =time_features
    self.features_todrop = features_todrop
    self.display_types = display_types
  
  def fit(self,dataset,y=None): # learning data types of all the columns
    '''
    Args: 
      data: accepts a pandas data frame
    Returns:
      Panda Data Frame
    '''
    data = dataset.copy()
    # remove sepcial char from column names
    #data.columns= data.columns.str.replace('[,]','')

    # we will take float as numberic, object as categorical from the begning
    # fir int64, we will check to see what is the proportion of unique counts to the total lenght of the data
    # if proportion is lower, then it is probabaly categorical 
    # however, proportion can be lower / disturebed due to samller denominator (total lenghth / number of samples)
    # so we will take the following chart
    # 0-50 samples, threshold is 24%
    # 50-100 samples, th is 12%
    # 50-250 samples , th is 4.8%
    # 250-500 samples, th is 2.4%
    # 500 and above 2% or belwo
   
    # if there are inf or -inf then replace them with NaN
    data.replace([np.inf,-np.inf],np.NaN,inplace=True)

    # also make sure that all the column names are string 
    data.columns = [str(i) for i in data.columns]
  
   
    # we canc check if somehow everything is object, we can try converting them in float
    for i in data.select_dtypes(include=['object']).columns:
      try:
        data[i] = data[i].astype('int64')
      except:
        None
    
    # if data type is bool , convert to categorical
    for i in data.columns:
      if data[i].dtype=='bool':
        data[i] = data[i].astype('object')
    

    # some times we have id column in the data set, we will try to find it and then  will drop it if found
    len_samples = len(data)
    self.id_columns = []
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtype in ['int64','float64']:
        if i not in self.numerical_features:
          if sum(data[i].isna()) == 0: 
            if len(data[i].unique()) == len_samples:
              # we extract column and sort it
              features = data[i].sort_values()
              # no we subtract i+1-th value from i-th (calculating increments)
              increments = features.diff()[1:]
              # if all increments are 1 (with float tolerance), then the column is ID column
              if sum(np.abs(increments-1) < 1e-7) == len_samples-1:
                self.id_columns.append(i)
      
    data_len = len(data)                        
        
    # wiith csv , if we have any null in  a colum that was int , panda will read it as float.
    # so first we need to convert any such floats that have NaN and unique values are lower than 20
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtypes == 'float64':
        # count how many Nas are there
        na_count = sum(data[i].isna())
        # count how many digits are there that have decimiles
        count_float = np.nansum([ False if r.is_integer() else True for r in data[i]])
        # total decimiels digits
        count_float = count_float - na_count # reducing it because we know NaN is counted as a float digit
        # now if there isnt any float digit , & unique levales are less than 20 and there are Na's then convert it to object
        if ( (count_float == 0) & (len(data[i].unique()) <=20) & (na_count>0) ):
          data[i] = data[i].astype('object')
        

    # should really be an absolute number say 20
    # length = len(data.iloc[:,0])
    # if length in range(0,51):
    #   th=.25
    # elif length in range(51,101):
    #   th=.12
    # elif length in range(101,251):
    #   th=.048
    # elif length in range(251,501):
    #   th=.024
    # elif length > 500:
    #   th=.02

    # if column is int and unique counts are more than two, then: (exclude target)
    for i in data.drop(self.target,axis=1).columns:
      if data[i].dtypes == 'int64': #((data[i].dtypes == 'int64') & (len(data[i].unique())>2))
        if len(data[i].unique()) <=20: #hard coded
          data[i]= data[i].apply(str)
        else:
          data[i]= data[i].astype('float64')


    # # if colum is objfloat  and only have two unique counts , this is probabaly one hot encoded
    # # make it object
    for i in data.columns:
      if ((data[i].dtypes == 'float64') & (len(data[i].unique())==2)):
        data[i]= data[i].apply(str)
    
    
    #for time & dates
    #self.drop_time = [] # for now we are deleting time columns
    for i in data.drop(self.target,axis=1).columns:
      # we are going to check every first row of every column and see if it is a date
      match = datefinder.find_dates(data[i].values[0]) # to get the first value
      try:
        for m in match:
          if isinstance(m, datetime) == True:
            data[i] = pd.to_datetime(data[i])
            #self.drop_time.append(i)  # for now we are deleting time columns
      except:
        continue

    # now in case we were given any specific columns dtypes in advance , we will over ride theos 
    if len(self.categorical_features) > 0:
      for i in self.categorical_features:
        try:
          data[i]=data[i].apply(str)
        except:
          data[i]=dataset[i].apply(str)

    if len(self.numerical_features) > 0:
      for i in self.numerical_features:
        try:
          data[i]=data[i].astype('float64')
        except:
          data[i]=dataset[i].astype('float64')
    
    if len(self.time_features) > 0:
      for i in self.time_features:
        try:
          data[i]=pd.to_datetime(data[i])
        except:
          data[i]=pd.to_datetime(dataset[i])

    # table of learent types
    self.learent_dtypes = data.dtypes
    #self.training_columns = data.drop(self.target,axis=1).columns
    
    # if there are inf or -inf then replace them with NaN
    data.replace([np.inf,-np.inf],np.NaN,inplace=True)
    
    # lets remove duplicates
    # remove duplicate columns (columns with same values)
    #(too expensive on bigger data sets)
    # data_c = data.T.drop_duplicates()
    # data = data_c.T
    #remove columns with duplicate name 
    data = data.loc[:,~data.columns.duplicated()]
    # Remove NAs
    data.dropna(axis=0, how='all', inplace=True)
    data.dropna(axis=1, how='all', inplace=True)
    # remove the row if target column has NA
    data = data[~data[self.target].isnull()]
            

    #self.training_columns = data.drop(self.target,axis=1).columns

    # since due to transpose , all data types have changed, lets change the dtypes to original---- not required any more since not transposing any more
    # for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
    #   data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    
    if self.display_types == True:
      display(wg.Text(value="Following data types have been inferred automatically, if they are correct press enter to continue or type 'quit' otherwise.",layout =Layout(width='100%')),display_id='m1')
      
      dt_print_out = pd.DataFrame(self.learent_dtypes, columns=['Feature_Type'])
      dt_print_out['Data Type'] = ""
      
      for i in dt_print_out.index:
        if i != self.target:
          if dt_print_out.loc[i,'Feature_Type'] == 'object':
            dt_print_out.loc[i,'Data Type'] = 'Categorical'
          elif dt_print_out.loc[i,'Feature_Type'] == 'float64':
            dt_print_out.loc[i,'Data Type'] = 'Numeric'
          elif dt_print_out.loc[i,'Feature_Type'] == 'datetime64[ns]':
            dt_print_out.loc[i,'Data Type'] = 'Date'
          #elif dt_print_out.loc[i,'Feature_Type'] == 'int64':
          #  dt_print_out.loc[i,'Data Type'] = 'Categorical'
        else:
          dt_print_out.loc[i,'Data Type'] = 'Label'

      # for ID column:
      for i in dt_print_out.index:
        if i in self.id_columns:
          dt_print_out.loc[i,'Data Type'] = 'ID Column'
      
      # if we added the dummy  target column , then drop it 
      dt_print_out.drop(index='dummy_target',errors='ignore',inplace=True)
      # drop any columns that were asked to drop
      dt_print_out.drop(index=self.features_todrop,errors='ignore',inplace=True)


      display(dt_print_out[['Data Type']])
      self.response = input()
      

      if self.response in ['quit','Quit','exit','EXIT','q','Q','e','E','QUIT','Exit']:
        sys.exit('Read the documentation of setup to learn how to overwrite data types over the inferred types. setup function must run again before you continue modeling.')
      
      
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)
    
    return(data)
  
  def transform(self,dataset,y=None):
    '''
      Args: 
        data: accepts a pandas data frame
      Returns:
        Panda Data Frame
    '''
    data = dataset.copy()
    # also make sure that all the column names are string 
    data.columns = [str(i) for i in data.columns]

    # if there are inf or -inf then replace them with NaN
    data.replace([np.inf,-np.inf],np.NaN,inplace=True)
    
    # remove sepcial char from column names
    #data.columns= data.columns.str.replace('[,]','')

    #very first thing we need to so is to check if the training and test data hace same columns
    #exception checking   
    import sys

    for i in self.final_training_columns:  
      if i not in data.columns:
        sys.exit('(Type Error): test data does not have column ' + str(i) + " which was used for training")

    ## we only need to take test columns that we used in ttaining (test in production may have a lot more columns)
    data = data[self.final_training_columns]

    
    # just keep picking the data and keep applying to the test data set (be mindful of target variable)
    for i in data.columns: # we are taking all the columns in test , so we dot have to worry about droping target column
      data[i] = data[i].astype(self.learent_dtypes[self.learent_dtypes.index==i])
    
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)

     # drop custome columns
    data.drop(self.features_todrop,axis=1,errors='ignore',inplace=True)
    
    return(data)

  # fit_transform
  def fit_transform(self,dataset,y=None):

    data= dataset.copy()
    # since this is for training , we dont nees any transformation since it has already been transformed in fit
    data = self.fit(data)

    # additionally we just need to treat the target variable
    # for ml use ase
    if ((self.ml_usecase == 'classification') &  (data[self.target].dtype=='object')):
      le = LabelEncoder()
      data[self.target] = le.fit_transform(np.array(data[self.target]))

      # now get the replacement dict
      rev= le.inverse_transform(range(0,len(le.classes_)))
      rep = np.array(range(0,len(le.classes_)))
      self.replacement={}
      for i,k in zip(rev,rep):
        self.replacement[i] = k

      # self.u = list(pd.unique(data[self.target]))
      # self.replacement = np.arange(0,len(self.u))
      # data[self.target]= data[self.target].replace(self.u,self.replacement)
      # data[self.target] = data[self.target].astype('int64')
      # self.replacement = pd.DataFrame(dict(target_variable=self.u,replaced_with=self.replacement))

    
    # drop time columns
    #data.drop(self.drop_time,axis=1,errors='ignore',inplace=True)

    # drop id columns
    data.drop(self.id_columns,axis=1,errors='ignore',inplace=True)

    # drop custome columns
    data.drop(self.features_todrop,axis=1,errors='ignore',inplace=True)
    
    # finally save a list of columns that we would need from test data set
    self.final_training_columns = data.drop(self.target,axis=1).columns

    
    return(data)
# _______________________________________________________________________________________________________________________
# Imputation
class Simple_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes all type of data (numerical,categorical & Time).
      Highly recommended to run Define_dataTypes class first
      Numerical values can be imputed with mean or median 
      categorical missing values will be replaced with "Other"
      Time values are imputed with the most frequesnt value
      Ignores target (y) variable    
      Args: 
        Numeric_strategy: string , all possible values {'mean','median'}
        categorical_strategy: string , all possible values {'not_available','most frequent'}
        target: string , name of the target variable

  '''

  def __init__(self,numeric_strategy,categorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.categorical_strategy = categorical_strategy
  
  def fit(self,dataset,y=None): #
    data = dataset.copy()
    # make a table for numerical variable with strategy stats
    if self.numeric_strategy == 'mean':
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmean)
    else:
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmedian)

    self.numeric_columns = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).columns

    #for Catgorical , 
    if self.categorical_strategy == 'most frequent':
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_stats = pd.DataFrame(columns=self.categorical_columns) # place holder
      for i in (self.categorical_stats.columns):
        self.categorical_stats.loc[0,i] = data[i].value_counts().index[0]
    else:
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
    
    # for time, there is only one way, pick up the most frequent one
    self.time_columns = data.drop(self.target,axis=1).select_dtypes(include=['datetime64[ns]']).columns
    self.time_stats = pd.DataFrame(columns=self.time_columns) # place holder
    for i in (self.time_columns):
      self.time_stats.loc[0,i] = data[i].value_counts().index[0]
    return(data)

      
  
  def transform(self,dataset,y=None):
    data = dataset.copy() 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      data[i].fillna(s,inplace=True)
    
    # for categorical columns
    if self.categorical_strategy == 'most frequent':
      for i in (self.categorical_stats.columns):
        #data[i].fillna(self.categorical_stats.loc[0,i],inplace=True)
        data[i] = data[i].fillna(self.categorical_stats.loc[0,i])
        data[i] = data[i].apply(str)    
    else: # this means replace na with "not_available"
      for i in (self.categorical_columns):
        data[i].fillna("not_available",inplace=True)
        data[i] = data[i].apply(str)
    # for time
    for i in (self.time_stats.columns):
        data[i].fillna(self.time_stats.loc[0,i],inplace=True)
    
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy() 
    data= self.fit(data)
    return(self.transform(data))

# _______________________________________________________________________________________________________________________
# Imputation with surrogate columns
class Surrogate_Imputer(BaseEstimator,TransformerMixin):
  '''
    Imputes feature with surrogate column (numerical,categorical & Time).
      - Highly recommended to run Define_dataTypes class first
      - it is also recommended to only apply this to features where it makes business sense to creat surrogate column
      - feature name has to be provided
      - only able to handle one feature at a time
      - Numerical values can be imputed with mean or median 
      - categorical missing values will be replaced with "Other"
      - Time values are imputed with the most frequesnt value
      - Ignores target (y) variable    
      Args: 
        feature_name: string, provide features name
        feature_type: string , all possible values {'numeric','categorical','date'}
        strategy: string ,all possible values {'mean','median','not_available','most frequent'}
        target: string , name of the target variable

  '''
  def __init__(self,numeric_strategy,categorical_strategy,target_variable):
    self.numeric_strategy = numeric_strategy
    self.target = target_variable
    self.categorical_strategy = categorical_strategy
  
  def fit(self,dataset,y=None): #
    data = dataset.copy()
    # make a table for numerical variable with strategy stats
    if self.numeric_strategy == 'mean':
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmean)
    else:
      self.numeric_stats = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).apply(np.nanmedian)

    self.numeric_columns = data.drop(self.target,axis=1).select_dtypes(include=['float64','int64']).columns
    # also need to learn if any columns had NA in training
    self.numeric_na = pd.DataFrame(columns=self.numeric_columns)
    for i in self.numeric_columns:
      if data[i].isna().any() == True:
        self.numeric_na.loc[0,i] = True
      else:
        self.numeric_na.loc[0,i] = False 

    #for Catgorical , 
    if self.categorical_strategy == 'most frequent':
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_stats = pd.DataFrame(columns=self.categorical_columns) # place holder
      for i in (self.categorical_stats.columns):
        self.categorical_stats.loc[0,i] = data[i].value_counts().index[0]
      # also need to learn if any columns had NA in training, but this is only valid if strategy is "most frequent"
      self.categorical_na = pd.DataFrame(columns=self.categorical_columns)
      for i in self.categorical_columns:
        if sum(data[i].isna()) > 0:
          self.categorical_na.loc[0,i] = True
        else:
          self.categorical_na.loc[0,i] = False        
    else:
      self.categorical_columns = data.drop(self.target,axis=1).select_dtypes(include=['object']).columns
      self.categorical_na = pd.DataFrame(columns=self.categorical_columns)
      self.categorical_na.loc[0,:] = False #(in this situation we are not making any surrogate column)
    
    # for time, there is only one way, pick up the most frequent one
    self.time_columns = data.drop(self.target,axis=1).select_dtypes(include=['datetime64[ns]']).columns
    self.time_stats = pd.DataFrame(columns=self.time_columns) # place holder
    self.time_na = pd.DataFrame(columns=self.time_columns)
    for i in (self.time_columns):
      self.time_stats.loc[0,i] = data[i].value_counts().index[0]
    
    # learn if time columns were NA
    for i in self.time_columns:
      if data[i].isna().any() == True:
        self.time_na.loc[0,i] = True
      else:
        self.time_na.loc[0,i] = False
    
    return(data) # nothing to return

      
  
  def transform(self,dataset,y=None):
    data = dataset.copy() 
    # for numeric columns
    for i,s in zip(data[self.numeric_columns].columns,self.numeric_stats):
      array = data[i].isna()
      data[i].fillna(s,inplace=True)
      # make a surrogate column if there was any
      if self.numeric_na.loc[0,i] == True:
        data[i+"_surrogate"]= array
        # make it string
        data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)

    
    # for categorical columns
    if self.categorical_strategy == 'most frequent':
      for i in (self.categorical_stats.columns):
        #data[i].fillna(self.categorical_stats.loc[0,i],inplace=True)
        array = data[i].isna()
        data[i] = data[i].fillna(self.categorical_stats.loc[0,i])
        data[i] = data[i].apply(str)  
        # make surrogate column
        if self.categorical_na.loc[0,i] == True:
          data[i+"_surrogate"]= array
          # make it string
          data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)
    else: # this means replace na with "not_available"
      for i in (self.categorical_columns):
        data[i].fillna("not_available",inplace=True)
        data[i] = data[i].apply(str)
        # no need to make surrogate since not_available is itself a new colum
    
    # for time
    for i in (self.time_stats.columns):
      array = data[i].isna()
      data[i].fillna(self.time_stats.loc[0,i],inplace=True)
      # make surrogate column
      if self.time_na.loc[0,i] == True:
        data[i+"_surrogate"]= array
        # make it string
        data[i+"_surrogate"]= data[i+"_surrogate"].apply(str)
    
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    data= self.fit(data)
    return(self.transform(data))
# _______________________________________________________________________________________________________________________
# Zero and Near Zero Variance
class Zroe_NearZero_Variance(BaseEstimator,TransformerMixin):
  '''
    - it eliminates the features having zero variance
    - it eliminates the features haveing near zero variance
    - Near zero variance is determined by 
      -1) Count of unique points divided by the total length of the feature has to be lower than a pre sepcified threshold 
      -2) Most common point(count) divided by the second most common point(count) in the feature is greater than a pre specified threshold
      Once both conditions are met , the feature is dropped  
    -Ignores target variable
      
      Args: 
        threshold_1: float (between 0.0 to 1.0) , default is .10 
        threshold_2: int (between 1 to 100), default is 20 
        tatget variable : string, name of the target variable

  '''

  def __init__(self,target,threshold_1=0.1,threshold_2=20):
    self.threshold_1 = threshold_1
    self.threshold_2 = threshold_2
    self.target = target
  
  def fit(self,dataset,y=None): # from training data set we are going to learn what columns to drop
    data = dataset.copy()
    self.to_drop= []
    self.sampl_len = len(data[self.target])
    for i in data.drop(self.target,axis=1).columns:
      # get the number of unique counts
      u = pd.DataFrame( data[i].value_counts()).sort_values(by=i,ascending=False, inplace=False)
      # take len of u and divided it by the total sample numbers, so this will check the 1st rule , has to be low say 10%
      #import pdb; pdb.set_trace()
      first=len(u)/self.sampl_len
      # then check if most common divided by 2nd most common ratio is 20 or more
      if len(u[i]) == 1: # this means that if column is non variance , automatically make the number big to drop it
        second=100
      else:
        second = u.iloc[0,0]/u.iloc[1,0]
    # if both conditions are true then drop the column, however, we dont want to alter column that indicate NA's
      if ((first <= 0.10) and (second >=20) and (i[-10:]!='_surrogate')):
        self.to_drop.append(i) 
    # now drop if the column has zero variance
      if (((second ==100) and (i[-10:]!='_surrogate'))):
        self.to_drop.append(i) 

  
  def transform(self,dataset,y=None): # since it is only for training data set , nothing here
    data= dataset.copy()
    data.drop(self.to_drop,axis=1,inplace=True)
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data= dataset.copy()
    self.fit(data)
    return(self.transform(data))
#____________________________________________________________________________________________________________________________
# rare catagorical variables
class Catagorical_variables_With_Rare_levels(BaseEstimator,TransformerMixin):
  '''
    -Merges levels in catagorical features with more frequent level  if they appear less than a threshold count 
      e.g. Col=[a,a,a,a,b,b,c,c]
      if threshold is set to 2 , then c will be mrged with b because both are below threshold
      There has to be atleast two levels belwo threshold for this to work 
      the process will keep going until all the levels have atleast 2(threshold) counts
    -Only handles catagorical features
    -It is recommended to run the Zroe_NearZero_Variance and Define_dataTypes first
    -Ignores target variable 
      Args: 
        threshold: int , default 10
        target: string , name of the target variable
        new_level_name: string , name given to the new level generated, default 'others'

  '''

  def __init__(self,target,new_level_name='others_infrequent',threshold=.05):
    self.threshold = threshold
    self.target = target
    self.new_level_name = new_level_name
  def fit(self,dataset,y=None): # we will learn for what columnns what are the level to merge as others
    # every level of the catagorical feature has to be more than threshols, if not they will be clubed togather as "others"
    # in order to apply, there should be atleast two levels belwo the threshold ! 
    # creat a place holder
    data = dataset.copy()
    self.ph = pd.DataFrame(columns=data.drop(self.target,axis=1).select_dtypes(include="object").columns)
    #ph.columns = df.columns# catagorical only 
    for i in data[self.ph.columns].columns:
        # determine the infrequebt count
        v_c = data[i].value_counts()
        count_th = round(v_c.quantile(self.threshold))
        a = np.sum(pd.DataFrame(data[i].value_counts().sort_values()) [i]  <= count_th)
        if a >= 2: # rare levels has to be atleast two
          count = pd.DataFrame( data[i].value_counts().sort_values())
          count.columns = ['fre']
          count = count[count['fre']<=count_th]
          to_club = list(count.index)
          self.ph.loc[0,i] = to_club
        else:
          self.ph.loc[0,i] = []
    # # also need to make a place holder that keep records of all the levels , and in case a new level appears in test we will change it to others
    # self.ph_level = pd.DataFrame(columns=data.drop(self.target,axis=1).select_dtypes(include="object").columns)
    # for i in self.ph_level.columns:
    #   self.ph_level.loc[0,i] = list(data[i].value_counts().sort_values().index)

  
  def transform(self,dataset,y=None): # 
    # transorm 
    data = dataset.copy()
    for i in data[self.ph.columns].columns:
      t_replace = self.ph.loc[0,i]
      data[i].replace(to_replace=t_replace,value=self.new_level_name,inplace=True)
    return(data)
  
  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    self.fit(data)
    return(self.transform(data))

# _______________________________________________________________________________________________________________________
# new catagorical level in test
class New_Catagorical_Levels_in_TestData(BaseEstimator,TransformerMixin):
  '''
    -This treats if a new level appears in the test dataset catagorical's feature (i.e a level on whihc model was not trained previously) 
    -It simply replaces the new level in test data set with the most frequent or least frequent level in the same feature in the training data set
    -It is recommended to run the Zroe_NearZero_Variance and Define_dataTypes first
    -Ignores target variable 
      Args: 
        target: string , name of the target variable
        replacement_strategy:string , 'least frequent' or 'most frequent' (default 'most frequent' )

  '''

  def __init__(self,target,replacement_strategy='most frequent'):
    self.target = target
    self.replacement_strategy = replacement_strategy

  def fit(self,data,y=None): 
    # need to make a place holder that keep records of all the levels , and in case a new level appears in test we will change it to others
    self.ph_train_level = pd.DataFrame(columns=data.drop(self.target,axis=1).select_dtypes(include="object").columns)
    for i in self.ph_train_level.columns:
      if self.replacement_strategy == "least frequent": 
        self.ph_train_level.loc[0,i] = list(data[i].value_counts().sort_values().index)
      else:
        self.ph_train_level.loc[0,i] = list(data[i].value_counts().index)

  
  def transform(self,data,y=None): # 
    # transorm 
    # we need to learn the same for test data , and then we will compare to check what levels are new in there
    self.ph_test_level = pd.DataFrame(columns=data.drop(self.target,axis=1,errors='ignore').select_dtypes(include="object").columns)
    for i in self.ph_test_level.columns:
        self.ph_test_level.loc[0,i] = list(data[i].value_counts().sort_values().index)
    
    # new we have levels for both test and train, we will start comparing and replacing levels in test set (Only if test set has new levels)
    for i in self.ph_test_level.columns:
      new = list((set(self.ph_test_level.loc[0,i]) - set(self.ph_train_level.loc[0,i])))
      # now if there is a difference , only then replace it 
      if len(new) > 0 :
        data[i].replace(new,self.ph_train_level.loc[0,i][0],inplace=True)

    return(data)
  
  def fit_transform(self,data,y=None): #There is no transformation happening in training data set, its all about test 
    self.fit(data)
    return(data)


# _______________________________________________________________________________________________________________________
# Group akin features
class Group_Similar_Features(BaseEstimator,TransformerMixin):
  '''
    - Given a list of features , it creates aggregate features 
    - features created are Min, Max, Mean, Median, Mode & Std
    - Only works on numerical features
      Args: 
        list_of_similar_features: list of list, string , e.g. [['col',col2],['col3','col4']]
        group_name: list, group name/names to be added as prefix to aggregate features, e.g ['gorup1','group2']
  '''

  def __init__(self,group_name=[],list_of_grouped_features=[[]]):
    self.list_of_similar_features = list_of_grouped_features
    self.group_name = group_name
    # if list of list not given
    try:
      np.array(self.list_of_similar_features).shape[0]
    except:
      raise("Group_Similar_Features: list_of_grouped_features is not provided as list of list")
      
  
  def fit(self,data,y=None):
    # nothing to learn
      return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()
    # # only going to process if there is an actual missing value in training data set
    if len(self.list_of_similar_features) > 0:
      for f,g in zip(self.list_of_similar_features,self.group_name): 
        data[g+'_Min'] = data[f].apply(np.min,1)
        data[g+'_Max'] = data[f].apply(np.max,1)
        data[g+'_Mean'] = data[f].apply(np.mean,1)
        data[g+'_Median'] = data[f].apply(np.median,1)
        data[g+'_Mode'] = stats.mode(data[f],1)[0]
        data[g+'_Std'] = data[f].apply(np.std,1)
        
      return(data)
    else:
      return(data)

  def fit_transform(self,data,y=None):
    self.fit(data)
    return(self.transform(data))
    

#____________________________________________________________________________________________________________________________________________________________________
# Binning for Continious
class Binning(BaseEstimator,TransformerMixin):
  '''
    - Converts numerical variables to catagorical variable through binning
    - Number of binns are automitically determined through Sturges method
    - Once discretize, original feature will be dropped
        Args:
            features_to_discretize: list of featur names to be binned

  '''

  def __init__(self, features_to_discretize):
    self.features_to_discretize =features_to_discretize
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()
    #only do if features are provided
    if len(self.features_to_discretize) > 0:
      data_t = self.disc.transform(np.array(data[self.features_to_discretize]).reshape(-1,self.len_columns))
      # make pandas data frame
      data_t = pd.DataFrame(data_t,columns=self.features_to_discretize,index=data.index)
      # all these columns are catagorical
      data_t = data_t.astype(str)
      # drop original columns
      data.drop(self.features_to_discretize,axis=1,inplace=True)
      # add newly created columns
      data = pd.concat((data,data_t),axis=1)
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # only do if features are given

    if len(self.features_to_discretize) > 0:

      # place holder for all the features for their binns  
      self.binns = []
      for i in self.features_to_discretize:
        # get numbr of binns
        hist, bin_edg = np.histogram(data[i],bins='sturges')
        self.binns.append(len(hist))

      # how many colums to deal with
      self.len_columns = len(self.features_to_discretize)
      # now do fit transform 
      self.disc = KBinsDiscretizer(n_bins=self.binns, encode='ordinal', strategy='kmeans')
      data_t = self.disc.fit_transform(np.array(data[self.features_to_discretize]).reshape(-1,self.len_columns))
      # make pandas data frame
      data_t = pd.DataFrame(data_t,columns=self.features_to_discretize,index=data.index)
      # all these columns are catagorical
      data_t = data_t.astype(str)
      # drop original columns
      data.drop(self.features_to_discretize,axis=1,inplace=True)
      # add newly created columns
      data = pd.concat((data,data_t),axis=1)

    return(data)
# ______________________________________________________________________________________________________________________
# Scaling & Power Transform
class Scaling_and_Power_transformation(BaseEstimator,TransformerMixin):
  '''
    -Given a data set, applies Min Max, Standar Scaler or Power Transformation (yeo-johnson)
    -it is recommended to run Define_dataTypes first
    - ignores target variable 
      Args: 
        target: string , name of the target variable
        function_to_apply: string , default 'zscore' (standard scaler), all other {'minmaxm','yj','quantile','robust','maxabs'} ( min max,yeo-johnson & quantile power transformation, robust and MaxAbs scaler )

  '''

  def __init__(self,target,function_to_apply='zscore',random_state_quantile=42):
    self.target = target
    self.function_to_apply = function_to_apply
    self.random_state_quantile = random_state_quantile
    # self.transform_target = transform_target
    # self.ml_usecase = ml_usecase
  
  def fit(self,dataset,y=None):
    
    data=dataset.copy()
    # we only want to apply if there are numeric columns
    self.numeric_features = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=["float64",'int64']).columns
    if len(self.numeric_features) > 0:
      if self.function_to_apply == 'zscore':
        self.scale_and_power = StandardScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'minmax':
        self.scale_and_power = MinMaxScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'yj':
        self.scale_and_power = PowerTransformer(method='yeo-johnson',standardize=True)
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'quantile':
        self.scale_and_power = QuantileTransformer(random_state=self.random_state_quantile,output_distribution='normal')
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'robust':
        self.scale_and_power = RobustScaler()
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'maxabs':
        self.scale_and_power = MaxAbsScaler()
        self.scale_and_power.fit(data[self.numeric_features])

      else:
        return(None)
    else:
      return(None)
    

  
  def transform(self,dataset,y=None):
    data = dataset.copy()
    
    if len(self.numeric_features) > 0:
      self.data_t = pd.DataFrame(self.scale_and_power.transform(data[self.numeric_features]))
      # we need to set the same index as original data
      self.data_t.index = data.index
      self.data_t.columns = self.numeric_features
      for i in self.numeric_features:
        data[i]= self.data_t[i]
      return(data)
    
    else:
      return(data) 

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    self.fit(data)
    # convert target if appropriate
    # default behavious is quantile transformer
    # if ((self.ml_usecase == 'regression') and (self.transform_target == True)):
    #   self.scale_and_power_target = QuantileTransformer(random_state=self.random_state_quantile,output_distribution='normal')
    #   data[self.target]=self.scale_and_power_target.fit_transform(np.array(data[self.target]).reshape(-1,1))
      
    return(self.transform(data))

# ______________________________________________________________________________________________________________________
# Scaling & Power Transform
class Target_Transformation(BaseEstimator,TransformerMixin):
  '''
    - Applies Power Transformation (yeo-johnson , Box-Cox) to target variable (Applicable to Regression only)
      - 'bc' for Box_Coc & 'yj' for yeo-johnson, default is Box-Cox
    - if target containes negtive / zero values , yeo-johnson is automatically selected 
    
  '''

  def __init__(self,target,function_to_apply='bc'):
    self.target = target
    self.function_to_apply = function_to_apply
    if self.function_to_apply == 'bc':
      self.function_to_apply = 'box-cox'
    else:
      self.function_to_apply = 'yeo-johnson'

  
  def fit(self,dataset,y=None):
    return(None)
    

  
  def transform(self,dataset,y=None):
    return(dataset) 

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # if target has zero or negative values use yj instead
    if any(data[self.target]<= 0):
      self.function_to_apply = 'yeo-johnson'
    # apply transformation
    self.p_transform_target = PowerTransformer(method=self.function_to_apply)
    data[self.target]=self.p_transform_target.fit_transform(np.array(data[self.target]).reshape(-1,1))
      
    return(data)
# __________________________________________________________________________________________________________________________
# Time feature extractor
class Make_Time_Features(BaseEstimator,TransformerMixin):
  '''
    -Given a time feature , it extracts more features
    - Only accepts / works where feature / data type is datetime64[ns]
    - full list of features is:
      ['month','weekday',is_month_end','is_month_start','hour']
    - all extracted features are defined as string / object
    -it is recommended to run Define_dataTypes first
      Args: 
        time_feature: list of feature names as datetime64[ns] , default empty/none , if empty/None , it will try to pickup dates automatically where data type is datetime64[ns]
        list_of_features: list of required features , default value ['month','weekday','is_month_end','is_month_start','hour']

  '''

  def __init__(self,time_feature=[],list_of_features=['month','weekday','is_month_end','is_month_start','hour']):
    self.time_feature = time_feature
    self.list_of_features_o = list_of_features
    return(None)

  def fit(self,data,y=None):

    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()

    # run fit transform first

    # start making features for every column in the time list
    for i in self.time_feature:
      # make month column if month is choosen
      if 'month' in self.list_of_features_o:
        data[i+"_month"] = [datetime.date(r).month for r in data[i]]
        data[i+"_month"] = data[i+"_month"].apply(str)

      # make weekday column if weekday is choosen ( 0 for monday 6 for sunday)
      if 'weekday' in self.list_of_features_o:
        data[i+"_weekday"] = [datetime.weekday(r) for r in data[i]]
        data[i+"_weekday"] = data[i+"_weekday"].apply(str)
      
      # make Is_month_end column  choosen
      if 'is_month_end' in self.list_of_features_o:
        data[i+"_is_month_end"] = [ 1 if calendar.monthrange(datetime.date(r).year,datetime.date(r).month)[1] == datetime.date(r).day  else 0 for r in data[i] ]
        data[i+"_is_month_end"] = data[i+"_is_month_end"].apply(str)
        
      
      # make Is_month_start column if choosen
      if 'is_month_start' in self.list_of_features_o:
        data[i+"_is_month_start"] = [ 1 if datetime.date(r).day == 1 else 0 for r in data[i] ]
        data[i+"_is_month_start"] = data[i+"_is_month_start"].apply(str)
      
      # make hour column if choosen
      if 'hour' in self.list_of_features_o:
        h = [ datetime.time(r).hour for r in data[i] ]
        if sum(h) > 0:  
          data[i+"_hour"] = h
          data[i+"_hour"] = data[i+"_hour"].apply(str)
    
    # we dont need time columns any more 
    data.drop(self.time_feature,axis=1,inplace=True)

    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # if no columns names are given , then pick datetime columns
    if len(self.time_feature) == 0 :
      self.time_feature = [i for i in data.columns if data[i].dtype == 'datetime64[ns]']
    
    # now start making features for every column in the time list
    for i in self.time_feature:
      # make month column if month is choosen
      if 'month' in self.list_of_features_o:
        data[i+"_month"] = [datetime.date(r).month for r in data[i]]
        data[i+"_month"] = data[i+"_month"].apply(str)

      # make weekday column if weekday is choosen ( 0 for monday 6 for sunday)
      if 'weekday' in self.list_of_features_o:
        data[i+"_weekday"] = [datetime.weekday(r) for r in data[i]]
        data[i+"_weekday"] = data[i+"_weekday"].apply(str)
      
      # make Is_month_end column  choosen
      if 'is_month_end' in self.list_of_features_o:
        data[i+"_is_month_end"] = [ 1 if calendar.monthrange(datetime.date(r).year,datetime.date(r).month)[1] == datetime.date(r).day  else 0 for r in data[i] ]
        data[i+"_is_month_end"] = data[i+"_is_month_end"].apply(str)
        
      
      # make Is_month_start column if choosen
      if 'is_month_start' in self.list_of_features_o:
        data[i+"_is_month_start"] = [ 1 if datetime.date(r).day == 1 else 0 for r in data[i] ]
        data[i+"_is_month_start"] = data[i+"_is_month_start"].apply(str)
      
      # make hour column if choosen
      if 'hour' in self.list_of_features_o:
        h = [ datetime.time(r).hour for r in data[i] ]
        if sum(h) > 0:  
          data[i+"_hour"] = h
          data[i+"_hour"] = data[i+"_hour"].apply(str)
    
    # we dont need time columns any more 
    data.drop(self.time_feature,axis=1,inplace=True)

    return(data)

 #____________________________________________________________________________________________________________________________________________________________________
# Ordinal transformer
class Ordinal(BaseEstimator,TransformerMixin):
  '''
    - converts categorical features into ordinal values 
    - takes a dataframe , and information about column names and ordered categories as dict
    - returns float panda data frame
  '''

  def __init__(self, info_as_dict):
    self.info_as_dict = info_as_dict
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()
    new_data_test = pd.DataFrame(self.enc.transform(data[self.info_as_dict.keys()]),columns= self.info_as_dict.keys(),index= data.index)
    for i in self.info_as_dict.keys():
      data[i] = new_data_test[i]
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # creat categories from given keys in the data set
    cat_list = []
    for i in self.info_as_dict.values():
      i = [np.array(i)]
      cat_list = cat_list + i
    
    # now do fit transform 
    self.enc = OrdinalEncoder(cat_list)
    new_data_train = pd.DataFrame(self.enc.fit_transform(data.loc[:,self.info_as_dict.keys()]),columns=self.info_as_dict,index= data.index )
    # new_data = pd.DataFrame(self.enc.fit_transform(data.loc[:,self.info_as_dict.keys()]))
    for i in self.info_as_dict.keys():
      data[i] = new_data_train[i]
      
    return(data)
# _______________________________________________________________________________________________________________________

# make dummy variables
class Dummify(BaseEstimator,TransformerMixin):
  '''
    - makes one hot encoded variables for dummy variable
    - it is HIGHLY recommended to run the Select_Data_Type class first
    - Ignores target variable

      Args: 
        target: string , name of the target variable
  '''

  def __init__(self,target):
    self.target = target

    
    # creat ohe object 
    self.ohe = OneHotEncoder(handle_unknown='ignore')
  
  def fit(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # we need to learn the column names once the training data set is dummify
      # save non categorical data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      self.target_column =  data[[self.target]]
      # # plus we will only take object data types
      try:
        self.data_columns  = pd.get_dummies(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).columns
      except:
        self.data_columns = []
      # # now fit the trainin column
      self.ohe.fit(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object')))
    else:
      None
    return(None)
 
  def transform(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      # only for test data
      self.data_nonc = data.drop(self.target,axis=1,errors='ignore').select_dtypes(exclude=('object'))
      # fit without target and only categorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      #now put target , numerical and categorical variables back togather
      data = pd.concat((self.data_nonc,data_dummies),axis=1)
      del(self.data_nonc)
      return(data)
    else:
      return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # will only do this if there are categorical variables 
    if len(data.select_dtypes(include=('object')).columns) > 0:
      self.fit(data)
      # fit without target and only categorical columns
      array = self.ohe.transform(data.drop(self.target,axis=1,errors='ignore').select_dtypes(include=('object'))).toarray()
      data_dummies = pd.DataFrame(array,columns= self.data_columns)
      data_dummies.index = self.data_nonc.index
      # now put target , numerical and categorical variables back togather
      data = pd.concat((self.target_column,self.data_nonc,data_dummies),axis=1)
      # remove unwanted attributes
      del(self.target_column,self.data_nonc)
      return(data)
    else:
      return(data)

# _______________________________________________________________________________________________________________________
# Outlier
class Outlier(BaseEstimator,TransformerMixin):
  '''
    - Removes outlier using ABOD,KNN,IFO,PCA & HOBS using hard voting
    - Only takes numerical / One Hot Encoded features
  '''

  def __init__(self,target,contamination=.20, random_state=42, methods=['knn','iso','pca']):
    self.target = target
    self.contamination = contamination
    self.random_state = random_state
    self.methods = methods

    return(None)

  def fit(self,data,y=None):
    #self.abod.fit(data)
    return(None)

  def transform(self,data,y=None):
    return(data)

  def fit_transform(self,dataset,y=None):

    # dummify if there are any obects 
    if len(dataset.select_dtypes(include='object').columns)>0:
      self.dummy = Dummify(self.target)
      data = self.dummy.fit_transform(dataset)
    else:
      data = dataset.copy()

    # reduce data size for faster processing
    # try:
    #   data = data.astype('float16')
    # except:
    #   None

    if 'knn' in self.methods:
      self.knn = KNN(contamination=self.contamination)
      self.knn.fit(data.drop(self.target,axis=1))
      knn_predict = self.knn.predict(data.drop(self.target,axis=1))
      data['knn'] = knn_predict
    
    if 'iso' in self.methods:
      self.iso = IForest(contamination=self.contamination,random_state=self.random_state,behaviour='new')
      self.iso.fit(data.drop(self.target,axis=1))
      iso_predict = self.iso.predict(data.drop(self.target,axis=1))
      data['iso'] = iso_predict

    if 'pca' in self.methods:
      self.pca = PCA_od(contamination=self.contamination,random_state=self.random_state)
      self.pca.fit(data.drop(self.target,axis=1))
      pca_predict = self.pca.predict(data.drop(self.target,axis=1))
      data['pca'] = pca_predict

    data['vote_outlier'] = 0
    
    for i in self.methods:
      data['vote_outlier'] = data['vote_outlier'] + data[i]
    
  
    # data['vote_outlier'] =  data['knn']  + data['iso'] + data['pca'] 
    self.outliers = data[data['vote_outlier']== len(self.methods)]
    # self.outliers = data[data['vote_outlier']==3]
    
    return(dataset[[True if i not in self.outliers.index else False for i in dataset.index]])


#____________________________________________________________________________________________________________________________________________________________________
# Column Name cleaner transformer
class Clean_Colum_Names(BaseEstimator,TransformerMixin):
  '''
    - Cleans special chars that are not supported by jason format
  '''

  def __init__(self):
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data= dataset.copy()
    data.columns= data.columns.str.replace('[,}{\]\[\:\"\']','')
    return(data)

  def fit_transform(self,dataset,y=None):
    data= dataset.copy()
    data.columns= data.columns.str.replace('[,}{\]\[\:\"\']','')
    return(data)
#__________________________________________________________________________________________________________________________________________________________________________
# Clustering entire data
class Cluster_Entire_Data(BaseEstimator,TransformerMixin):
  '''
    - Applies kmeans clustering to the entire data set and produce clusters
    - Highly recommended to run the DataTypes_Auto_infer class first
      Args:
          target_variable: target variable (integer or numerical only)
          check_clusters_upto: to determine optimum number of kmeans clusters, set the uppler limit of clusters
  '''

  def __init__(self, target_variable, check_clusters_upto=20, random_state=42):
    self.target = target_variable
    self.check_clusters = check_clusters_upto +1
    self.random_state = random_state
    
    

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data = dataset.copy()
    # first convert to dummy
    if len(data.select_dtypes(include='object').columns)>0:
      data_t1 = self.dummy.transform(data)
    else:
      data_t1 = data.copy()

    # # # now make PLS 
    # # data_t1 = self.pls.transform(data_t1)
    # # data_t1 = self.pca.transform(data_t1)
    # # now predict with the clustes
    predict =pd.DataFrame(self.k_object.predict(data_t1),index=data.index)
    data['data_cluster'] = predict
    data['data_cluster'] = data['data_cluster'].astype('object')
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # first convert to dummy (if there are objects in data set)
    if len(data.select_dtypes(include='object').columns)>0:
      self.dummy = Dummify(self.target)
      data_t1 = self.dummy.fit_transform(data)
      data_t1 = data_t1.drop(self.target,axis=1)
    else:
      data_t1 = data.drop(self.target,axis=1).copy()
      
    
    # now make PLS 
    # self.pls = PLSRegression(n_components=len(data_t1.columns)-1)
    # data_t1 = self.pls.fit_transform(data_t1.drop(self.target,axis=1),data_t1[self.target])[0]
    # self.pca = PCA(n_components=len(data_t1.columns)-1)
    # data_t1 = self.pca.fit_transform(data_t1.drop(self.target,axis=1))
    
    # we are goign to make a place holder , for 2 to 20 clusters
    self.ph = pd.DataFrame(np.arange(2,self.check_clusters,1), columns= ['clusters']) 
    self.ph['Silhouette'] = float(0)
    self.ph['calinski'] = float(0)

    # Now start making clusters
    for k in self.ph.index:
        c =  self.ph['clusters'][k]
        self.k_object =  cluster.KMeans(n_clusters= c,init='k-means++',precompute_distances='auto',n_init=10,random_state=self.random_state)
        self.k_object.fit(data_t1)
        self.ph.iloc[k,1] = metrics.silhouette_score(data_t1,self.k_object.labels_)
        self.ph.iloc[k,2] = metrics.calinski_harabasz_score(data_t1,self.k_object.labels_)
    
    # now standardize the scores and make a total column
    m = MinMaxScaler((-1,1))
    self.ph['calinski'] = m.fit_transform(np.array(self.ph['calinski']).reshape(-1,1))
    self.ph['Silhouette'] = m.fit_transform(np.array(self.ph['Silhouette']).reshape(-1,1))
    self.ph['total']= self.ph['Silhouette'] + self.ph['calinski']
    # sort it by total column and take the first row column 0 , that would represent the optimal clusters
    try:
      self.clusters = int(self.ph[self.ph['total'] == max(self.ph['total'])]['clusters'])
    except: # in case there isnt a decisive measure , take calinski as yeard stick
      self.clusters= int(self.ph[self.ph['calinski'] == max(self.ph['calinski'])]['clusters'])
    # Now make the final cluster object
    self.k_object =  cluster.KMeans(n_clusters= self.clusters,init='k-means++',precompute_distances='auto',n_init=10,random_state=self.random_state)
    # now do fit predict
    predict =pd.DataFrame(self.k_object.fit_predict(data_t1),index=data.index)
    data['data_cluster'] = predict
    data['data_cluster'] = data['data_cluster'].astype('object')

    return(data)
#__________________________________________________________________________________________________________________________________________
# Clustering catagorical data
class Reduce_Cardinality_with_Clustering(BaseEstimator,TransformerMixin):
  '''
    - Reduces the level of catagorical column / cardinality through clustering 
    - Highly recommended to run the DataTypes_Auto_infer class first
      Args:
          target_variable: target variable (integer or numerical only)
          catagorical_feature: list of features on which clustering  is to be applied / cardinality to be reduced
          check_clusters_upto: to determine optimum number of kmeans clusters, set the uppler limit of clusters
  '''

  def __init__(self, target_variable, catagorical_feature=[], check_clusters_upto=30,random_state=42):
    self.target = target_variable
    self.feature = catagorical_feature
    self.check_clusters = check_clusters_upto + 1
    self.random= random_state
    
    

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data= dataset.copy()
    # we already know which leval belongs to whihc cluster , so all w need is to replace levels with clusters we already have from training data set
    for i,z in zip(self.feature,self.ph_data):
      data[i] = data[i].replace(list(z['levels']),z['cluster'])
    
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # first convert to dummy
    if len(data.select_dtypes(include='object').columns)>0:
      self.dummy = Dummify(self.target)
      data_t = self.dummy.fit_transform(data.drop(self.feature,axis=1))
      #data_t1 = data_t1.drop(self.target,axis=1)
    else:
      data_t = data.drop(self.feature,axis=1).copy()

    # now make PLS 
    self.pls = PLSRegression(n_components=2) # since we are only using two componenets to group #PLSRegression(n_components=len(data_t1.columns)-1)
    data_pls = self.pls.fit_transform(data_t.drop(self.target,axis=1),data_t[self.target])[0]

    # # now we will take one component and then we calculate mean, median, min, max and sd of that one component grouped by the catagorical levels
    self.ph_data = []
    self.ph_clusters = []
    for i in self.feature:
      data_t1 = pd.DataFrame(dict(levels=data[i],comp1=data_pls[:,0],comp2=data_pls[:,1]),index=data.index)
      # now group by feature
      data_t1 = data_t1.groupby('levels')
      data_t1 = data_t1[['comp1','comp2']].agg(['mean', 'median','min','max','std']) #this gives us a df with only numeric columns (min , max ) and level as index
      # some time if a level has only one record  its std will come up as NaN, so convert NaN to 1
      data_t1.fillna(1,inplace=True)
      
      # now number of clusters cant be more than the number of samples in aggregated data , so
      self.check_clusters = min (self.check_clusters,len(data_t1)) 

      # # we are goign to make a place holder , for 2 to 20 clusters
      self.ph = pd.DataFrame(np.arange(2,self.check_clusters,1), columns= ['clusters']) 
      self.ph['Silhouette'] = float(0)
      self.ph['calinski'] = float(0)

    # Now start making clusters
      for k in self.ph.index:
          c =  self.ph['clusters'][k]
          self.k_object =  cluster.KMeans(n_clusters= c,init='k-means++',precompute_distances='auto',n_init=10,random_state=self.random)
          self.k_object.fit(data_t1)
          self.ph.iloc[k,1] = metrics.silhouette_score(data_t1,self.k_object.labels_)
          self.ph.iloc[k,2] = metrics.calinski_harabasz_score(data_t1,self.k_object.labels_)
      
      # now standardize the scores and make a total column
      m = MinMaxScaler((-1,1))
      self.ph['calinski'] = m.fit_transform(np.array(self.ph['calinski']).reshape(-1,1))
      self.ph['Silhouette'] = m.fit_transform(np.array(self.ph['Silhouette']).reshape(-1,1))
      self.ph['total']= self.ph['Silhouette'] + self.ph['calinski']
      # sort it by total column and take the first row column 0 , that would represent the optimal clusters
      try:
        self.clusters = int(self.ph[self.ph['total'] == max(self.ph['total'])]['clusters'])
      except: # in case there isnt a decisive measure , take calinski as yeard stick
        self.clusters= int(self.ph[self.ph['calinski'] == max(self.ph['calinski'])]['clusters'])
      self.ph_clusters.append(self.ph)
      # Now make the final cluster object
      self.k_object =  cluster.KMeans(n_clusters= self.clusters,init='k-means++',precompute_distances='auto',n_init=10,random_state=self.random)
      # now do fit predict
      predict =self.k_object.fit_predict(data_t1)
      # put it back with the group by aggregate columns
      data_t1['cluster'] = predict
      data_t1['cluster'] = data_t1['cluster'] .apply(str)
      # now we dont need all the columns, only the cluster column is required along with the index (index also has a name , we  groupy as "levels")
      data_t1 = data_t1[['cluster']]
      # now convert index ot the column
      data_t1.reset_index(level=0, inplace=True) # this table now only contains every level and its cluster
      #self.data_t1= data_t1
      # we can now replace cluster with the original level in the original data frame
      data[i] = data[i].replace(list(data_t1['levels']),data_t1['cluster'])
      self.ph_data.append(data_t1)

    return(data)

#____________________________________________________________________________________________________________________________________________
# Clustering catagorical data
class Reduce_Cardinality_with_Counts(BaseEstimator,TransformerMixin):
  '''
    - Reduces the level of catagorical column by replacing levels with their count & converting objects into float
      Args:
          catagorical_feature: list of features on which clustering is to be applied
  '''

  def __init__(self, catagorical_feature=[]):
    self.feature = catagorical_feature    

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    data= dataset.copy()
    # we already know level counts 
    for i,z,k in zip(self.feature,self.ph_data,self.ph_u):
      data[i] = data[i].replace(k,z['counts'])
      data[i] = data[i].astype('float64')
    
    return(data)

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()
    # 
    self.ph_data = []
    self.ph_u= []
    for i in self.feature:
      data_t1 = pd.DataFrame(dict(levels=data[i].groupby(data[i], sort=False).count().index,counts =data[i].groupby(data[i], sort=False).count().values))
      u = data[i].unique()
      # replace levels with counts
      data[i].replace(u,data_t1['counts'],inplace=True)
      data[i] = data[i].astype('float64')
      self.ph_data.append(data_t1)
      self.ph_u.append(u)
    
    return(data)
#____________________________________________________________________________________________________________________________________________
# take noneliner transformations
class Make_NonLiner_Features(BaseEstimator,TransformerMixin):
  '''
    - convert numerical features into polynomial features
    - it is HIGHLY recommended to run the Autoinfer_Data_Type class first
    - Ignores target variable
    - it picks up data type float64 as numerical 
    - for multiclass classification problem , set subclass arg to 'multi'

      Args: 
        target: string , name of the target variable
        Polynomial_degree: int ,default 2  
  '''

  def __init__(self,target,ml_usecase= 'classification',Polynomial_degree=2,other_nonliner_features= ['sin','cos','tan'],top_features_to_pick=.20,random_state=42,subclass='ignore'):
    self.target = target
    self.Polynomial_degree = Polynomial_degree
    self.ml_usecase = ml_usecase
    self.other_nonliner_features = other_nonliner_features
    self.top_features_to_pick = top_features_to_pick
    self.random_state = random_state
    self.subclass = subclass
  
  def fit(self,data,y=None):
    # nothing to learn
      return(None)
 
  def transform(self,dataset,y=None):# same application for test and train
    data = dataset.copy()
    
    self.numeric_columns = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include="float64").columns
    if self.Polynomial_degree >= 2: # dont run anything if powr is les than 2
      #self.numeric_columns = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include="float64").columns
      # start taking powers
      for i in range(2,self.Polynomial_degree+1):
        ddc_power = np.power(data[self.numeric_columns],i)
        ddc_col = list(ddc_power.columns)
        ii = str(i)
        ddc_col = [ddc_col + '_Power'+ ii for ddc_col in ddc_col]
        ddc_power.columns = ddc_col
        #put it back with data dummy
        #data = pd.concat((data,ddc_power),axis=1) 
    else:
       ddc_power = pd.DataFrame()
    
    # take sin:
    if 'sin' in self.other_nonliner_features:
      ddc_sin = np.sin(data[self.numeric_columns])
      ddc_col = list(ddc_sin.columns)
      ddc_col = ["sin("+ i +")" for i in ddc_col]
      ddc_sin.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_sin),axis=1) 
    else:
      ddc_sin = pd.DataFrame()


    # take cos:
    if 'cos' in self.other_nonliner_features:
      ddc_cos = np.cos(data[self.numeric_columns])
      ddc_col = list(ddc_cos.columns)
      ddc_col = ["cos("+ i +")" for i in ddc_col]
      ddc_cos.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_cos),axis=1)
    else:
       ddc_cos = pd.DataFrame()

    # take tan:
    if 'tan' in self.other_nonliner_features:
      ddc_tan = np.tan(data[self.numeric_columns])
      ddc_col = list(ddc_tan.columns)
      ddc_col = ["tan("+ i +")" for i in ddc_col]
      ddc_tan.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_tan),axis=1) 
    else:
       ddc_tan = pd.DataFrame()      
  
    #dummy_all
    dummy_all= pd.concat((data,ddc_power,ddc_sin,ddc_cos,ddc_tan),axis=1)
    # we can select top features using RF
    # # and we only want to do this if the dummy all have more than 50 features 
    # if len(dummy_all.columns) > 71:

  
    return(dummy_all[self.columns_to_keep])


  def fit_transform(self,dataset,y=None):
    
    data = dataset.copy()
    
    self.numeric_columns = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include="float64").columns
    if self.Polynomial_degree >= 2: # dont run anything if powr is les than 2
      #self.numeric_columns = data.drop(self.target,axis=1,errors='ignore').select_dtypes(include="float64").columns
      # start taking powers
      for i in range(2,self.Polynomial_degree+1):
        ddc_power = np.power(data[self.numeric_columns],i)
        ddc_col = list(ddc_power.columns)
        ii = str(i)
        ddc_col = [ddc_col + '_Power'+ ii for ddc_col in ddc_col]
        ddc_power.columns = ddc_col
        #put it back with data dummy
        #data = pd.concat((data,ddc_power),axis=1) 
    else:
       ddc_power = pd.DataFrame()
    
    # take sin:
    if 'sin' in self.other_nonliner_features:
      ddc_sin = np.sin(data[self.numeric_columns])
      ddc_col = list(ddc_sin.columns)
      ddc_col = ["sin("+ i +")" for i in ddc_col]
      ddc_sin.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_sin),axis=1) 
    else:
      ddc_sin = pd.DataFrame()


    # take cos:
    if 'cos' in self.other_nonliner_features:
      ddc_cos = np.cos(data[self.numeric_columns])
      ddc_col = list(ddc_cos.columns)
      ddc_col = ["cos("+ i +")" for i in ddc_col]
      ddc_cos.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_cos),axis=1)
    else:
       ddc_cos = pd.DataFrame()

    # take tan:
    if 'tan' in self.other_nonliner_features:
      ddc_tan = np.tan(data[self.numeric_columns])
      ddc_col = list(ddc_tan.columns)
      ddc_col = ["tan("+ i +")" for i in ddc_col]
      ddc_tan.columns = ddc_col
      #put it back with data dummy
      #data = pd.concat((data,ddc_tan),axis=1) 
    else:
       ddc_tan = pd.DataFrame()      
  
    #dummy_all
    dummy_all= pd.concat((data[[self.target]],ddc_power,ddc_sin,ddc_cos,ddc_tan),axis=1)
    # we can select top features using our Feature Selection Classic transformer
    afs = Advanced_Feature_Selection_Classic(target=self.target,ml_usecase=self.ml_usecase,top_features_to_pick=self.top_features_to_pick,random_state=self.random_state,subclass=self.subclass)
    dummy_all_t = afs.fit_transform(dummy_all)

    data= pd.concat((data,dummy_all_t),axis=1)
     # # making sure no duplicated columns are there
    data = data.loc[:,~data.columns.duplicated()] 
    self.columns_to_keep = data.drop(self.target,axis=1).columns 

    return(data)

#______________________________________________________________________________________________________________________________________________________
#Feature Selection
class Advanced_Feature_Selection_Classic(BaseEstimator,TransformerMixin):
  '''
    - Selects important features and reduces the feature space. Feature selection is based on Random Forest , Light GBM and Correlation
    - to run on multiclass classification , set the subclass argument to 'multi'
  '''

  def __init__(self,target,ml_usecase='classification',top_features_to_pick=.10, random_state=42, subclass='ignore'):
    self.target= target
    self.ml_usecase = ml_usecase
    self.top_features_to_pick =1-top_features_to_pick
    self.random_state = random_state
    self.subclass = subclass

    return(None)

  def fit(self,dataset,y=None):
    return(None)

  def transform(self,dataset,y=None):
    # return the data with onlys specific columns
    data= dataset.copy()
    # self.selected_columns.remove(self.target)
    return(data[self.selected_columns_test])
    # return(data)

  def fit_transform(self,dataset,y=None):
    
    dummy_all = dataset.copy()

    # Random Forest
    max_fe = min(70, int(np.sqrt(len(dummy_all.columns))))
    max_sa = min(1000, int(np.sqrt(len(dummy_all))))

    if self.ml_usecase == 'classification':
      m = rfc(100,max_depth=5,max_features=max_fe,n_jobs=-1,max_samples=max_sa,random_state=self.random_state)
    else:
      m = rfr(100,max_depth=5,max_features=max_fe,n_jobs=-1,max_samples=max_sa,random_state=self.random_state)

    m.fit(dummy_all.drop(self.target,axis=1),dummy_all[self.target])
    # self.fe_imp_table= pd.DataFrame(m.feature_importances_,columns=['Importance'],index=dummy_all.drop(self.target,axis=1).columns).sort_values(by='Importance',ascending= False)
    self.fe_imp_table= pd.DataFrame(m.feature_importances_,columns=['Importance'],index=dummy_all.drop(self.target,axis=1).columns)
    self.fe_imp_table = self.fe_imp_table[self.fe_imp_table['Importance']>= self.fe_imp_table.quantile(self.top_features_to_pick)[0]]
    top = self.fe_imp_table.index
    dummy_all_columns_RF= dummy_all[top].columns


    #LightGBM
    max_fe = min(70, int(np.sqrt(len(dummy_all.columns))))
    max_sa = min(float(1000/len(dummy_all)), float(np.sqrt(len(dummy_all)/len(dummy_all))))

    if self.ml_usecase == 'classification':
      m = lgbmc(n_estimators=100,max_depth=5,n_jobs=-1,subsample=max_sa,random_state=self.random_state)
    else:
      m = lgbmr(n_estimators=100,max_depth=5,n_jobs=-1,subsample=max_sa,random_state=self.random_state)
    m.fit(dummy_all.drop(self.target,axis=1),dummy_all[self.target])
    # self.fe_imp_table= pd.DataFrame(m.feature_importances_,columns=['Importance'],index=dummy_all.drop(self.target,axis=1).columns).sort_values(by='Importance',ascending= False)
    self.fe_imp_table= pd.DataFrame(m.feature_importances_,columns=['Importance'],index=dummy_all.drop(self.target,axis=1).columns)
    self.fe_imp_table = self.fe_imp_table[self.fe_imp_table['Importance']>= self.fe_imp_table.quantile(self.top_features_to_pick)[0]]
    top = self.fe_imp_table.index
    dummy_all_columns_LGBM= dummy_all[top].columns


    # we can now select top correlated feature
    if self.subclass != 'multi':
      corr = pd.DataFrame(np.corrcoef(dummy_all.T))
      corr.columns = dummy_all.columns
      corr.index = dummy_all.columns
      # corr = corr[self.target].abs().sort_values(ascending=False)[0:self.top_features_to_pick+1]
      corr = corr[self.target].abs()
      corr = corr[corr.index!=self.target] # drop the target column
      corr = corr[corr >= corr.quantile(self.top_features_to_pick)]
      corr = pd.DataFrame(dict(features=corr.index,value=corr)).reset_index(drop=True)
      corr = corr.drop_duplicates(subset='value')
      corr = corr['features']
      # corr = pd.DataFrame(dict(features=corr.index,value=corr)).reset_index(drop=True)
      # corr = corr.drop_duplicates(subset='value')[0:self.top_features_to_pick+1]
      # corr = corr['features']
    else:
      corr = list()

    self.dummy_all_columns_RF = dummy_all_columns_RF
    self.dummy_all_columns_LGBM = dummy_all_columns_LGBM
    self.corr = corr

    self.selected_columns = list(set([self.target]+list(dummy_all_columns_RF) + list(corr) +list(dummy_all_columns_LGBM)))
    
    self.selected_columns_test = dataset[self.selected_columns].drop(self.target,axis=1).columns
    return(dataset[self.selected_columns])
#_

#______________________________________________________________________________________________________________________________________________________
#Boruta Feature Selection algorithm
# Base on: https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py
class Boruta_Feature_Selection(BaseEstimator, TransformerMixin):
  """
          Boruta selection algorithm base on s borutaPy klearn-contrib and
          Miron B Kursa, https://m2.icm.edu.pl/boruta/
          Takes features and select the most important
            Args:
              target (str): target column name
              ml_usecase (str): case: classification or regression
              top_features_to_pick: to make...
              max_iteration {int): overall iterations of shuffle and train forests 
              alpha {float): p-value on which 
              the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
              A value of .5 means both measure have equal say 
  """
  def __init__(self,target,ml_usecase='classification',top_features_to_pick=.10,
               max_iteration = 25, alpha=0.05, percentile=65,
               random_state=42, subclass='ignore'):
    self.target= target
    self.ml_usecase = ml_usecase
    self.top_features_to_pick =1-top_features_to_pick
    self.random_state = random_state
    self.subclass = subclass
    self.max_iteration = max_iteration
    self.alpha = alpha
    self.percentile = percentile

    return(None)


  def fit(self,dataset,y=None):
    return(None)


  def transform(self,dataset,y=None):
    # return the data with columns which match the threshold
    data = dataset.copy()
    # self.selected_columns.remove(self.target)
    return(data[self.selected_columns_test])


  def fit_transform(self, dataset, y=None):
    dummy_data = dataset.copy()
    X, y = dummy_data.drop(self.target, axis=1), dummy_data[self.target]
    n_sample, n_feat = X.shape
    _iter = 1
    # holds the decision about each feature:
    # 0  - default state = tentative in original code
    # 1  - accepted in original code
    # -1 - rejected in original code
    dec_reg = np.zeros(n_feat, dtype=np.int)
    # counts how many times a given feature was more important than
    # the best of the shadow features
    shadow_max = list()
    hits = np.zeros(n_feat, dtype=np.int)
    tent_hits = np.zeros(n_feat)
    while np.any(dec_reg == 0) and _iter < self.max_iteration:
      # get tentative features
      x_ind = self._get_idx(X, dec_reg)
      X_tent = X.iloc[:, x_ind].copy()
      X_boruta = X_tent.copy()
      # create boruta features
      for col in X_tent.columns:
        X_boruta["shadow_{}".format(col)] = np.random.permutation(X_tent[col])
      # train imputator
      feat_imp_X, feat_imp_shadow = self._inputator(X_boruta, X_tent, y, dec_reg)
      # add shadow percentile to history
      thresh = np.percentile(feat_imp_shadow, self.percentile)
      shadow_max.append(thresh)
      # confirm hits
      cur_imp_no_nan = feat_imp_X
      cur_imp_no_nan[np.isnan(cur_imp_no_nan)] = 0
      h_ = np.where(cur_imp_no_nan > thresh)[0]
      hits[h_] += 1
      # add importance to tentative hits
      tent_hits[x_ind] += feat_imp_X
      # do statistical testsa
      dec_reg = self._do_tests(dec_reg, hits, _iter)
      if _iter < self.max_iteration:
        _iter += 1
        
    # fix tentative onse if exist
    print(dec_reg)
    confirmed = np.where(dec_reg == 1)[0]
    
    tentative = np.where(dec_reg == 0)[0]
    if len(tentative) == 0:
      confirmed_cols = X.columns[confirmed]
    else:
      median_tent = np.median(tent_hits[tentative])
      tentative_confirmed = np.where(median_tent > np.median(shadow_max))[0]
      tentative = tentative[tentative_confirmed]
      confirmed_cols = X.columns[np.concatenate((confirmed, tentative), axis=0)]
    
    self.confirmed_cols = confirmed_cols.tolist()
    self.confirmed_cols.append(self.target)
    
    self.selected_columns_test = dataset[self.confirmed_cols].drop(self.target,axis=1).columns

    return dataset[self.confirmed_cols]

  def _get_idx(self, X, dec_reg):
    x_ind = np.where(dec_reg >= 0)[0]
    # be sure that dataset have more than 5 columns
    if len(x_ind) < 5 and X.shape[1] > 5:
      additional = [i for i in range(X.shape[1]) if i not in x_ind]
      length = 6 - len(x_ind)
      x_ind = np.concatenate((x_ind, np.random.choice(additional, length, replace=False)))
      return x_ind
    elif len(x_ind) < 5 and X.shape[1] < 5:
      return x_ind
    else:
      return x_ind


  def _inputator(self, X_boruta, X, y, dec_reg):
    feat_imp_X = feat_imp_shadow = np.zeros(X.shape[1])
    # Random Forest
    max_fe = min(70, int(np.sqrt(len(X.columns))))
    max_sa = min(1000, int(np.sqrt(len(X))))
    if self.ml_usecase == 'classification':
      m = lgbmc(n_estimators=100,max_depth=5,n_jobs=-1,subsample=max_sa,
                bagging_fraction=0.99, random_state=self.random_state)
    else:
      m = lgbmr(n_estimators=100,max_depth=5,n_jobs=-1,subsample=max_sa,
                bagging_fraction=0.99, random_state=self.random_state)
    
    m.fit(X_boruta, y)
    ### store feature importance
    feat_imp_X = m.feature_importances_[:len(X.columns)]
    feat_imp_shadow = m.feature_importances_[len(X.columns):]
      
    return feat_imp_X, feat_imp_shadow
    
  def _do_tests(self, dec_reg, hit_reg, _iter):
    active_features = np.where(dec_reg >= 0)[0]
    hits = hit_reg[active_features]
    # get uncorrected p values based on hit_reg
    to_accept_ps = stats.binom.sf(hits - 1, _iter, .5).flatten()
    to_reject_ps = stats.binom.cdf(hits, _iter, .5).flatten()

    # as in th original Boruta, we simply do bonferroni correction
    # with the total n_feat in each iteration
    to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
    to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))
  
    # find features which are 0 and have been rejected or accepted
    to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
    to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]
  
    # updating dec_reg
    dec_reg[active_features[to_accept]] = 1
    dec_reg[active_features[to_reject]] = -1
    return dec_reg


#_________________________________________________________________________________________________________________________________________
class Fix_multicollinearity(BaseEstimator,TransformerMixin):
  """
          Fixes multicollinearity between predictor variables , also considering the correlation between target variable.
          Only applies to regression or two class classification ML use case
          Takes numerical and one hot encoded variables only
            Args:
              threshold (float): The utmost absolute pearson correlation tolerated beyween featres from 0.0 to 1.0
              target_variable (str): The target variable/column name
              correlation_with_target_threshold: minimum absolute correlation required between every feature and the target variable , default 1.0 (0.0 to 1.0)
              correlation_with_target_preference: float (0.0 to 1.0), default .08 ,while choosing between a pair of features w.r.t multicol & correlation target , this gives 
              the option to favour one measur to another. e.g. if value is .6 , during feature selection tug of war, correlation target measure will have a higher say.
              A value of .5 means both measure have equal say 
  """
  # mamke a constructer
  
  def __init__ (self,threshold,target_variable,correlation_with_target_threshold= 0.0,correlation_with_target_preference=1.0):
      self.threshold = threshold
      self.target_variable = target_variable
      self.correlation_with_target_threshold= correlation_with_target_threshold
      self.target_corr_weight = correlation_with_target_preference
      self.multicol_weight = 1-correlation_with_target_preference
  
  # Make fit method
  
  def fit (self,data,y=None):
    '''
        Args:
            data = takes preprocessed data frame
        Returns:
            None
    '''
      
    #global data1
    self.data1 = data.copy()
    # try:
    #   self.data1 = self.data1.astype('float16')
    # except:
    #   None
    # make an correlation db with abs correlation db
    # self.data_c = self.data1.T.drop_duplicates()
    # self.data1 = self.data_c.T
    corr = pd.DataFrame(np.corrcoef(self.data1.T))
    corr.columns = self.data1.columns
    corr.index = self.data1.columns
    # self.corr_matrix = abs(self.data1.corr())
    self.corr_matrix = abs(corr)

    # for every diagonal value, make it Nan
    self.corr_matrix.values[tuple([np.arange(self.corr_matrix.shape[0])]*2)] = np.NaN
    
    # Now Calculate the average correlation of every feature with other, and get a pandas data frame
    self.avg_cor = pd.DataFrame(self.corr_matrix.mean())
    self.avg_cor['feature']= self.avg_cor.index
    self.avg_cor.reset_index(drop=True, inplace=True)
    self.avg_cor.columns =  ['avg_cor','features']
    
    # Calculate the correlation with the target
    self.targ_cor = pd.DataFrame(self.corr_matrix[self.target_variable].dropna())
    self.targ_cor['feature']= self.targ_cor.index
    self.targ_cor.reset_index(drop=True, inplace=True)
    self.targ_cor.columns =  ['target_variable','features']
    
    # Now, add a column for variable name and drop index
    self.corr_matrix['column'] = self.corr_matrix.index
    self.corr_matrix.reset_index(drop=True,inplace=True)
    
    # now we need to melt it , so that we can correlation pair wise , with two columns 
    self.cols =self.corr_matrix.column
    self.melt = self.corr_matrix.melt(id_vars= ['column'],value_vars=self.cols).sort_values(by='value',ascending=False).dropna()

    # now bring in the avg correlation for first of the pair
    self.merge = pd.merge(self.melt,self.avg_cor,left_on='column',right_on='features').drop('features',axis=1)

    # now bring in the avg correlation for second of the pair
    self.merge = pd.merge(self.merge,self.avg_cor,left_on='variable',right_on='features').drop('features',axis=1)

    # now bring in the target correlation for first of the pair
    self.merge = pd.merge(self.merge,self.targ_cor,left_on='column',right_on='features').drop('features',axis=1)

    # now bring in the avg correlation for second of the pair
    self.merge = pd.merge(self.merge,self.targ_cor,left_on='variable',right_on='features').drop('features',axis=1)

    # sort and save
    self.merge = self.merge.sort_values(by='value',ascending=False)

    # we need to now eleminate all the pairs that are actually duplicate e.g cor(x,y) = cor(y,x) , they are the same , we need to find these and drop them
    self.merge['all_columns'] = self.merge['column'] + self.merge['variable']

    # this puts all the coresponding pairs of features togather , so that we can only take one, since they are just the duplicates
    self.merge['all_columns'] = [sorted(i) for i in self.merge['all_columns'] ]

    # now sort by new column
    self.merge = self.merge.sort_values(by='all_columns')

    # take every second colums
    self.merge = self.merge.iloc[::2, :]

    # make a ranking column to eliminate features
    self.merge['rank_x'] = round(self.multicol_weight*(self.merge['avg_cor_y']- self.merge['avg_cor_x']) +  self.target_corr_weight*(self.merge['target_variable_x'] - self.merge['target_variable_y']),6) # round it to 6 digits
    self.merge1 = self.merge # delete here
    ## Now there will be rows where the rank will be exactly zero, these is where the value (corelartion between features) is exactly one ( like price and price^2)
    ## so in that case , we can simply pick one of the variable
    # but since , features can be in either column, we will drop one column (say 'column') , only if the feature is not in the second column (in variable column)
    # both equations below will return the list of columns to drop from here 
    # this is how it goes

    ## For the portion where correlation is exactly one !
    self.one = self.merge[self.merge['rank_x']==0]

    # this portion is complicated 
    # table one have all the paired variable having corelation of 1
    # in a nutshell, we can take any column (one side of pair) and delete the other columns (other side of the pair)
    # however one varibale can appear more than once on any of the sides , so we will run for loop to find all pairs...
    # here it goes
    # take a list of all (but unique ) variables that have correlation 1 for eachother, we will make two copies
    self.u_all = list(pd.unique(pd.concat((self.one['column'],self.one['variable']),axis=0)))
    self.u_all_1 = list(pd.unique(pd.concat((self.one['column'],self.one['variable']),axis=0)))
    # take a list of features (unique) for the first side of the pair
    self.u_column  = pd.unique(self.one['column'])
    
    # now we are going to start picking each variable from one column (one side of the pair) , check it against the other column (other side of the pair)
    # to pull all coresponding / paired variables  , and delete thoes newly varibale names from all unique list
    
    for i in self.u_column:
      #print(i)
      r = self.one[self.one['column']==i]['variable'].values
      for q in r:
        if q in self.u_all:
          #print("_"+q)
          self.u_all.remove(q)

    # now the unique column contains the varibales that should remain, so in order to get the variables that should be deleted :
    self.to_drop =(list(set(self.u_all_1)-set(self.u_all)))


    # self.to_drop_a =(list(set(self.one['column'])-set(self.one['variable'])))
    # self.to_drop_b =(list(set(self.one['variable'])-set(self.one['column'])))
    # self.to_drop = self.to_drop_a + self.to_drop_b

    ## now we are to treat where rank is not Zero and Value (correlation) is greater than a specific threshold
    self.non_zero = self.merge[(self.merge['rank_x']!= 0.0) & (self.merge['value'] >= self.threshold)]

    # pick the column to delete
    self.non_zero_list = list(np.where(self.non_zero['rank_x'] < 0 , self.non_zero['column'], self.non_zero['variable']))

    # add two list
    self.to_drop = self.to_drop + self.non_zero_list

    #make sure that target column is not a part of the list
    try:
        self.to_drop.remove(self.target_variable)
    except:
        self.to_drop
    
    self.to_drop = self.to_drop

    # now we want to keep only the columns that have more correlation with traget by a threshold
    self.to_drop_taret_correlation=[] 
    if self.correlation_with_target_threshold != 0.0:
      corr = pd.DataFrame(np.corrcoef(data.drop(self.to_drop,axis=1).T),columns= data.drop(self.to_drop,axis=1).columns,index=data.drop(self.to_drop,axis=1).columns)
      self.to_drop_taret_correlation = corr[self.target_variable].abs()
      # self.to_drop_taret_correlation = data.drop(self.to_drop,axis=1).corr()[self.target_variable].abs()
      self.to_drop_taret_correlation = self.to_drop_taret_correlation [self.to_drop_taret_correlation < self.correlation_with_target_threshold ]
      self.to_drop_taret_correlation = list(self.to_drop_taret_correlation.index)
      #self.to_drop = self.corr + self.to_drop
      try:
        self.to_drop_taret_correlation.remove(self.target_variable)
      except:
        self.to_drop_taret_correlation
      

  # now Transform
  def transform(self,dataset,y=None):
    '''
        Args:f
            data = takes preprocessed data frame
        Returns:
            data frame
    '''
    data = dataset.copy()
    data.drop(self.to_drop,axis=1,inplace=True)
    # now drop less correlated data
    data.drop(self.to_drop_taret_correlation,axis=1,inplace=True,errors='ignore')
    return(data)
  
  # fit_transform
  def fit_transform(self,data, y=None):
    
    '''
        Args:
            data = takes preprocessed data frame
        Returns:
            data frame
    '''
    self.fit(data)
    return(self.transform(data))

#____________________________________________________________________________________________________________________________________________________________________
# handle perfect multicollinearity 
class Remove_100(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return data frame while removing features that are perfectly correlated (droping one)
  '''

  def __init__(self,target):
    self.target = target
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    return(dataset.drop(self.columns_to_drop,axis=1))

  def fit_transform(self,dataset,y=None):
    data = dataset.copy()

    targetless_data = data.drop(self.target, axis=1)

    # correlation should be calculated between at least two features, if there is only 1, there is nothing to delete
    if len(targetless_data.columns) <= 1:
      return data

    corr = pd.DataFrame(np.corrcoef(targetless_data.T))
    corr.columns = targetless_data.columns
    corr.index = targetless_data.columns
    corr_matrix = abs(corr)

    # Now, add a column for variable name and drop index
    corr_matrix['column'] = corr_matrix.index
    corr_matrix.reset_index(drop=True,inplace=True)

    # now we need to melt it , so that we can correlation pair wise , with two columns 
    cols =corr_matrix.column
    melt = corr_matrix.melt(id_vars= ['column'],value_vars=cols).sort_values(by='value',ascending=False)#.dropna()
    melt['value'] = round(melt['value'],2) # round it to two digits

    # now pick variables where value is one and 'column' != variabe ( both columns are not same)
    c1 = melt['value'] == 1.00
    c2 = melt['column'] != melt['variable']
    melt = melt[((c1 == True) & (c2 == True)) ]

    # we need to now eleminate all the pairs that are actually duplicate e.g cor(x,y) = cor(y,x) , they are the same , we need to find these and drop them
    melt['all_columns'] = melt['column'] + melt['variable']

    # this puts all the coresponding pairs of features togather , so that we can only take one, since they are just the duplicates
    melt['all_columns'] = [sorted(i) for i in melt['all_columns'] ]

    # # now sort by new column
    melt = melt.sort_values(by='all_columns')

    # # take every second colums
    melt = melt.iloc[::2, :]

    # lets keep the columns on the left hand side of the table
    self.columns_to_drop = melt['variable']

    return(data.drop(self.columns_to_drop,axis=1))

#_______________________________________________________________________________________________________________________________________________________________________________________________
# custome DFS 
class DFS_Classic(BaseEstimator,TransformerMixin):
  '''
    - Automated feature interactions using multiplication, division , addition & substraction
    - Only accepts numeric / One Hot Encoded features
    - Takes DF, return same DF 
    - for Multiclass classification problem , set subclass arg as 'multi'
  '''

  def __init__(self,target,ml_usecase='classification',interactions=['multiply','divide','add','subtract'],top_features_to_pick_percentage=.05, random_state=42,subclass='ignore'):
    self.target= target
    self.interactions = interactions
    self.top_n_correlated = top_features_to_pick_percentage #(this will be 1- top_features , but handled in the Advance_feature_selection )
    self.ml_usecase = ml_usecase
    self.random_state = random_state
    self.subclass = subclass
    

    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):

    data= dataset.copy()
     # for multiplication:
    # we need bot catagorical and numerical columns

    if 'multiply' in self.interactions:
      
      data_multiply = pd.concat([data.drop(self.target,axis=1,errors='ignore').mul(col[1], axis="index") for col in data.drop(self.target,axis=1,errors='ignore').iteritems()], axis=1)
      data_multiply.columns = ["_multiply_".join([i, j]) for j in data.drop(self.target,axis=1,errors='ignore').columns for i in data.drop(self.target,axis=1,errors='ignore').columns]
      # we dont need to apply rest of conditions
      data_multiply.index = data.index 
    else:
      data_multiply= pd.DataFrame()
    
     # for division, we only want it to apply to numerical columns
    if 'divide' in self.interactions:
      
      data_divide = pd.concat([data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].div(col[1], axis="index") for col in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].iteritems()], axis=1)
      data_divide.columns = ["_divide_".join([i, j]) for j in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns for i in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns]
      data_divide.replace([np.inf,-np.inf],0,inplace=True)
      data_divide.fillna(0,inplace=True)
      data_divide.index = data.index
    else:
      data_divide= pd.DataFrame()
    
     # for addition, we only want it to apply to numerical columns
    if 'add' in self.interactions:

      data_add = pd.concat([data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].add(col[1], axis="index") for col in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].iteritems()], axis=1)
      data_add.columns = ["_add_".join([i, j]) for j in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns for i in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns]
      data_add.index = data.index
    else:
      data_add= pd.DataFrame()
    
    # for substraction, we only want it to apply to numerical columns
    if 'subtract' in self.interactions:
      
      data_substract = pd.concat([data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].sub(col[1], axis="index") for col in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].iteritems()], axis=1)
      data_substract.columns = ["_subtract_".join([i, j]) for j in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns for i in data.drop(self.target,axis=1,errors='ignore')[self.numeric_columns].columns]
      data_substract.index = data.index
    else:
      data_substract= pd.DataFrame()

    # get all the dummy data combined
    dummy_all = pd.concat((data,data_multiply,data_divide,data_add,data_substract),axis=1)
    del(data_multiply)
    del(data_divide)
    del(data_add)
    del(data_substract)
    # now only return the columns we want:
    return(dummy_all[self.columns_to_keep])

    

  def fit_transform(self,dataset,y=None):

    data= dataset.copy()

    # we need to seperate numerical and ont hot encoded columns
    # self.ohe_columns = [i if ((len(data[i].unique())==2) & (data[i].unique()[0] in [0,1]) & (data[i].unique()[1] in [0,1]) ) else None for i in data.drop(self.target,axis=1).columns]
    self.ohe_columns =[i for i in data.columns if len(data[i].unique()) == 2 and data[i].unique()[0] in [0,1] and data[i].unique()[1] in [0,1]]
    # self.ohe_columns = [i for i in self.ohe_columns if i is not None]
    self.numeric_columns = [i for i in data.drop(self.target,axis=1).columns if i not in self.ohe_columns]
    target_variable = data[[self.target]]

    # for multiplication:
    # we need bot catagorical and numerical columns

    if 'multiply' in self.interactions:
      data_multiply = pd.concat([data.drop(self.target,axis=1).mul(col[1], axis="index") for col in data.drop(self.target,axis=1).iteritems()], axis=1)
      data_multiply.columns = ["_multiply_".join([i, j]) for j in data.drop(self.target,axis=1).columns for i in data.drop(self.target,axis=1).columns]
      # we dont need columns that are self interacted
      col = ["_multiply_".join([i, j]) for j in data.drop(self.target,axis=1).columns for i in data.drop(self.target,axis=1).columns if i!=j]
      data_multiply = data_multiply[col]
      # we dont need columns where the sum of the total column is null (to catagorical variables never happening togather)
      col1 = [i for i in data_multiply.columns if np.nansum(data_multiply[i])!=0 ]
      data_multiply = data_multiply[col1]
      data_multiply.index = data.index
    else:
      data_multiply= pd.DataFrame()

    # for division, we only want it to apply to numerical columns
    if 'divide' in self.interactions:
      data_divide = pd.concat([data.drop(self.target,axis=1)[self.numeric_columns].div(col[1], axis="index") for col in data.drop(self.target,axis=1)[self.numeric_columns].iteritems()], axis=1)
      data_divide.columns = ["_divide_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns]
      # we dont need columns that are self interacted
      col = ["_divide_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns if i!=j]
      data_divide = data_divide[col]
      # we dont need columns where the sum of the total column is null (to catagorical variables never happening togather)
      col1 = [i for i in data_divide.columns if np.nansum(data_divide[i])!=0 ]
      data_divide = data_divide[col1]
      # additionally we need to fill anll the possible NaNs
      data_divide.replace([np.inf,-np.inf],0,inplace=True)
      data_divide.fillna(0,inplace=True)
      data_divide.index = data.index
    else:
      data_divide= pd.DataFrame()

    # for addition, we only want it to apply to numerical columns
    if 'add' in self.interactions:
      data_add = pd.concat([data.drop(self.target,axis=1)[self.numeric_columns].add(col[1], axis="index") for col in data.drop(self.target,axis=1)[self.numeric_columns].iteritems()], axis=1)
      data_add.columns = ["_add_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns]
      # we dont need columns that are self interacted
      col = ["_add_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns if i!=j]
      data_add = data_add[col]
      # we dont need columns where the sum of the total column is null (to catagorical variables never happening togather)
      col1 = [i for i in data_add.columns if np.nansum(data_add[i])!=0 ]
      data_add = data_add[col1]
      data_add.index = data.index
    else:
      data_add= pd.DataFrame()

    # for substraction, we only want it to apply to numerical columns
    if 'subtract' in self.interactions:
      data_substract = pd.concat([data.drop(self.target,axis=1)[self.numeric_columns].sub(col[1], axis="index") for col in data.drop(self.target,axis=1)[self.numeric_columns].iteritems()], axis=1)
      data_substract.columns = ["_subtract_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns]
      # we dont need columns that are self interacted
      col = ["_subtract_".join([i, j]) for j in data.drop(self.target,axis=1)[self.numeric_columns].columns for i in data.drop(self.target,axis=1)[self.numeric_columns].columns if i!=j]
      data_substract = data_substract[col]
      # we dont need columns where the sum of the total column is null (to catagorical variables never happening togather)
      col1 = [i for i in data_substract.columns if np.nansum(data_substract[i])!=0 ]
      data_substract = data_substract[col1]
      data_substract.index = data.index
    else:
      data_substract= pd.DataFrame()

    # get all the dummy data combined
    dummy_all = pd.concat((data_multiply,data_divide,data_add,data_substract),axis=1)
    del(data_multiply)
    del(data_divide)
    del(data_add)
    del(data_substract)

    dummy_all[self.target] = target_variable
    self.dummy_all = dummy_all

  
    # apply advanced feature selectio 
    afs = Advanced_Feature_Selection_Classic(target=self.target,ml_usecase=self.ml_usecase,top_features_to_pick=self.top_n_correlated , random_state=self.random_state, subclass=self.subclass)
    dummy_all_t = afs.fit_transform(dummy_all) 
    
    data_fe_final = pd.concat((data,dummy_all_t),axis=1)     # self.data_fe[self.corr]
    # # making sure no duplicated columns are there
    data_fe_final = data_fe_final.loc[:,~data_fe_final.columns.duplicated()] # new added
    # # remove thetarget column
    # # this is the final data we want that includes original , fe data plus impact of top n correlated
    self.columns_to_keep = data_fe_final.drop(self.target,axis=1).columns 
    del(dummy_all)
    del(dummy_all_t)
   
    return(data_fe_final)



#____________________________________________________________________________________________________________________________________________________________________
# Empty transformer
class Empty(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return same DF 
  '''

  def __init__(self):
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,data,y=None):
    return(data)

  def fit_transform(self,data,y=None):
    return(self.transform(data))

#____________________________________________________________________________________________________________________________________
# reduce feature space
class Reduce_Dimensions_For_Supervised_Path(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return same DF with different types of dimensionality reduction modles (pca_liner , pca_kernal, tsne , pls, incremental)
    - except pca_liner, every other method takes integer as number of components 
    - only takes numeric variables (float & One Hot Encoded)
    - it is intended to solve supervised ML usecases , such as classification / regression
  '''

  def __init__(self, target, method='pca_liner', variance_retained_or_number_of_components=.99, random_state=42):
    self.target= target
    self.variance_retained = variance_retained_or_number_of_components
    self.random_state= random_state
    self.method = method
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    if self.method in ['pca_liner' , 'pca_kernal', 'tsne' , 'incremental']: #if self.method in ['pca_liner' , 'pca_kernal', 'tsne' , 'incremental','psa']
      data_pca = self.pca.transform(dataset)
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      return(data_pca)
    else:
      return(dataset)

  def fit_transform(self,dataset,y=None):

    if self.method == 'pca_liner':
      self.pca = PCA(self.variance_retained,random_state=self.random_state)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'pca_kernal': # take number of components only
      self.pca = KernelPCA(self.variance_retained,kernel='rbf',random_state=self.random_state,n_jobs=-1)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    # elif self.method == 'pls': # take number of components only
    #   self.pca = PLSRegression(self.variance_retained,scale=False)
    #   # fit transform
    #   data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1),dataset[self.target])[0] 
    #   data_pca = pd.DataFrame(data_pca)
    #   data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
    #   data_pca.index = dataset.index
    #   data_pca[self.target] = dataset[self.target]
    #   return(data_pca)
    elif self.method == 'tsne': # take number of components only
      self.pca = TSNE(self.variance_retained,random_state=self.random_state)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    elif self.method == 'incremental': # take number of components only
      self.pca = IncrementalPCA(self.variance_retained)
      # fit transform
      data_pca = self.pca.fit_transform(dataset.drop(self.target,axis=1))
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      data_pca[self.target] = dataset[self.target]
      return(data_pca)
    else:
      return(dataset)


#___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one
def Preprocess_Path_One(train_data,target_variable,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=True,
                                imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                apply_zero_nearZero_variance = False,
                                club_rare_levels = False, rara_level_threshold_percentage =0.05,
                                apply_untrained_levels_treatment= False,untrained_levels_treatment_method = 'least frequent',
                                apply_ordinal_encoding = False, ordinal_columns_and_categories= {},
                                apply_cardinality_reduction=False, cardinal_method = 'cluster', cardinal_features=[],
                                apply_binning=False, features_to_binn =[],
                                apply_grouping= False , group_name=[] , features_to_group_ListofList=[[]],
                                apply_polynomial_trigonometry_features = False, max_polynomial=2,trigonometry_calculations=['sin','cos','tan'], top_poly_trig_features_to_select_percentage=.20,
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                target_transformation= False,target_transformation_method ='bc',
                                remove_outliers = False, outlier_contamination_percentage= 0.01,outlier_methods=['pca','iso','knn'],
                                apply_feature_selection = False, feature_selection_top_features_percentage=.80,
                                feature_selection_method = 'classic',
                                remove_multicollinearity = False, maximum_correlation_between_features= 0.90,
                                remove_perfect_collinearity= False,
                                apply_feature_interactions= False, feature_interactions_to_apply=['multiply','divide','add','subtract'],feature_interactions_top_features_to_select_percentage=.01,
                                cluster_entire_data= False, range_of_clusters_to_try=20,
                                apply_pca = False , pca_method = 'pca_liner',pca_variance_retained_or_number_of_components =.99 ,
                                random_state=42
                                

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Ordinal Encoder
      - 4) Drop categorical variables that have zero variance or near zero variance
      - 5) Club categorical variables levels togather as a new level (other_infrequent) that are rare / at the bottom 5% of the variable distribution
      - 6) Club unseen levels in test dataset with most/least frequent levels in train dataset 
      - 7) Reduce high cardinality in categorical features using clustering or counts
      - 8) Generate sub features from time feature such as 'month','weekday',is_month_end','is_month_start' & 'hour'
      - 9) Group features by calculating min, max, mean, median & sd of similar features
      -10) Make nonliner features (polynomial, sin , cos & tan)
      -11) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile,maxabs,robust) , including option to transform target variable
      -12) Apply binning to continious variable when numeric features are provided as a list 
      -13) Detect & remove outliers using isolation forest, knn and PCA
      -14) Apply clusters to segment entire data
      -15) One Hot / Dummy encoding
      -16) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      -17) Feature Selection throuh Random Forest , LightGBM and Pearson Correlation / Boruta algorithm
      -18) Fix multicollinearity
      -19) Feature Interaction (DFS) , multiply , divided , add and substract features
      -20) Apply diamension reduction techniques such as pca_liner, pca_kernal, incremental, tsne 
          - except for pca_liner, all other method only takes number of component (as integer) i.e no variance explaination metohd available  
  '''
  global c2, subcase

  # also make sure that all the column names are string 
  train_data.columns = [str(i) for i in train_data.columns]
  if test_data is not None:
    test_data.columns = [str(i) for i in test_data.columns]
  


  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  if ml_usecase is None:
    if ( ( (c1) & (c2) ) | (c3)   ):
      ml_usecase ='classification'
    else:
      ml_usecase ='regression'
  
  if ((len(train_data[target_variable].unique()) > 2) and (ml_usecase != 'regression')):
    subcase = 'multi'
  else:
    subcase = 'binary'
  
  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,categorical_features=categorical_features,numerical_features=numerical_features,time_features=time_features,features_todrop=features_todrop,display_types=display_types)

  
  # for imputation
  if imputation_type == "simple imputer":
    global imputer
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,categorical_strategy=categorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,categorical_strategy=categorical_imputation_strategy,target_variable=target_variable)
  
  # for zero_near_zero
  if apply_zero_nearZero_variance == True:
    global znz
    znz = Zroe_NearZero_Variance(target=target_variable)
  else:
    znz = Empty()

  # for rare levels clubbing:
  
  if club_rare_levels == True:
    global club_R_L
    club_R_L = Catagorical_variables_With_Rare_levels(target=target_variable,threshold=rara_level_threshold_percentage)
  else:
    club_R_L= Empty()

  # untrained levels in test
  if apply_untrained_levels_treatment ==  True:
    global new_levels 
    new_levels = New_Catagorical_Levels_in_TestData(target=target_variable,replacement_strategy=untrained_levels_treatment_method)
  else:
    new_levels= Empty()

  # untrained levels in test(ordinal specific)
  if apply_untrained_levels_treatment ==  True:
    global new_levels1 
    new_levels1 = New_Catagorical_Levels_in_TestData(target=target_variable,replacement_strategy=untrained_levels_treatment_method)
  else:
    new_levels1= Empty()
 
  # cardinality:
  global cardinality
  if apply_cardinality_reduction==True and cardinal_method =='cluster':
    cardinality = Reduce_Cardinality_with_Clustering(target_variable=target_variable, catagorical_feature=cardinal_features, check_clusters_upto=50,random_state=random_state)
  elif apply_cardinality_reduction==True and cardinal_method =='count':
    cardinality = Reduce_Cardinality_with_Counts(catagorical_feature=cardinal_features)
  else:
    cardinality= Empty()

  # ordinal coding
  if apply_ordinal_encoding == True:
    # we need to make sure that if the columns chosen by user have NA & imputer strategy is not_availablle then we add that to the category first
    for i in ordinal_columns_and_categories.keys():
      if sum(train_data[i].isna()) > 0:
        if categorical_imputation_strategy=='not_available':
          lis = ['not_available'] + ordinal_columns_and_categories[i]
          ordinal_columns_and_categories.update({i:lis})
    
    global ordinal
    ordinal = Ordinal(info_as_dict=ordinal_columns_and_categories)
  else:
    ordinal = Empty()

  # grouping
  if apply_grouping == True:
    global group
    group = Group_Similar_Features(group_name=group_name,list_of_grouped_features=features_to_group_ListofList)
  else:
    group = Empty()

  # non_liner_features
  if apply_polynomial_trigonometry_features == True:
    global nonliner
    nonliner = Make_NonLiner_Features(target=target_variable,ml_usecase=ml_usecase,Polynomial_degree=max_polynomial,other_nonliner_features=trigonometry_calculations,top_features_to_pick=top_poly_trig_features_to_select_percentage,random_state=random_state,subclass=subcase)
  else:
    nonliner = Empty()

  # binning 
  if apply_binning == True:
    global binn
    binn = Binning(features_to_discretize=features_to_binn)
  else:
    binn = Empty()

  # scaling & power transform
  if scale_data == True:
    global scaling 
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method,random_state_quantile=random_state)
  else: 
    scaling = Empty()
  
  if Power_transform_data== True:
    global P_transform
    P_transform = Scaling_and_Power_transformation(target=target_variable,function_to_apply=Power_transform_method,random_state_quantile=random_state)
  else:
    P_transform= Empty()
  
  # target transformation
  if ((target_transformation == True) and (ml_usecase == 'regression')):
    global pt_target
    pt_target = Target_Transformation(target=target_variable,function_to_apply=target_transformation_method)
  else:
    pt_target = Empty()


  # for Time Variables
  global feature_time
  feature_time = Make_Time_Features()
  global dummy
  dummy = Dummify(target_variable)

  # remove putliers
  if remove_outliers == True:
    global rem_outliers
    rem_outliers= Outlier(target=target_variable,contamination=outlier_contamination_percentage,random_state=random_state,methods=outlier_methods )
  else:
    rem_outliers = Empty()
  
  # cluster all data:
  if cluster_entire_data ==True:
    global cluster_all
    cluster_all = Cluster_Entire_Data(target_variable=target_variable,check_clusters_upto=range_of_clusters_to_try, random_state = random_state )
  else:
    cluster_all = Empty()
  
  # clean column names for special char
  clean_names =Clean_Colum_Names()

  # feature selection 
  if apply_feature_selection != None:
    global feature_select
    # TODO: add autoselect 
    if feature_selection_method == 'classic':
      feature_select = Advanced_Feature_Selection_Classic(target=target_variable,ml_usecase=ml_usecase,top_features_to_pick=feature_selection_top_features_percentage,random_state=random_state,subclass=subcase)
    elif feature_selection_method == 'boruta':
      feature_select = Boruta_Feature_Selection(target=target_variable,ml_usecase=ml_usecase,top_features_to_pick=feature_selection_top_features_percentage,
                                                random_state=random_state,subclass=subcase)
  else:
    feature_select= Empty()
  
  # removing multicollinearity
  global fix_multi
  if remove_multicollinearity == True and subcase != 'multi':
    fix_multi = Fix_multicollinearity(target_variable=target_variable,threshold=maximum_correlation_between_features)
  elif remove_multicollinearity == True and subcase == 'multi':
    fix_multi = Fix_multicollinearity(target_variable=target_variable,threshold=maximum_correlation_between_features,correlation_with_target_preference=0.0)
  else:
    fix_multi = Empty()
  
  # remove 100% collinearity
  if remove_perfect_collinearity == True:
    global fix_perfect
    fix_perfect = Remove_100(target=target_variable)
  else:
    fix_perfect = Empty()

  
  # apply dfs
  if apply_feature_interactions == True:
    global dfs
    dfs = DFS_Classic(target=target_variable,ml_usecase=ml_usecase,interactions=feature_interactions_to_apply,top_features_to_pick_percentage=feature_interactions_top_features_to_select_percentage,random_state=random_state,subclass=subcase )
  else:
    dfs= Empty()

  
  # apply pca
  if apply_pca == True:
    global pca
    pca = Reduce_Dimensions_For_Supervised_Path(target=target_variable,method = pca_method ,variance_retained_or_number_of_components=pca_variance_retained_or_number_of_components, random_state=random_state)
  else:
    pca= Empty()

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('new_levels1',new_levels1), # specifically used for ordinal, so that if a new level comes in a feature that was marked ordinal can be handled 
                 ('ordinal',ordinal),
                 ('cardinality',cardinality),
                 ('znz',znz),
                 ('club_R_L',club_R_L),
                 ('new_levels',new_levels),
                 ('feature_time',feature_time),
                 ('group',group),
                 ('nonliner',nonliner),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('pt_target',pt_target),
                 ('binn',binn),
                 ('rem_outliers',rem_outliers),
                 ('cluster_all',cluster_all),
                 ('dummy',dummy),
                 ('fix_perfect',fix_perfect),
                 ('clean_names',clean_names),
                 ('feature_select',feature_select),
                 ('fix_multi',fix_multi),
                 ('dfs',dfs),
                 ('pca',pca)
                 ])
  
  if test_data is not None:
    return(pipe.fit_transform(train_data),pipe.transform(test_data))
  else:
    return(pipe.fit_transform(train_data))



# ______________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one_unsupervised
def Preprocess_Path_Two(train_data,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=False,
                                imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                apply_zero_nearZero_variance = False,
                                club_rare_levels = False, rara_level_threshold_percentage =0.05,
                                apply_untrained_levels_treatment= False,untrained_levels_treatment_method = 'least frequent',
                                apply_cardinality_reduction=False, cardinal_method = 'cluster', cardinal_features=[],
                                apply_ordinal_encoding = False, ordinal_columns_and_categories= {}, 
                                apply_binning=False, features_to_binn =[],
                                apply_grouping= False , group_name=[] , features_to_group_ListofList=[[]],
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                remove_outliers = False, outlier_contamination_percentage= 0.01,outlier_methods=['pca','iso','knn'],
                                remove_multicollinearity = False, maximum_correlation_between_features= 0.90,
                                remove_perfect_collinearity= False,  
                                apply_pca = False , pca_method = 'pca_liner',pca_variance_retained_or_number_of_components =.99 , 
                                random_state=42

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - THIS IS BUILt FOR UNSUPERVISED LEARNING
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Ordinal Encoder
      - 4) Drop categorical variables that have zero variance or near zero variance
      - 5) Club categorical variables levels togather as a new level (other_infrequent) that are rare / at the bottom 5% of the variable distribution
      - 6) Club unseen levels in test dataset with most/least frequent levels in train dataset 
      - 7) Reduce high cardinality in categorical features using clustering or counts
      - 8) Generate sub features from time feature such as 'month','weekday',is_month_end','is_month_start' & 'hour'
      - 9) Group features by calculating min, max, mean, median & sd of similar features
      -10) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile,maxabs,robust) , including option to transform target variable
      -11) Apply binning to continious variable when numeric features are provided as a list 
      -12) Detect & remove outliers using isolation forest, knn and PCA
      -13) One Hot / Dummy encoding
      -14) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      -15) Fix multicollinearity
      -16) Apply diamension reduction techniques such as pca_liner, pca_kernal, incremental, tsne 
          - except for pca_liner, all other method only takes number of component (as integer) i.e no variance explaination metohd available 
  '''
  
  
  # also make sure that all the column names are string 
  train_data.columns = [str(i) for i in train_data.columns]
  if test_data is not None:
    test_data.columns = [str(i) for i in test_data.columns]
  
  # just make a dummy target variable
  target_variable = 'dummy_target'
  train_data[target_variable] = 2
  # just to add diversified values to target
  train_data.loc[0:3,target_variable] = 3
 
  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  # dummy usecase
  ml_usecase ='regression'
  

  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,categorical_features=categorical_features,numerical_features=numerical_features,time_features=time_features,features_todrop=features_todrop,display_types=display_types)

  
  # for imputation
  global imputer
  if imputation_type == "simple imputer":
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,categorical_strategy=categorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,categorical_strategy=categorical_imputation_strategy,target_variable=target_variable)
  
  # for zero_near_zero
  global znz
  if apply_zero_nearZero_variance == True:
    znz = Zroe_NearZero_Variance(target=target_variable)
  else:
    znz = Empty()
 
  # for rare levels clubbing:
  global club_R_L
  if club_rare_levels == True:
    club_R_L = Catagorical_variables_With_Rare_levels(target=target_variable,threshold=rara_level_threshold_percentage)
  else:
    club_R_L= Empty()
  
  # untrained levels in test
  if apply_untrained_levels_treatment ==  True:
    global new_levels 
    new_levels = New_Catagorical_Levels_in_TestData(target=target_variable,replacement_strategy=untrained_levels_treatment_method)
  else:
    new_levels= Empty()

  # untrained levels in test(ordinal specific)
  if apply_untrained_levels_treatment ==  True:
    global new_levels1 
    new_levels1 = New_Catagorical_Levels_in_TestData(target=target_variable,replacement_strategy=untrained_levels_treatment_method)
  else:
    new_levels1= Empty()

  # cardinality:
  global cardinality
  if apply_cardinality_reduction==True and cardinal_method =='cluster':
    # cardinality = Reduce_Cardinality_with_Clustering(target_variable=target_variable, catagorical_feature=cardinal_features, check_clusters_upto=50,random_state=random_state)
    cardinality= Empty()
  elif apply_cardinality_reduction==True and cardinal_method =='count':
    cardinality = Reduce_Cardinality_with_Counts(catagorical_feature=cardinal_features)
  else:
    cardinality= Empty()
  
 # ordinal coding
  if apply_ordinal_encoding == True:
    # we need to make sure that if the columns chosen by user have NA & imputer strategy is not_availablle then we add that to the categories first
    for i in ordinal_columns_and_categories.keys():
      if sum(train_data[i].isna()) > 0:
        if categorical_imputation_strategy=='not_available':
          lis = ['not_available'] + ordinal_columns_and_categories[i]
          ordinal_columns_and_categories.update({i:lis})
    
    global ordinal
    ordinal = Ordinal(info_as_dict=ordinal_columns_and_categories)
  else:
    ordinal = Empty()
  
  # grouping
  if apply_grouping == True:
    global group
    group = Group_Similar_Features(group_name=group_name,list_of_grouped_features=features_to_group_ListofList)
  else:
    group = Empty()
  

  # binning 
  global binn

  if apply_binning == True:
    binn = Binning(features_to_discretize=features_to_binn)
  else:
    binn = Empty()
  
  # for scaling
  global scaling ,P_transform
  if scale_data == True:
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method,random_state_quantile=random_state)
  else: 
    scaling = Empty()
  
  if Power_transform_data== True:
    P_transform = Scaling_and_Power_transformation(target=target_variable,function_to_apply=Power_transform_method,random_state_quantile=random_state)
  else:
    P_transform= Empty()

  # for Time Variables
  global feature_time
  feature_time = Make_Time_Features()
  global dummy
  dummy = Dummify(target_variable)
  
  # remove putliers
  if remove_outliers == True:
    global rem_outliers
    rem_outliers= Outlier(target=target_variable,contamination=outlier_contamination_percentage,random_state=random_state,methods=outlier_methods )
  else:
    rem_outliers = Empty()

  # clean column names for special char
  clean_names =Clean_Colum_Names()

  # removing multicollinearity
  if remove_multicollinearity == True: 
    global fix_multi
    fix_multi = Fix_multicollinearity(target_variable=target_variable,threshold=maximum_correlation_between_features,correlation_with_target_preference=0.0)
  else:
    fix_multi = Empty()

  # remove 100% collinearity
  if remove_perfect_collinearity == True:
    global fix_perfect
    fix_perfect = Remove_100(target=target_variable)
  else:
    fix_perfect = Empty()
  
  # apply pca
  global pca
  if apply_pca == True:
    pca = Reduce_Dimensions_For_Supervised_Path(target=target_variable,method = pca_method ,variance_retained_or_number_of_components=pca_variance_retained_or_number_of_components, random_state=random_state)
  
  else:
    pca= Empty()

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('new_levels1',new_levels1), # specifically used for ordinal, so that if a new level comes in a feature that was marked ordinal can be handled 
                 ('ordinal',ordinal),
                 ('cardinality',cardinality),
                 ('znz',znz),
                 ('club_R_L',club_R_L),
                 ('new_levels',new_levels),
                 ('feature_time',feature_time),
                 ('group',group),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('binn',binn),
                 ('fix_perfect',fix_perfect),
                 ('rem_outliers',rem_outliers),
                 ('dummy',dummy),
                 ('clean_names',clean_names),
                 ('fix_multi',fix_multi),
                 ('pca',pca)
                 ])
  
  if test_data is not None:
    train_t = pipe.fit_transform(train_data)
    test_t = pipe.transform(test_data)
    return(train_t.drop(target_variable,axis=1),test_t)
  else:
    train_t = pipe.fit_transform(train_data)
    return(train_t.drop(target_variable,axis=1))
