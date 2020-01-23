import pandas as pd
import numpy as np
import ipywidgets as wg 
from IPython.display import display
from ipywidgets import Layout
from sklearn.base import BaseEstimator , TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
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
        if sum(data[i].isna()) == 0: 
          if len(data[i].unique()) == len_samples:
            min_number = min(data[i])
            max_number = max(data[i])
            arr = np.arange(min_number,max_number+1,1)
            try:
              all_match = sum(data[i] == arr)
              if all_match == len_samples:
                self.id_columns.append(i) 
            except:
              None 
    
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
        data[i]=data[i].apply(str)
    
    if len(self.numerical_features) > 0:
      for i in self.numerical_features:
        data[i]=data[i].astype('float64')
    
    if len(self.time_features) > 0:
      for i in self.time_features:
        data[i]=pd.to_datetime(data[i])

    # table of learent types
    self.learent_dtypes = data.dtypes
    #self.training_columns = data.drop(self.target,axis=1).columns

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
      display(wg.Text(value="Following data types have been inferred automatically, if they are correct press enter to continue or type 'quit' otherwise.",layout =Layout(width='100%')))
      
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
# Scaling & Power Transform
class Scaling_and_Power_transformation(BaseEstimator,TransformerMixin):
  '''
    -Given a data set, applies Min Max, Standar Scaler or Power Transformation (yeo-johnson)
    -it is recommended to run Define_dataTypes first
    - ignores target variable 
      Args: 
        target: string , name of the target variable
        function_to_apply: string , default 'zscore' (standard scaler), all other {'minmaxm','yj','quantile'} ( min max,yeo-johnson & quantile power transformation )

  '''

  def __init__(self,target,function_to_apply='zscore',random_state_quantile=42):
    self.target = target
    self.function_to_apply = function_to_apply
    self.random_state_quantile = random_state_quantile
  
  def fit(self,dataset,y=None):
    data = dataset.copy()
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
        self.scale_and_power = PowerTransformer(method='yeo-johnson',standardize=False)
        self.scale_and_power.fit(data[self.numeric_features])
      elif  self.function_to_apply == 'quantile':
        self.scale_and_power = QuantileTransformer(random_state=self.random_state_quantile,output_distribution='normal')
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

  def fit_transform(self,data,y=None):
    self.fit(data)
    return(self.transform(data))
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
#____________________________________________________________________________________________________________________________________________________________________
# Simple Liner PCA transformer
class Liner_PCA(BaseEstimator,TransformerMixin):
  '''
    - Takes DF, return same DF with PCA 
    - only takes numeric variables (float & One Hot Encoded)
  '''

  def __init__(self, target, variance_retained=.99, random_state=42):
    self.target= target
    self.variance_retained = variance_retained
    self.random_state= random_state
    return(None)

  def fit(self,data,y=None):
    return(None)

  def transform(self,dataset,y=None):
    if self.is_categ > 0:
      data_pca = self.pca.transform(dataset)
      data_pca = pd.DataFrame(data_pca)
      data_pca.columns = ["Component_"+str(i) for i in np.arange(1,len(data_pca.columns)+1)]
      data_pca.index = dataset.index
      return(data_pca)
    else:
      return(dataset)

  def fit_transform(self,dataset,y=None):
    self.is_categ = len([i for i in dataset.columns if len(dataset[i].unique()) == 2 and dataset[i].unique()[0] in [0,1] and dataset[i].unique()[1] in [0,1]])
    # we will only apply this if there are catagorical variables
    if self.is_categ > 0:
      # We are only running this if 
      # define
      self.pca = PCA(self.variance_retained,random_state=self.random_state)
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
def Preprocess_Path_One(train_data,target_variable,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],
                               imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                random_state=42

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile)
      - 4) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      - 3) One Hot / Dummy encoding
  '''

  # WE NEED TO AUTO INFER the ml use case
  c1 = train_data[target_variable].dtype == 'int64'
  c2 = len(train_data[target_variable].unique()) <= 20
  c3 = train_data[target_variable].dtype == 'object'
  
  if ml_usecase is None:
    if ( ( (c1) & (c2) ) | (c3)   ):
      ml_usecase ='classification'
    else:
      ml_usecase ='regression'
  
  
  global dtypes 
  dtypes = DataTypes_Auto_infer(target=target_variable,ml_usecase=ml_usecase,categorical_features=categorical_features,numerical_features=numerical_features,time_features=time_features,features_todrop=features_todrop)

  
  # for imputation
  global imputer
  if imputation_type == "simple imputer":
    imputer = Simple_Imputer(numeric_strategy=numeric_imputation_strategy, target_variable= target_variable,categorical_strategy=categorical_imputation_strategy)
  else:
    imputer = Surrogate_Imputer(numeric_strategy=numeric_imputation_strategy,categorical_strategy=categorical_imputation_strategy,target_variable=target_variable)

  global scaling ,P_transform
  if scale_data == True:
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method)
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
  
  # clean column names for special char
  clean_names =Clean_Colum_Names()
  

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('feature_time',feature_time),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('dummy',dummy),
                 ('clean_names',clean_names)
                 ])
  
  if test_data is not None:
    return(pipe.fit_transform(train_data),pipe.transform(test_data))
  else:
    return(pipe.fit_transform(train_data))



# ______________________________________________________________________________________________________________________________________________________
# preprocess_all_in_one_unsupervised
def Preprocess_Path_Two(train_data,ml_usecase=None,test_data =None,categorical_features=[],numerical_features=[],time_features=[],features_todrop=[],display_types=False,
                               imputation_type = "simple imputer" ,numeric_imputation_strategy='mean',categorical_imputation_strategy='not_available',
                                scale_data= False, scaling_method='zscore',
                                Power_transform_data = False, Power_transform_method ='quantile',
                                apply_pca = True , pca_variance_retained =.99 , 
                                random_state=42

                               ):
  
  '''
    Follwoing preprocess steps are taken:
      - THIS IS BUILD OF UNSUPERVISED LEARNING , FOLLOWES SAME PATH AS PATH ONE
      - 1) Auto infer data types 
      - 2) Impute (simple or with surrogate columns)
      - 3) Scales & Power Transform (zscore,minmax,yeo-johnson,quantile)
      - 4) Remove special characters from column names such as commas, square brackets etc to make it competible with jason dependednt models
      - 3) One Hot / Dummy encoding
  '''
  
  # just make a dummy target variable
  target_variable = 'dummy_target'
  train_data[target_variable] = 2

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

  global scaling ,P_transform
  if scale_data == True:
    scaling = Scaling_and_Power_transformation(target=target_variable,function_to_apply=scaling_method)
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
  
  # clean column names for special char
  clean_names =Clean_Colum_Names()

  # apply pca
  global pca
  if apply_pca == True:
    pca = Liner_PCA(target=target_variable, variance_retained=pca_variance_retained, random_state=random_state)
  else:
    pca= Empty()

  global pipe
  pipe = Pipeline([
                 ('dtypes',dtypes),
                 ('imputer',imputer),
                 ('feature_time',feature_time),
                 ('scaling',scaling),
                 ('P_transform',P_transform),
                 ('dummy',dummy),
                 ('clean_names',clean_names),
                 ('pca',pca)
                 ])
  
  if test_data is not None:
    train_t = pipe.fit_transform(train_data)
    test_t = pipe.transform(test_data)
    return(train_t.drop(target_variable,axis=1),test_t)
  else:
    train_t = pipe.fit_transform(train_data)
    return(train_t.drop(target_variable,axis=1))
