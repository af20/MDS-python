import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ML.library import *

def do_ETL():

  # Data management
  import pandas as pd

  # Data preprocessing and trasformation (ETL)
  from sklearn.impute import SimpleImputer
  from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
  from sklearn.pipeline import Pipeline
  from sklearn.compose import ColumnTransformer
  import matplotlib.pyplot as plt


  # Math and Stat modules
  import numpy as np


  df = pd.read_csv('ML/healthcare-dataset-stroke-data.csv')
  v_stoke = df['stroke'].tolist()
  df.dropna(subset=['bmi'], inplace=True)
  df.drop(columns=['id', 'stroke'], inplace=True)
  # df.hist(figsize=(22,9)); plt.show()


  df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1}).tolist()
  df['Residence_type'] = df['Residence_type'].map({'Urban': 0, 'Rural': 1})


  v_float_cols_dtype = list(set(df.select_dtypes(include=['float64']).columns)) # v_float_cols = ['age', 'avg_glucose_level', 'bmi']
  v_float_normal_cols, v_float_not_normal_cols = [], []
  for col in v_float_cols_dtype:
    is_normal = libg_get_if_distibution_is_normal(df[col].tolist()) # is_normal = check_norm_2(df[col].tolist()) 
    if is_normal == True:
      v_float_normal_cols.append(col)
    else:
      v_float_not_normal_cols.append(col)
  # print('v_float_normal_cols', v_float_normal_cols, '\nv_float_not_normal_cols:', v_float_not_normal_cols)  # v_float_normal_cols = ['age', 'bmi'] # v_float_not_normal_cols = ['avg_glucose_level']

  one_hot_pipeline = Pipeline([
    ('imputer', FunctionTransformer(unknown_imputer)),
    ('ordinal', OneHotEncoder())
  ])

  float_normal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),  # sostituisco i valori null / unknown
    ('scaler', StandardScaler())                    # normalizzo
  ])

  float_not_normal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
  ])

  data_preprocessing = ColumnTransformer(
    [
      ('float_normal', float_normal_pipeline, v_float_normal_cols),
      ('float_not_normal', float_not_normal_pipeline, v_float_not_normal_cols),
      ('category', one_hot_pipeline, ['gender', 'work_type', 'smoking_status'])
    ],
      remainder = 'passthrough'
  )

  # Calcolo la feature matrix
  Matrix = data_preprocessing.fit_transform(df)
  return Matrix, v_stoke

'''
  da scalare (check normalità)
    - age
    - avg_glucose_level
    - bmi

  one hot (+ unknown_imputer)
    - gender (male, female)
    - work_type (['Private' 'Self-employed' 'Govt_job' 'children' 'Never_worked'])
    - smoking_status ['formerly smoked' 'never smoked' 'smokes' 'Unknown']    <---------- unknown_imputer
  map
    - ever_married (Yes/No -> 0,1)
    - Residence_type ['Urban' 'Rural'] --> 0,1
  ok
    - hypertension (0,1)
    - heart_disease (0,1)
  
  unknown_imputer
    - smoking_status ['Unknown']
  
  result
    - stoke

  NOTE prof
    - avg_glucose_level(Usare RobustScaler())
    - BMI rapporto tra 2 normali, quindi distribuz.normale ==> meglio StandardScaler() NOTE ASK: giusto fare il test normalità

'''
