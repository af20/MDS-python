'''
BOZZA DI PROGETTO

- Dataset kaggle (Heart failure Prediction Dataset)
- TASK classif bin superv

  abbiamo serie info su vari pazienti, 
  in base a caratt paziente dobbiamo dire se 
  predittive su insorgere o meno ictus

  (1) ETL processing (veloce) - poca manipolaz (one hot enc)
    dati mancanti ce ne so alcuni, risolvibili con drop

  (2) Costruire IS OS  - 20% test

  (3) Scegliere classif. + adatti x qst scopo
    Perceptr.
    Logistic Regr
    Support Vec Machine

  (4) Cross-Validation (misura perf)
    calcola accuracy, prec, rec, f1-score

    quali modelli si comportano meglio?
      distribuz. misure perf (box-plot) o media, stdev performance
  
  NOTE: <1000 elem per fold
'''
# Data management
import pandas as pd

# Data preprocessing and trasformation (ETL)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml, load_iris, make_moons, make_classification


# Math and Stat modules
import numpy as np
from scipy.stats import sem
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# Unsupervised Learning

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl


df = pd.read_csv('Classification/healthcare-dataset-stroke-data.csv')
df.dropna(subset=['bmi'],inplace=True)


print(df['smoking_status'].unique())
v_one_hot = ['gender', 'work_type', 'Residence_type']

'''
DA DROPPARE 
  id
DA CATEGORIZZARE
  age, avg_glucose_level

DA RADOMIZZARE (most frequent weighted)
  smoking_status (ha Unknown)

OUTCOME
  stroke

# ever_married   Yes=1   No=0
# TODO Residence_type   ['Urban' 'Rural'] ==> One hot   o   0,1 ??  ===>   0,1 (riduciamo 1 dimensione)
'''

df.drop(columns=['id'], inplace=True)


'''
  numeric_features = credit_data.select_dtypes(include=['int64','float64']).columns
  cat_features = credit_data.select_dtypes(exclude=['int64','float64']).columns
  test_pipeline = ColumnTransformer([('numeric', StandardScaler(), numeric_features), ('category', OneHotEncoder(), cat_features)])
  trasformed_credit = test_pipeline.fit_transform(credit_data)

'''