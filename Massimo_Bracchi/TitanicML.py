import os

# Data management
import pandas as pd

# Data preprocessing and trasformation (ETL)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
# Math and Stat modules
import numpy as np
from scipy.special import logit, expit
from scipy import stats

# Visualization
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#importo il dataset

titanic_dataset = pd.read_csv('~/Documents/MachineLearning/Titanic-Dataset.csv')

titanic_dataset.head(4)

titanic_dataset.info()

titanic_dataset.nunique().sort_values()

#elimino le 2 righe con un dato mancante nella feature Embarked
titanic_dataset.dropna(subset=['Embarked'], inplace=True, axis=0) 

#estraggo la colonna delle label ma non la manipolo dato che è già espressa in codice binario
titanic_dataset_label = titanic_dataset['Survived']

#tolgo la colonna delle label, la colonna Cabin perchè ha troppi dati mancanti e altre colonne che ritengo inutili
titanic_dataset = titanic_dataset.drop(columns=['Survived','PassengerId','Name','Cabin','Ticket'])

'''
Ci sono 177 righe mancanti nella colonna Age.
Andremo a fare sia un'imputazione dei dati mancanti inserendo il valore della media calcolata sui dati che abbiamo a disposizione (cioè 
circa 30), sia una binarizzazione 
'''

age_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('binarizer', Binarizer(threshold=18))
])

category_pipeline = Pipeline([
    ('ordinal', OneHotEncoder())
])

standard_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

data_preprocessing = ColumnTransformer([
    ('age', age_pipeline, ['Age']),
    ('gender', OrdinalEncoder (categories=['male','female']), ['Sex']),
    ('pclass', category_pipeline, ['Pclass']),
    ('embarked', category_pipeline, ['Embarked']),
    ('fare', RobustScaler(), ['Fare']),
    ('bro', standard_pipeline, ['SibSp']),
    ('son', standard_pipeline,['Parch'])
])

feature_matrix = data_preprocessing.fit_transform(titanic_dataset)
