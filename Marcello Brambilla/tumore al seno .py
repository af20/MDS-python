#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, FunctionTransformer, Binarizer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import  ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.impute import SimpleImputer
from matplotlib.pyplot import figure
from sklearn.metrics import accuracy_score


# La sfida chiave contro la sua individuazione è come classificare i tumori in maligni (cancerosi) o benigni (non cancerosi). 
# Ti chiediamo di completare l'analisi della classificazione di questi tumori utilizzando l'apprendimento automatico (con SVM) e il set di dati (diagnostico) del cancro al seno del Wisconsin.

# 1.1 STEP 1: ETL PROCESSING

# In[2]:


breast_dataset = pd.read_csv('OneDrive/Documenti/GitHub/MDS-python/Marcello Brambilla/breast-cancer.csv')


# In[3]:


breast_dataset


# In[9]:


breast_dataset.head()


# In[5]:


breast_label = breast_dataset['diagnosis'].map(
    {'B':0,
     'M':1
    }
).values
breast_dataset.drop(columns=['id','diagnosis'], inplace = True)


# In[10]:


breast_dataset.info()


# In[39]:


data_preprocessing = ColumnTransformer([
    ('scaler',StandardScaler(), ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'])
    ],
    remainder = 'passthrough'
)


# In[40]:


feature_matrix = data_preprocessing.fit_transform(breast_dataset)


# In[41]:


fm = pd.DataFrame(feature_matrix)
fm


# 1.2 STEP 2: TRAINING E TEST SETS
# dividiamo il dataset in training e test sets in modo tale che il test set contenga il 20% dei record.

# In[42]:


X_train, X_test, y_train, y_test = train_test_split(feature_matrix, breast_label, test_size = 0.2, random_state = 42)


# 1.3 STEP 3: LA SCELTA DEGLI ALGORITMI/MODELLI DA UTILIZZARE
# -Perceptron
# -LogisticRegression
# -Support Vector Machine

# 1.3.1 PECPTRON

# In[43]:


for _ in range(10):
    X_train, X_test, y_train, y_test = train_test_split(feature_matrix, breast_label, test_size = 0.2)
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train) # apprendo su training
    predicted_test = perceptron.predict(X_test) # predico sul test
    print(np.sum(predicted_test == y_test)/len(y_test))


# ## STEP 4: cross-validation
# 
# Utilizzanod 5-fold cross-validation devo valutare le performance dei diversi modelli (la scelta degli iperparametri per ora non e' vincolata). Nel dettaglio si devono utilizzare come misure di performance:
# - accuracy
# - precision
# - recall 
# - f1-score
# 
# Per ogni modello si deve costruire la distribuzione della misura di performance (un box plot e' sufficiente), oppure calcolare media e deviazione standard.

# 1.4.1 A.PERCETTRONE

# In[63]:


p = Perceptron()
cvs = cross_val_score(p, X_train, y_train, cv = 5)


# In[64]:


cvs


# In[65]:


np.mean(cvs)


# In[66]:


np.std(cvs)


# In[67]:


plt.boxplot(cvs)


# In[68]:


y_train_predicted = cross_val_predict(p, X_train, y_train, cv = 5)


# In[70]:


as1 = accuracy_score(y_train, y_train_predicted, normalize=True)
as1


# In[71]:


confusion_matrix(y_train, y_train_predicted)


# In[72]:


precision_score(y_train, y_train_predicted)


# In[73]:


recall_score(y_train, y_train_predicted)


# In[75]:


f1_score(y_train, y_train_predicted)


# In[ ]:




