import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Data management
import pandas as pd

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

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl

from ML.ETL import do_ETL
'''
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
'''

Matrix, v_stoke = do_ETL()