import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
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

def libg_get_if_distibution_is_normal(sample1):
  '''
    Test whether a sample differs from a normal distribution.
      This function tests the null hypothesis that a sample comes from a normal distribution. 
      It is based on D’Agostino and Pearson’s [1], [2] test that combines skew and kurtosis to produce an omnibus test of normality.

      alpha = 1e-3 # = 1 / 1000
      print("         pvalue = {:g}".format(pvalue), '         stat', stat)
      if pvalue < alpha:  # null hypothesis: x comes from a normal distribution
          print("The null hypothesis can be rejected,the distribution is normal ---- pvalue:", pvalue)
      else:
          print("The null hypothesis cannot be rejected,the distribution is NOT normal ---- pvalue", pvalue)
      #arr = [1,2,4,-1,4,2,0,2,4,4,6,-2,-3,4,5,2,5,4,-2,-1,3,2,1]
      #x = libg_get_if_distibution_is_normal(arr)

  '''
  from scipy import stats

  if len(sample1) < 10:
    return None

  stat,pvalue = stats.normaltest(sample1)
  pvalue_adj = max(0.000000001, pvalue) # 1 miliardo
  score = int(1/pvalue_adj)
  print(stat,pvalue,pvalue_adj,score)

  min_score_for_normality_of_distribution = 100
  if score >= min_score_for_normality_of_distribution:
    return False
  else:
    return True




def check_norm_2(x):
  from scipy import stats
  k2, p = stats.normaltest(x)
  alpha = 1e-3
  #print("p = {:g}".format(p))
  p = 8.4713e-19
  if p < alpha:  # null hypothesis: x comes from a normal distribution
    return True
    print("The null hypothesis can be rejected")
  else:
    return False
    print("The null hypothesis cannot be rejected")


def unknown_imputer(X, missing_value = 'Unknown'):
  X = X.values
  unique_values, count = np.unique(X,return_counts=True)
  num_nan = count[unique_values == missing_value]
  counting = count[unique_values != missing_value]
  values = unique_values[unique_values != missing_value]
  X_new = X.copy()
  freq = counting / np.sum(counting)
  X_new[X_new == missing_value] = np.random.choice(values,size=num_nan,p=freq)
  return X_new
