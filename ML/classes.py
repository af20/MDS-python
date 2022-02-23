import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Data management
import pandas as pd
import pickle

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
from scipy.stats import sem, randint
from random import choice

# Supervised Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold, RepeatedKFold, ShuffleSplit, StratifiedShuffleSplit, learning_curve, validation_curve, cross_validate
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# Hyperparameter Optimization
#from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.fixes import loguniform

# Unsupervised Learning

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

import seaborn as sns
from sklearn.tree import export_graphviz



class c_Perceptron:
  def __init__(self, X_train, X_test, y_train, y_test):
    perceptron = Perceptron() # perceptron.fit(X_train, y_train) # accuracy = np.sum(predicted_test == y_test) / len(y_test)
    #v_accuracy = cross_val_score(perceptron, X_train, y_train, cv = 5) # perceptron_score
    #accuracy = np.mean(v_accuracy)
    #print('v_accuracy', v_accuracy, '    accuracy', accuracy)
    self.y_train_predicted = y_train_predicted = cross_val_predict(perceptron, X_train, y_train, cv = 10)
    C = confusion_matrix(y_train, y_train_predicted)
    self.accuracy = (C[0][0]+C[1][1]) / (sum(C[0])+sum(C[1]))
    self.precision = precision_score(y_train, y_train_predicted)
    self.recall = recall_score(y_train, y_train_predicted)
    self.f1_score = f1_score(y_train, y_train_predicted)


class c_Logistic_Regression:
  def __init__(self, X_train, X_test, y_train, y_test, SOGLIA, max_iter=1000):
    logit_cls = LogisticRegression(max_iter = 1000)
    y_predicted_score = cross_val_predict(logit_cls, X_train, y_train, cv = 10, method='decision_function') # [-4.53289277 -3.72660414 -2.6409855  ... -3.53140

    '''
    train_sizes, train_scores, test_scores = learning_curve(logit_cls,
                                                            X=X,
                                                            y=Y,
                                                            train_sizes= [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                            cv = 10,
                                                            n_jobs = -1,
                                                            shuffle = True
                                                        )
    y_pred = logreg.predict(X_test)
    '''
    prec, recall, soglia = precision_recall_curve(y_train, y_predicted_score)   #for i in range(len(prec)):  print(soglia[i], '   p:', prec[i], '   r:', recall[i])
    # XXX ????? COSA è prec (ha 3918 valori, come X_train)
    # print(soglia) # [-7.03620597 -7.03143287 -7.02950994 ... -0.1958
    soglia_prec = soglia[np.argmax(prec >= SOGLIA)] #   SOGLIA=0.9   #  soglia_prec 0.23674 ---- massimo valore dell'array soglia   #  max_soglia = max(soglia)
    y_predicted_score = y_predicted_score >= soglia_prec # [False False False ... False False False]
    self.precision = precision_score(y_train, y_predicted_score)
    self.recall = recall_score(y_train, y_predicted_score)
    self.f1_score = f1_score(y_train, y_predicted_score)



class c_linear_SVM:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.svm_clf = LinearSVC(C=1, max_iter=50000)
    self.svm_clf.fit(X, Y)
  
  def plot_learning_curves(self):
    Cs = [0.01, 0.1, 1, 10, 100] # definire un insieme di valori di C tenendo in considerazione le precedenti osservazioni sul suo effetto 
    fig = plt.figure(figsize=(18,6))
    for i, c in enumerate(Cs):
      print('Training SVM per C =', c)
      svm_cls = LinearSVC(C = c, max_iter=50000)

      train_sizes, train_scores, test_scores = learning_curve(svm_cls, X = self.X, y = self.Y, train_sizes=np.linspace(0.1,1,10), cv = 5, n_jobs=-1, shuffle = True)

      train_mean = np.mean(train_scores, axis=1)
      train_std = np.std(train_scores, axis=1)
      test_mean = np.mean(test_scores, axis=1)
      test_std = np.std(test_scores, axis=1)
      ax = fig.add_subplot(150+(i+1))
      ax.plot(train_sizes, train_mean,
          color='blue', marker='o',
          markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes,
                  train_mean + train_std,
                  train_mean - train_std,
                  alpha=0.15, color='blue')
      ax.plot(train_sizes, test_mean,
          color='green', linestyle='--',
          marker='s', markersize=5,
          label='Validation accuracy')
      ax.fill_between(train_sizes,
                  test_mean + test_std,
                  test_mean - test_std,
                  alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.8,1))
      ax.set_xlabel('Dimensione del training set')
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
    plt.show()





'''
  Funzioni per plottare
'''
def plot_dataset(X, Y, axes):
  plt.plot(X[:, 0][Y==0], X[:, 1][Y==0], "bs")
  plt.plot(X[:, 0][Y==1], X[:, 1][Y==1], "g^")
  plt.axis(axes)
  plt.grid(True, which='both')
  plt.xlabel(r"$x_1$", fontsize=20)
  plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
  plt.figure(figsize=(12, 4))


def plot_predictions(clf, axes):
  x0s = np.linspace(axes[0], axes[1], 100)
  x1s = np.linspace(axes[2], axes[3], 100)
  x0, x1 = np.meshgrid(x0s, x1s)
  X = np.c_[x0.ravel(), x1.ravel()]
  y_pred = clf.predict(X).reshape(x0.shape)
  y_decision = clf.decision_function(X).reshape(x0.shape)
  plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
  plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)


class c_non_linear_SVM:
  def __init__(self, X, Y, kernel, D):
    assert kernel in ['poly', 'rbf'], "Wrong kernel modality"

    if kernel == 'poly':
      self.SVM = SVC(kernel="poly", degree=D['degree'], coef0=D['coef0'], C=D['C'])
    elif kernel == 'rbf':
      self.SVM = SVC(kernel="poly", gamma=D['gamma'], C=D['C'])


    
    self.X = X
    self.Y = Y
    self.SVM.fit(X, Y)

  def plot_2d_chart(self):
    # stampa N grafici, non 1 solo
    plt.subplot(121)
    plot_predictions(self.SVM, [-1.5, 2.5, -1, 1.5])
    plot_dataset(self.X, self.Y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=2, r=1, C=5$", fontsize=18)

    plt.subplot(122)
    plot_predictions(self.SVM, [-1.5, 2.5, -1, 1.5])
    plot_dataset(self.X, self.Y, [-1.5, 2.5, -1, 1.5])
    plt.title(r"$d=7, r=1, C=5$", fontsize=18)

    plt.show()



  def plot_rbf_hyperparams(self):
    # stampa N grafici separati, non 1 solo unico

    gamma1, gamma2 = 0.1, 5
    C1, C2 = 0.001, 1000
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

    svm_clfs = []
    for gamma, C in hyperparams:
      rbf_kernel_svm_clf = SVC(kernel="rbf", gamma=gamma, C=C)
      rbf_kernel_svm_clf.fit(self.X, self.Y)
      svm_clfs.append(rbf_kernel_svm_clf)

    plt.figure(figsize=(11, 7))

    for i, svm_clf in enumerate(svm_clfs):
      plt.subplot(221 + i)
      plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
      plot_dataset(self.X, self.Y, [-1.5, 2.5, -1, 1.5])
      gamma, C = hyperparams[i]
      plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    plt.show()
  


  def plot_learning_curves(self):
    gamma1, gamma2 = 0.1, 2
    C1, C2 = 0.01, 5
    hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for gamma, C in hyperparams:
      rbf_kernel_svm_clf = SVC(kernel="rbf", gamma = gamma, C = C)
      train_size, train_scores, test_scores = learning_curve(rbf_kernel_svm_clf,
                                                        X=self.X,
                                                        y=self.Y,
                                                        train_sizes=np.linspace(0.1,1.0,10),
                                                        cv=5,
                                                        n_jobs=-1)
      print('fatto {}, {}'.format(gamma,C))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(4):
      ax = fig.add_subplot(221+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"$\gamma={}, C={}$".format(*hyperparams[i]), fontsize=18)
    plt.show()


class c_decision_tree:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y
    self.tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42) # min_samples_leaf
    self.tree_clf.fit(X, Y)
  

  def plot_decision_boundary(self):
    def plot_decision_boundary(clf, X, y, axes=[-1.5, 2.5, -1, 1.5], iris=True, legend=False, plot_training=True, alpha = 0.8):
        x1s = np.linspace(axes[0], axes[1], 100)
        x2s = np.linspace(axes[2], axes[3], 100)
        x1, x2 = np.meshgrid(x1s, x2s)
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
        custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
        if not iris:
            custom_cmap2 = ListedColormap(['#7d7d58','#4c4c7f','#507d50'])
            plt.contour(x1, x2, y_pred, cmap=custom_cmap2, alpha=alpha)
        if plot_training:
            plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
            plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
            plt.plot(X[:, 0][y==2], X[:, 1][y==2], "g^", label="Iris-Virginica")
            plt.axis(axes)
        if iris:
            plt.xlabel("Petal length", fontsize=14)
            plt.ylabel("Petal width", fontsize=14)
        else:
            plt.xlabel(r"$x_1$", fontsize=18)
            plt.ylabel(r"$x_2$", fontsize=18, rotation=0)
        if legend:
            plt.legend(loc="lower right", fontsize=14)

    plt.figure(figsize=(8, 4))
    plot_decision_boundary(self.tree_clf, self.X, self.Y)
    plt.plot([2.45, 2.45], [0, 3], "k-", linewidth=2)
    plt.plot([2.45, 7.5], [1.75, 1.75], "k--", linewidth=2)
    plt.text(1.40, 1.0, "Profondita'=0", fontsize=15)
    plt.text(3.2, 1.80, "Profondita'=1", fontsize=13)
    plt.show()

  def plot_learning_curve(self):
    min_leaf = [5, 10, 100, 200, 350]

    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for mlf in min_leaf:
      dt_mlf = DecisionTreeClassifier(min_samples_leaf=mlf, random_state=42, max_depth=15)
      train_size, train_scores, test_scores = learning_curve(dt_mlf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=10,
                                                          n_jobs=-1)
      print('fatto {}'.format(mlf))
      train_means.append(np.mean(train_scores, axis=1))
      train_stds.append(np.std(train_scores, axis=1))
      test_means.append(np.mean(test_scores, axis=1))
      test_stds.append(np.std(test_scores, axis=1))
      train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(5):
      ax = fig.add_subplot(231+i)
      ax.plot(train_sizes[i], train_means[i], color='blue', marker='o', markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i], train_means[i] + train_stds[i], train_means[i] - train_stds[i], alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i], color='green', linestyle='--', marker='s', markersize=5, label='Validation accuracy')
      ax.fill_between(train_sizes[i], test_means[i] + test_stds[i], test_means[i] - test_stds[i], alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.4,1))
      ax.set_ylabel('Accuracy')
      ax.legend(loc='lower right')
      ax.set_title(r"min_sam_leaf:{}".format(min_leaf[i]), fontsize=18)
    plt.show()


class c_ensemble:
  def __init__(self, X, Y, estimators, voting):
    # Hard e Soft voting
    self.X = X
    self.Y = Y
    self.voting_clf = VotingClassifier(estimators=estimators, voting=voting) # voting_clf
    

  def visualize_ensemble_performance(self):
    scoring = ['recall','accuracy', 'f1']
    scores = cross_validate(self.voting_clf, self.X, self.Y,
                            cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42),
                            return_estimator = True,
                            return_train_score= True,
                            scoring = scoring,
                            n_jobs=-1)
    data_score = []
    labels = []

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot()
    for clf in scores['estimator'][0].estimators_:
        scores_clf = cross_validate(clf, self.X, self.Y,
                        cv = StratifiedShuffleSplit(n_splits=20, test_size=0.2, random_state=42),
                        return_train_score= True,
                        scoring = scoring,
                        n_jobs=-1)
        data_score.extend([scores_clf[t+s] for s in scoring for t in ['train_','test_']])
        labels.extend([clf.__class__.__name__+'_'+t+s for s in scoring for t in ['train_','test_']])
    data_score.extend([scores[t+s] for s in scoring for t in ['train_','test_']])
    labels.extend(['Voting_'+t+s for s in scoring for t in ['train_','test_']])
    sns.boxplot(ax = ax,
                data = data_score,
                whis = [5, 95],
                palette = "vlag",
                orient = 'h'
                )
    ax.set(yticklabels=labels)
    plt.show()


class c_bagging:
  def __init__(self, X, Y):
    '''
      Un secondo approccio di ensemble e' utilizzare lo stesso modello di apprendimento 
      ma addestrato su diversi partizionamenti del training set. 
        BAGGING (o bootstrap aggregating): Se utilizzo un campionamento con reimmissione, 
        PASTING in caso di non reimmisione

    '''
    self.X = X
    self.Y = Y

    self.bag_clf = BaggingClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=500, max_samples=200, bootstrap=True, n_jobs=-1)
    self.dt_clf = DecisionTreeClassifier()

  def plot_accuracy(self):
    train_sizes, train_means, test_means, test_stds, train_stds = [],[],[],[],[]
    for clf in [self.bag_clf, self.dt_clf]:
        train_size, train_scores, test_scores = learning_curve(clf,
                                                          X=self.X,
                                                          y=self.Y,
                                                          train_sizes=np.linspace(0.1,1.0,10),
                                                          cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
                                                          n_jobs=-1)
        print('fatto {}'.format(clf))
        train_means.append(np.mean(train_scores, axis=1))
        train_stds.append(np.std(train_scores, axis=1))
        test_means.append(np.mean(test_scores, axis=1))
        test_stds.append(np.std(test_scores, axis=1))
        train_sizes.append(train_size)

    fig= plt.figure(figsize=(12, 8))
    for i in range(2):
      ax = fig.add_subplot(121+i)
      ax.plot(train_sizes[i], train_means[i],
          color='blue', marker='o',
          markersize=5, label='Training accuracy')
      ax.fill_between(train_sizes[i],
                  train_means[i] + train_stds[i],
                  train_means[i] - train_stds[i],
                  alpha=0.15, color='blue')
      ax.plot(train_sizes[i], test_means[i],
          color='green', linestyle='--',
          marker='s', markersize=5,
          label='Validation accuracy')
      ax.fill_between(train_sizes[i],
                  test_means[i] + test_stds[i],
                  test_means[i] - test_stds[i],
                  alpha=0.15, color='green')
      ax.grid()
      ax.set_ylim((0.8,1))
      chart = 'Bagging' if i == 0 else 'Dec.Tree'
      ax.set_ylabel('Accuracy - ' + chart)
      ax.legend(loc='lower right')
    plt.show()


class c_random_forest:
  def __init__(self, X, Y):
    '''
    BAGGING + DT
      E' un ensamble di Decision Trees utilizzando metodo Bagging con max_samples = len(training set)
    SOTTO-INSIEME FEATURES x crescita
      Nella crescita dell'albero non vengono considerate tutte le feature per cercare lo splitting migliore ma un sottoinsieme casuale. 
      In questo modo aumento la diversita' degli alberi.

    '''
    self.X = X
    self.Y = Y
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=0.2, stratify=Y)
    self.RF_clf = RandomForestClassifier(n_estimators=250, max_leaf_nodes=64, n_jobs=-1)#, max_features=10)
    self.RF_clf.fit(self.X_train, self.y_train)
    self.et_clf = ExtraTreesClassifier(n_estimators=250, max_leaf_nodes=64, n_jobs=-1)#, max_features=10)
    self.et_clf.fit(self.X_train, self.y_train)
    # named_feat_importance = dict(zip(columns_name, self.RF_clf.feature_importances_)) # l'importanza delle feature   # sorted(named_feat_importance.items(), key=lambda x:x[1], reverse=True)[:10]


  def plot_RF_ET(self):
    scores_rnf = cross_val_score(self.RF_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_et = cross_val_score(self.et_clf, self.X_train, self.y_train, cv=5, scoring='f1',n_jobs=-1)
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot()
    sns.boxplot(ax = ax,
                data = [scores_rnf, scores_et],
                palette = 'vlag',
                orient = 'h'
              )
    ax.set(yticklabels=['RF','ET'])
    plt.show()

  def ada_boosting(self):
    '''
      ## Boosting
          Sono metodi di ensemble in cui i classificatori sono addestrati in modo sequenziale, ed ogni classificatori corregge il classificatore precedente. 
          I metodi piu' importanti sono *AdaBoost* e *Gradient Boosting*.
      ### AdaBoost
          Un modo per correggere un classificatore e' focalizzari sulle istanze che il predecessore non classifica correttamente, cioe' sulla istanze piu' difficili.
          Per dare piu' importanza alle istanze non classificate correttamente aumento il peso di quelle istanze e addestro un classificatore con le istanze ri-pesate. Il processo viene ripetuto per il numero di classificatori nel pool.
    '''
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=6),
        n_estimators=100,
        algorithm='SAMME.R',
        learning_rate=0.5
    )
    scores_ada = cross_val_score(ada_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_rnf = cross_val_score(self.RF_clf, self.X_train, self.y_train, cv=5, scoring='f1', n_jobs=-1)
    scores_et = cross_val_score(self.et_clf, self.X_train, self.y_train, cv=5, scoring='f1',n_jobs=-1)
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot()

    sns.boxplot(ax = ax,
                data = [scores_rnf, scores_et, scores_ada],
                palette = 'vlag',
                orient = 'h'
              )
    ax.set(yticklabels=['RF','ET','ADA'])
    plt.show()


  def supp_plot_predictions(self, regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
      plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)

  def plot_predicts(self, X_rnd, y_rnd, tree_reg1, tree_reg2, tree_reg3, y2, y3):
    plt.figure(figsize=(11,11))
    plt.subplot(321)
    self.supp_plot_predictions([tree_reg1], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h_1(x_1)$", style="g-", data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Residui e DT", fontsize=16)

    plt.subplot(322)
    self.supp_plot_predictions([tree_reg1], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1)$", data_label="Training set")
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.title("Predizione Gradient Boosting", fontsize=16)

    plt.subplot(323)
    self.supp_plot_predictions([tree_reg2], X_rnd, y2, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_2(x_1)$", style="g-", data_style="k+", data_label="Residuals")
    plt.ylabel("$y - h_1(x_1)$", fontsize=16)

    plt.subplot(324)
    self.supp_plot_predictions([tree_reg1, tree_reg2], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1)$")
    plt.ylabel("$y$", fontsize=16, rotation=0)

    plt.subplot(325)
    self.supp_plot_predictions([tree_reg3], X_rnd, y3, axes=[-0.5, 0.5, -0.5, 0.5], label="$h_3(x_1)$", style="g-", data_style="k+")
    plt.ylabel("$y - h_1(x_1) - h_2(x_1)$", fontsize=16)
    plt.xlabel("$x_1$", fontsize=16)

    plt.subplot(326)
    self.supp_plot_predictions([tree_reg1, tree_reg2, tree_reg3], X_rnd, y_rnd, axes=[-0.5, 0.5, -0.1, 0.8], label="$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$")
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$y$", fontsize=16, rotation=0)
    plt.show()


  def gradient_boosting(self):
    '''
        Gradient Boosting
            Similmente ad AdaBoost, Gradient Boosting agisce in maniera sequenziale ma in ogni step 
            il classificatore apprende sugli errori residui del classificatore precedente.
    '''
    X_rnd = np.random.rand(100, 1) - 0.5

    y_rnd = 3*X_rnd[:, 0]**2 + 0.05 * np.random.randn(100)
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X_rnd, y_rnd)

    y2 = y_rnd - tree_reg1.predict(X_rnd)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg2.fit(X_rnd, y2)

    y3 = y2 - tree_reg2.predict(X_rnd)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg3.fit(X_rnd, y3)

    self.plot_predicts(X_rnd, y_rnd, tree_reg1, tree_reg2, tree_reg3, y2, y3)




class c_grid_search:
  def __init__(self, X, Y):
    param_grid = [
        {'kernel': ['rbf'], 'gamma': [.1, 5, 10], 'C': [0.1, 1, 1000]},
        {'kernel': ['poly'], 'degree': [1, 2, 3], 'coef0': [1, 10, 50]}
    ]

    svm_clf = SVC()

    grid_search = GridSearchCV(estimator=svm_clf,
                              param_grid = param_grid,
                              cv = 5,
                              scoring = 'f1',
                              n_jobs = -1
    )

    grid_search.fit(X, Y)
    print('best params:', grid_search.best_params_,'   Best: {}'.format(grid_search.best_score_), '   best_estimator', grid_search.best_estimator_)
    results = grid_search.cv_results_
    results = pd.DataFrame(results)[['mean_test_score','params']]
    results.sort_values(by='mean_test_score',ascending=False,inplace=True)
    print(results)



class c_grid_search_randomized:
  def __init__(self, X, Y):
    svm_clf = SVC()

    param_dict = {
        'kernel': ['poly'], 
        'degree': randint(2,8),
        'coef0': [1, 10, 50]
    }
    random_search = RandomizedSearchCV(estimator=svm_clf,
                                        param_distributions= param_dict,
                                        cv = 5,
                                        scoring = 'f1',
                                        n_iter=5,
                                        n_jobs = -1
                                      )

    random_search.fit(X, Y)
    results = pd.DataFrame(random_search.cv_results_)[['mean_test_score','params']]
    results.sort_values(by='mean_test_score',ascending=False)
    
    # GridSearchCv e RandomizedSearchCV permettono la ricerca dei parametri ottimali su estimator composti come Pipeline e ColumnTransformer.


class c_my_pipeline:
  def __init__(self):
    credit_card_data = pd.read_csv('ML/BankChurnersMissingData.csv')
    credit_card_data.dropna(subset=['Total_Revolving_Bal','Months_Inactive_12_mon'],
                      inplace=True
                      )

    # Estraggo la  colonna delle label e la rimuovo dal dataset
    credit_card_label = credit_card_data['Attrition_Flag'].map(
        {'Existing Customer':0,
        'Attrited Customer':1
        }
    ).values
    credit_card_data.drop(columns=['Attrition_Flag',
                                  'CLIENTNUM',
                                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2',
                                  'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'],
                          inplace=True)
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

    ui = FunctionTransformer(unknown_imputer)

    customer_age_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    category_pipeline = Pipeline([
        ('imputer', FunctionTransformer(unknown_imputer)),
        ('ordinal', OneHotEncoder())
    ])

    features_robust = ['Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy']
    features_standard = list(set(credit_card_data.select_dtypes(include=['int64','float64']).columns).difference(set(features_robust + ['Avg_Utilization_Ratio', 'Customer_Age'])))


    data_preprocessing = ColumnTransformer([
        ('age', customer_age_pipeline, ['Customer_Age']),
        ('gender', OrdinalEncoder(categories=[['M','F']]), ['Gender']),
        ('edu', category_pipeline, ['Education_Level']),
        ('status', category_pipeline, ['Marital_Status']),
        ('income', category_pipeline, ['Income_Category']),
        ('card', category_pipeline, ['Card_Category']),
        ('numeric_robust', RobustScaler(), features_robust),
        ('feature_standard', StandardScaler(), features_standard)
    ],
        remainder = 'passthrough'
    )

    clf_forest_pipeline = Pipeline(steps=[
        ('preprocessing', data_preprocessing),
        ('classifier', RandomForestClassifier())
    ])
    
    clf_forest_pipeline.get_params().keys()

    param_grid = [
      {'preprocessing__age__imputer__strategy': ['most_frequent','median'],
      'classifier__max_features': [8,10,'auto'],
      'classifier__max_depth':[2,5,8,None],
      'classifier__n_estimators':[10,20,50,100,150]
      }
    ]

    grid_search = GridSearchCV(clf_forest_pipeline, 
                                param_grid= param_grid,
                                cv=5,
                                n_jobs=-1,
                                scoring=['f1','accuracy'],
                                refit = 'f1'
                              )

    grid_search.fit(credit_card_data, credit_card_label)
    print('   best_score:', grid_search.best_score_)
    print('   grid_search.cv_results_ KEYS', grid_search.cv_results_.keys())
    results = pd.DataFrame(grid_search.cv_results_)[['mean_test_f1','params','mean_test_accuracy']]
    results = pd.DataFrame(grid_search.cv_results_)[['mean_test_f1','params','mean_test_accuracy']]
    results.sort_values(by='mean_test_f1',ascending=False)
    print('   best_params:', grid_search.best_params_)



from sklearn.datasets import fetch_openml, load_iris, make_moons, make_classification
X_moon, y_moon = make_moons(n_samples=1000, noise=0.15, random_state=42)
'''
x = c_linear_SVM(X_moon, y_moon)
x.plot_learning_curves()

x = c_non_linear_SVM(X_moon, y_moon, 'rbf', {'gamma':1, 'C':2})
x.plot_learning_curves()
'''

'''
# RF hard voting
log_clf = LogisticRegression(random_state=42)
svm_clf = SVC(random_state=42)
per_clf = Perceptron(random_state=42)

DT = c_ensemble(X_moon, y_moon, [('lr', log_clf), ('pc', per_clf), ('svc', svm_clf)], voting='hard')
DT.visualize_ensemble_performance()
'''

'''
bag = c_random_forest(X_moon, y_moon)
bag.gradient_boosting()
'''
#g = c_grid_search(X_moon, y_moon)

PL = c_my_pipeline()
