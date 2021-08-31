import numpy as np
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from utilities import plot_data, CleanUpDataFrame
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from utilities import SqrtTransform
from patsy import dmatrices
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn import linear_model as lm

df_org = pd.read_csv('transfusion.data')

predictors = ['Recency (months)', 'Time (months)', 'Frequency (times)', 'Monetary (c.c. blood)']

target = 'whether he/she donated blood in March 2007'

df = df_org.copy()
df['ratio'] = df['Recency (months)']/df['Time (months)']
df['rate'] = df['Frequency (times)']/(df['Recency (months)']+1)
df['rate_2'] = df['Frequency (times)']*np.exp((1/df['Time (months)']))

# sns.pairplot(df, hue=target)

X = df[predictors]
y = df[target]


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(X=X,y=y):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, val_index in split.split(X=X_train,y=y_train):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_val, y_val = X.iloc[val_index], y.iloc[val_index]

scaler = StandardScaler()
pca = PCA(whiten=True, n_components=0.95)

data_prep_pipeline = Pipeline([('sqrt_transform', None),
                               ('scaler', scaler),
                               ('pca', None)])


base_model = VotingClassifier([('rf', RandomForestClassifier()),
                               ('mlp', MLPClassifier(hidden_layer_sizes=(32,16,16, 16))),
                              ], n_jobs=-1, voting='soft')


base_model = BaggingClassifier( MLPClassifier(hidden_layer_sizes=(32,16,16)), n_estimators=10, bootstrap_features=True, n_jobs=-1)

model = Pipeline([('data_prep_pipeline', data_prep_pipeline),
                 ('base_model', base_model)])

params = [{'base_model__n_estimators': [10]}]
           
           
grid_search = GridSearchCV(model, params, cv = 10, scoring='neg_log_loss', n_jobs=-1)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

print(grid_search.best_score_, grid_search.best_params_)
print(log_loss(y_train, model.predict_proba(X_train)))


# calibrated_clf = CalibratedClassifierCV(base_estimator=grid_search.best_estimator_)

# calibrated_clf.fit(X_val, y_val)
# model = calibrated_clf
# print(log_loss(y_train, model.predict_proba(X_train)))
# print(log_loss(y_val, model.predict_proba(X_val)))


# y_pred_test = model.predict_proba(X_val)
# log_loss(y_val, y_pred_test)



