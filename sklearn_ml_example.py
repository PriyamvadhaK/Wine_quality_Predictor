# start
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.externals import joblib

# Load data into panda structure

dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')

# Print out some stats
print data.head()
print data.shape
print data.describe()

# Split data

y = data.quality
X = data.drop('quality', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, stratify=y)

print "Split datasets"
# Now we have 11 features

print X_train.shape
print X_test.shape
print y_train.shape
print y_test.shape

# We need to scale the features so we have a normalized feature space

scaler = preprocessing.StandardScaler().fit(X_train)

# This line basically transforms the data using the scaler
# The model fits on the transformed data 
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))

print pipeline.get_params()
print "Pipeline created"
# Set some hyperparameter values to do serach on
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Optimize the given hyperparameters over the training set

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
print "Gridsearch over"
# Fit and tune model

clf.fit(X_train, y_train)

print clf.best_params_

# Refit on entire training set 

print clf.refit

print "Refit over training set"

# Predict on unseen data

y_pred = clf.predict(X_test)

# Evaluate model

print "r2 error: ", r2_score(y_test, y_pred)
print "MSE: ", mean_squared_error(y_test, y_pred)

joblib.dump(clf, './rf_regressor_wine.pkl')
