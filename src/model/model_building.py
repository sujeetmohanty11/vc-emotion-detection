import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingClassifier

import yaml

estimator   = yaml.safe_load(open('params.yaml', 'r'))['model_building']['n_estimators']
rate        = yaml.safe_load(open('params.yaml', 'r'))['model_building']['learning_rate']


train_data = pd.read_csv('./data/features/train_bow.csv')


#Model Feed Data
X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

#Model Training
clf = GradientBoostingClassifier(n_estimators=estimator, learning_rate=rate)
clf.fit(X_train, y_train)

#Model Saving
pickle.dump(clf, open('model.pkl', 'wb'))