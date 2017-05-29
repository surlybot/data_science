import pandas as pd
import numpy as np
import os, sys
from time import time

# preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import PCA

# models
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from tpot import TPOTRegressor, TPOTClassifier
from sklearn.neighbors import KNeighborsClassifier

# sklearn-utils
from sklearn.pipeline import make_pipeline, make_union

# calibration and accuracy
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, log_loss
from sklearn.metrics import classification_report


#######################################################################################
# IDEAS

#kfold validation  |  < 50k rows
	# seperate data into two or more sections 'folds'
	# create one model per fold with data assigned to that fold 
		# used for validation and the rest for training

#if dataset.size > 800MB: split into train, valid, holdout
#else:
# cross validation @ 10,20,30,40,50,60,70,&80 % - holdout stay 20
#######################################################################################

#traindata
#testdata

# split 30/60/80/20h

# parameters {}

# gridsearchcv w/ parameter checks

# transformer runs - pca, normalizer, scaler

# create pipeline
#                 -- worker pool -- 

#     	|	 pipeline check eclf1			|
#		|				|					|
#		|				|					|
#		|				|					|
#		|				|					|

class VotingClassifier(object):
	"""Stripped-down version of VotingClassifier that uses prefit estimators"""
	def __init__(self, estimators, voting='hard', weights=None):
		self.estimators = [e[1] for e in estimators]
		self.named_estimators = dict(estimators)
		self.voting = voting
		self.weights = weights

	def fit(self, X, y, sample_weight=None):
		raise NotImplementedError
		
	def predict(self, X):
		""" Predict class labels for X.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.
		Returns
		----------
		maj : array-like, shape = [n_samples]
			Predicted class labels.
		"""

		check_is_fitted(self, 'estimators')
		if self.voting == 'soft':
			maj = np.argmax(self.predict_proba(X), axis=1)

		else:  # 'hard' voting
			predictions = self._predict(X)
			maj = np.apply_along_axis(lambda x:
									  np.argmax(np.bincount(x,
												weights=self.weights)),
									  axis=1,
									  arr=predictions.astype('int'))
		return maj

	def _collect_probas(self, X):
		"""Collect results from clf.predict calls. """
		return np.asarray([clf.predict_proba(X) for clf in self.estimators])

	def _predict_proba(self, X):
		"""Predict class probabilities for X in 'soft' voting """
		if self.voting == 'hard':
			raise AttributeError("predict_proba is not available when"
								 " voting=%r" % self.voting)
		check_is_fitted(self, 'estimators')
		avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
		return avg

	@property
	def predict_proba(self):
		"""Compute probabilities of possible outcomes for samples in X.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.
		Returns
		----------
		avg : array-like, shape = [n_samples, n_classes]
			Weighted average probability for each class per sample.
		"""
		return self._predict_proba

	def transform(self, X):
		"""Return class labels or probabilities for X for each estimator.
		Parameters
		----------
		X : {array-like, sparse matrix}, shape = [n_samples, n_features]
			Training vectors, where n_samples is the number of samples and
			n_features is the number of features.
		Returns
		-------
		If `voting='soft'`:
		  array-like = [n_classifiers, n_samples, n_classes]
			Class probabilities calculated by each classifier.
		If `voting='hard'`:
		  array-like = [n_samples, n_classifiers]
			Class labels predicted by each classifier.
		"""
		check_is_fitted(self, 'estimators')
		if self.voting == 'soft':
			return self._collect_probas(X)
		else:
			return self._predict(X)

	def _predict(self, X):
		"""Collect results from clf.predict calls. """
		return np.asarray([clf.predict(X) for clf in self.estimators]).T
		


def dummies(X, y):

	stock_classifiers = []

	# get quick results from DummyClassifiers
	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	clf = LogisticRegression(n_jobs=-1).fit(X_train, y_train)
	print("\nSimple LogisticRegression Score: {:.2f}".format(clf.score(X_test, y_test)))

	clf2 = DummyClassifier(strategy='most_frequent',random_state=0)
	clf2.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
	print("\nDummyClassifier using 'most_frequent' strategy: {:.2f}".format(clf2.score(X_test, y_test)))
		
	
		
def accuracy(y_true, y_pred):
    return float(sum(y_pred == y_true)) / len(y_true)


		
if __name__ == '__main__':
	
	train = pd.read_csv('numerai_training_data.csv')
	test = pd.read_csv('numerai_tournament_data.csv')
	
	y = train['target']					# target to predict
	X = train.drop('target', axis=1)	# training data

	t_id = test['t_id']
	x_test = test.drop('t_id', axis=1)
	
	# get dummy scores
	# select, preproc, gridsearchcv, whatever -- get gud models & prefit, send to ensemble
	# setup voting ensemble --> send to tpot
	# cross validation
	# score matrix: [brier loss, log loss, etc]

	#from sklearn.preprocessing import MultiLabelBinarizer
	#mlb = MultiLabelBinarizer()
	#CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])
	
	pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
	pipeline_optimizer.fit(X, y)
	result = pipeline_optimizer.score(x_test, t_id)
	print(result)
	
	pipeline_optimizer.export('tpot_exported_pipeline.py')




	
	
	

