print(__doc__)
import pandas as pd
import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.svm import LinearSVC
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import assert_greater_equal
from sklearn.datasets import make_blobs

from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, log_loss

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LogisticRegression as LR

from math import log

print('imports work')

# -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))

# sklearn.metrics.log_loss(y_true, y_pred, eps=1e-15, normalize=True, sample_weight=None, labels=None)


def logloss1(act, pred):
	""" Vectorised computation of logloss """
	
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1, pred)))
	ll = ll * -1.0/len(act)
	return ll

	

def logloss2(predicted, target):
	""" manual logloss calc """
	
	if len(predicted) != len(target):
		print('lengths not equal!')
		return None
		 
	target = [float(x) for x in target]   # make sure all float values
	predicted = [min([max([x,1e-15]),1-1e-15]) for x in predicted]  # within (0,1) interval
	 
	return -(1.0/len(target))*sum([target[i]*log(predicted[i]) +  
				(1.0-target[i])*log(1.0-predicted[i]) for i in range(len(target))])

									
	
def sk_logloss():
	''' calc logloss given a numpy array using sklearn.metrics.log_loss '''
	
	# testing multi-class setting with classifier that implements only decision function
	clf = LinearSVC()
	
	X, y_idex = make_blobs(n_samples=100, n_features=2, random_state=42,
							centers=3, cluster_std=3.0)
							
	# use categorical labels to check that CalibratedClassifierCV supps them correctly
	target = train['target']
	y = target[y_idx]
	
	X_train, y_train = X[::2], y[::2]
	X_test, y_test = X[1::2], y[1::2]
	
	clf.fit(X_train, y_train)
	
	for method in ['isotonic', 'sigmoid']:
		cal_clf = CalibratedClassifierCV(clf, method=method, cv=2)
		cal_clf.fit(X_train, y_train)
		probas = cal_clf.predict_proba(X_test)
		assert_array_almost_equal(np.sum(probas, axis=1), np.ones(len(X_test)))
		
		# check that log loss of calibrated classifier is smaller than naively turned
		# OvR decision function to probabilities via softmax
		def softmax(y_pred):
			e = np.exp(-y_pred)
			return e / e.sum(axis=1).reshape(-1,1)
		uncalibrated_log_loss = \
			log_loss(y_test, softmax(clf.decision_function(X_test)))
		calibrated_log_loss = log_loss(y_test, probas)
		assert_greater_equal(uncalibrated_log_loss, calibrated_log_loss)
		
	# test the calibration of a multiclass classifier decreases log-loss for
	# random forest classifier
	X, y = make_blobs(n_samples=100, n_features=2, random_state=42, cluster_std=3.0)
	X_train, y_train = X[::2], y[::2]
	X_test, y_test = X[1::2], y[1::2]
	
	clf = RandomForestClassifier(n_estimators=10, random_state=42)
	clf.fit(X_train, y_train)
	clf_probs = clf.predict_proba(X_test)
	loss = log_loss(y_test, clf_probs)
	
	for method in ['isotonic', 'sigmoid']:
		cal_clf = CalibratedClassifierCV(clf, method=method, cv=3)
		cal_clf.fit(X_train, y_train)
		cal_clf_probs = cal_clf.predict_proba(X_test)
		cal_loss = log_loss(y_test, cal_clf_probs)
		assert_greater(loss, cal_loss)
	
	

def numerai_pred(train, test):
	"""Load data, scale, train a linear model, output predictions"""

	# moved to main
	#train_file = './numerai_training_data.csv'
	#test_file = './numerai_tournament_data.csv'
	#output_file = './predictions_lr.csv'

	#

	train = pd.read_csv( train_file )
	test = pd.read_csv( test_file )

	#

	y_train = train.target.values

	x_train = train.drop( 'target', axis = 1 )
	x_test = test.drop( 't_id', axis = 1 )

	print( "training..." )

	lr = LR()
	lr.fit( x_train, y_train )

	print ( "predicting..." )

	p = lr.predict_proba( x_test )

	print ( "saving..." )

	test['probability'] = p[:,1]
	# moved to main
	#test.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )
	return test
	
	

def train_and_evaluate( y_train, x_train, y_val, x_val ):

	lr = LR()
	lr.fit( x_train, y_train )

	#
	
	p = lr.predict_proba( x_val )

	#
	
	auc = AUC( y_val, p[:,1] )
	ll = log_loss( y_val, p[:,1] )
	
	#
	
	return ( auc, ll )
	
	
	
def transform_train_and_evaluate( transformer ):
	
	global x_train, x_val, y_train
	
	#
	
	x_train_new = transformer.fit_transform( x_train )
	x_val_new = transformer.transform( x_val )
	
	#
	
	return train_and_evaluate( y_train, x_train_new, y_val, x_val_new )
		

	
if __name__ == '__main__':
	

	# read in data set as pandas dataframe (for any data exploration needed)
	train_file = './numerai_training_data.csv'
	test_file = './numerai_tournament_data.csv'
	output_file = './predictions_lr.csv'

	get_new_predictions = False
	if get_new_predictions:	
	
		# get predictions using logistic regression
		preds = numerai_pred(train_file, test_file)
		
		# output to csv
		preds.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )
	
	
	#####################################
	## validate_ numerai predictions
	#####################################
	
	d = pd.read_csv( train_file )
	#train, val = train_test_split( d, test_size = 5000 ) # 69.20%
	#train, val = train_test_split( d, test_size = 8000 ) # 69.12%
	#train, val = train_test_split( d, test_size = 6000 ) # 69.18%
	train, val = train_test_split( d, test_size = 13000 ) # 69.02%

	y_train = train.target.values
	y_val = val.target.values

	x_train = train.drop( 'target', axis = 1 )
	x_val = val.drop( 'target', axis = 1 )

	# train, predict, evaluate

	auc, ll = train_and_evaluate( y_train, x_train, y_val, x_val )

	print ( "No transformation" )
	print ( "AUC: {:.2%}, log loss: {:.2%} \n".format( auc, ll ) )

	# try different transformations for X
	# X is already scaled to (0,1) so these won't make much difference

	transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
		Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ) ]

	#poly_scaled = Pipeline([ ( 'poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
	#transformers.append( PolynomialFeatures(), poly_scaled )

	for transformer in transformers:

		print( transformer )
		auc, ll = transform_train_and_evaluate( transformer )
		print ( "AUC: {:.2%}, log loss: {:.2%} \n".format( auc, ll ) )
	
	
