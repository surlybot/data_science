import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score as AUC, accuracy_score as accuracy, log_loss
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV


VERSION = 5	#manually increment to maintain older predictions - only changes output csv name
SUCCESS_METRIC_THRESHOLD = .69		# can be used as a stop, if new model isn't 'better' - don't output


def decompositionPCA():
	pass


	
def classificationReport(train):         ############
	"""
	Scikit-learn does provide a convenience report when working on 
	classification problems to give you a quick idea of the accuracy 
	of a model using a number of measures.

	The classification_report() function displays the precision, recall, 
	f1-score and support for each class.
	
	The f1-score gives you the harmonic mean of precision and recall. 
	The scores corresponding to every class will tell you the accuracy 
	of the classifier in classifying the data points in that particular 
	class compared to all other classes.

	The support is the number of samples of the true response that lie in that class. 
	
	You can see good prediction and recall for the algorithm.
	
	>>>             precision    recall  f1-score   support
 
        0.0           0.77      0.87      0.82       162
        1.0           0.71      0.55      0.62        92
 
	avg / total       0.75      0.76      0.75       254
	"""
	
	#array = dataframe.values
	#X = array[:,0:8]
	#Y = array[:,8]
	
	test_size = 0.33
	seed = 7
	
	half = (len(train)/2)
	
	X = train[:,0:half]
	Y = train[:,half]
	
	X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
	model = LogisticRegression()
	model.fit(X_train, Y_train)
	predicted = model.predict(X_test)
	report = classification_report(Y_test, predicted)
	print(report)

	

def getDummyScores(model=None, params=None):

	#X, y = data, target
	#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	train = pd.read_csv('./numerai_training_data_.csv')
	y = train['target']
	X = train.drop('target', axis=1)

	X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

	clf = LogisticRegression(n_jobs=-1).fit(X_train, y_train)
	print("\nSimple LogisticRegression Score: {:.2f}".format(clf.score(X_test, y_test)))

	clf2 = DummyClassifier(strategy='most_frequent',random_state=0)
	clf2.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
	print("\nDummyClassifier using 'most_frequent' strategy: {:.2f}".format(clf2.score(X_test, y_test)))

	clf3 = DummyClassifier(strategy='prior',random_state=0)
	clf3.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
	print("\nDummyClassifier using 'prior' strategy: {:.2f}".format(clf3.score(X_test, y_test)))

	clf4 = DummyClassifier(strategy='stratified',random_state=0)
	clf4.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
	print("\nDummyClassifier using 'stratified' strategy: {:.2f}".format(clf4.score(X_test, y_test)))

	clf5 = DummyClassifier(strategy='uniform',random_state=0)
	clf5.fit(X_train, y_train)
	DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
	print("\nDummyClassifier using 'uniform' strategy: {:.2f}".format(clf5.score(X_test, y_test)))
	
	

def votingClassifier():
	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()
	#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
	#y = np.array([1, 1, 1, 2, 2, 2])

	train = pd.read_csv('./numerai_training_data_.csv')
	y = train['target']
	X = train.drop('target', axis=1)

	print('\nVoting Classsifiers: LogisticRegression, RandomForestClassifier, GaussianNB...\n')

	eclf1 = VotingClassifier(estimators=[
			 ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
	eclf1 = eclf1.fit(X, y)
	print('ECLF1 - Voting Classifier {Options: voting=hard}...')
	print('Predictions: {}'.format(eclf1.predict(X)))
	print('Classifier Score: {:.2f}'.format(eclf1.score(X,y)))
	#print(cross_val_score(eclf1, X, y, scoring='neg_log_loss'))	# not avail when voting hard (apparently)

	eclf2 = VotingClassifier(estimators=[
			 ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
			 voting='soft')
	eclf2 = eclf2.fit(X, y)
	print('\nECLF2 - Voting Classifier {Options: voting=soft}...')
	print('Predictions: {}'.format(eclf2.predict(X)))
	print('Classifier Score: {:.2f}'.format(eclf2.score(X,y)))
	results = cross_val_score(eclf2, X, y, scoring='neg_log_loss')
	print("eclf2: log loss: {:.2f} std: {}".format((abs(results.mean())*100), results.std()))

	eclf3 = VotingClassifier(estimators=[
			('lr', clf1), ('rf', clf2), ('gnb', clf3)],
		   voting='soft', weights=[2,1,1])
	eclf3 = eclf3.fit(X, y)
	print('\nECLF3 - Voting Classifier {Options: voting=soft, weights=[2,1,1] lr,rf,gnb}...')
	print('Predictions: {}'.format(eclf3.predict(X)))
	print('Classifier Score: {:.2f}'.format(eclf3.score(X,y)))
	results = cross_val_score(eclf3, X, y, scoring='neg_log_loss')
	print("eclf3: log loss: {:.2f} std: {}".format((abs(results.mean())*100), results.std()))


	
def predict(transformer=None, param_grid=None):

	train = pd.read_csv('./numerai_training_data_.csv')
	test = pd.read_csv('./numerai_tournament_data.csv')
	
	y_train = train['target']
	x_train = train.drop('target', axis=1)

	t_id = test['t_id']
	x_test = test.drop('t_id', axis=1)
	
	def make_preds(x_train, y_train, param_grid=None):
		grid_search = False
		cv_results = None
		
		print('training...')
		if param_grid:
			print('\t> using gridsearchcv...this will take a minute')
			grid_search = True
			clf = GridSearchCV(LogisticRegression(n_jobs=-1), param_grid)
		else:
			print('\t> using logisticRegression...')
			clf = LogisticRegression(n_jobs=-1)
			
		clf.fit(x_train, y_train)
		
		print('predicting...')
		preds = clf.predict_proba(x_test)

		test['probability'] = preds[:,1]
		
		print('saving...')
		test.to_csv('./predictions_v{}_6FEB.csv'.format(VERSION), columns=('t_id','probability'), index=None)
		
		if grid_search:
			cv_results = clf.cv_results_
			print(cv_results)
			return cv_results
	
		return 0
	
	if transformer:
		print('utilizing {} transformer for best results...'.format(transformer))
		x_train = transformer.fit_transform(x_train)
		
	if param_grid:
		cv_results = make_preds(x_train, y_train, param_grid)
		return cv_results	# returning results of gridsearch - dict of information - for further tweeking in main

	else:
		print('not utilizing a transformer, nor gridsearch...')
		make_preds(x_train, y_train)
		return 0


		
def getBestTransformer():
	""" test different transformers and parameters avail to LogisticRegression class to 
		get the best results """

	d = pd.read_csv('./numerai_training_data_.csv')
	
	best_ll = None		# return - best log loss score after loop
	best_xform = None	# return - best transform (by logloss score)
	models = {}			# return - dictionary of transformer(key)=[AUCscore,LLscore](values)
							# to play with further in main if desired
					
	tr, val = train_test_split(d, test_size=5000)

	y_tr = tr.target.values
	y_val = val.target.values

	x_tr = tr.drop('target', axis=1)
	x_val = val.drop('target',axis=1)

	transformers = [None, MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
		Normalizer(norm = 'l1'), Normalizer(norm = 'l2'), Normalizer(norm = 'max')]
	
	print('\nThe following scores are derived from training data and may not correlate to your test data...\n')
	for transformer in transformers:
	
		y_tr_tmp = y_tr
		y_val_tmp = y_val
		x_tr_tmp = x_tr
		x_val_tmp = x_val
	
		if transformer == None:
			LogisticRegression = LogisticRegression(n_jobs=-1)
			LogisticRegression.fit(x_tr_tmp, y_tr_tmp)
			p = LogisticRegression.predict_proba(x_val_tmp)
			auc = AUC(y_val_tmp, p[:,1] )
			ll = log_loss(y_val_tmp, p[:,1])
			
			print("No transformation")
			print("AUC: {:.2%}, log loss: {:.2%} \n".format(auc, ll))
			
			models['NoTransformer'] = [auc,ll]
			
		elif transformer:
			xtr_new = transformer.fit_transform(x_tr_tmp)
			xval_new = transformer.transform(x_val_tmp)
			
			LogisticRegression = LogisticRegression(n_jobs=1)
			LogisticRegression.fit(xtr_new, y_tr_tmp)
			p = LogisticRegression.predict_proba(xval_new)
			auc = AUC(y_val_tmp, p[:,1] )
			ll = log_loss(y_val_tmp, p[:,1])
			
			print(transformer)
			print("AUC: {:.2%}, log loss: {:.2%} \n".format(auc, ll))
			
			models[transformer] = [auc,ll]
			
		if best_ll == None:
			best_ll = ll
			best_xform = transformer
			
		elif best_ll:
			if ll < best_ll:
				best_ll = ll
				best_xform = transformer
	
	#print('Best log loss: {} \n'.format(best_ll))
	#print(models)

	return best_ll, best_xform, models


	
if __name__ == '__main__':

	# used by GridSearchCV for LogisticRegression class
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
				'class_weight': ['balanced', None]} 
	
	
	# custom transformer and prediction functions
	
	#top_ll, top_xformer, models = getBestTransformer()			# perform quick LR.score() using n transformers - reports/returns best scores
	#predict()													# pure LogisticRegression classification - no tuning
	#predict(top_xformer)										# use tranformer rankings, no gridsearch
	#cv_results = predict(top_xformer, param_grid)				# use tranformer rankings and gridsearch
	#cv_results = predict(None, param_grid)						# no transformer, yes gridsearch

	# voting classifiers
	votingClassifier()											# 3 Voting Classifiers each using: LR, RandForest, GaussianNB
																	# each ensemble uses different voting/weights
	# LogisticRegression v. DummyClassifier scores
	getDummyScores()											# 4 DummyClassifiers-most freq, prior, uniform, stratified
	
"""
To illustrate DummyClassifier, first let’s create an imbalanced dataset:
>>>

>>> from sklearn.datasets import load_iris
>>> from sklearn.model_selection import train_test_split
>>> iris = load_iris()
>>> X, y = iris.data, iris.target
>>> y[y != 1] = -1
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

Next, let’s compare the accuracy of SVC and most_frequent:
>>>

>>> from sklearn.dummy import DummyClassifier
>>> from sklearn.svm import SVC
>>> clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
>>> clf.score(X_test, y_test) 
0.63...
>>> clf = DummyClassifier(strategy='most_frequent',random_state=0)
>>> clf.fit(X_train, y_train)
DummyClassifier(constant=None, random_state=0, strategy='most_frequent')
>>> clf.score(X_test, y_test)  
0.57...

We see that SVC doesn’t do much better than a dummy classifier. Now, let’s change the kernel:
>>>

>>> clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
>>> clf.score(X_test, y_test)  
0.97...

We see that the accuracy was boosted to almost 100%. A cross validation strategy is recommended for a better estimate of the accuracy, if it is not too CPU costly. For more information see the Cross-validation: evaluating estimator performance section. Moreover if you want to optimize over the parameter space, it is highly recommended to use an appropriate methodology; see the Tuning the hyper-parameters of an estimator section for details.

More generally, when the accuracy of a classifier is too close to random, it probably means that something went wrong: features are not helpful, a hyperparameter is not correctly tuned, the classifier is suffering from class imbalance, etc...

DummyRegressor also implements four simple rules of thumb for regression:

    mean always predicts the mean of the training targets.
    median always predicts the median of the training targets.
    quantile always predicts a user provided quantile of the training targets.
    constant always predicts a constant value that is provided by the user.

In all these strategies, the predict method completely ignores the input data.	
Note that with all these strategies, the predict method completely ignores the input data!

>>> from sklearn.metrics import log_loss
>>> y_true = [0, 0, 1, 1]
>>> y_pred = [[.9, .1], [.8, .2], [.3, .7], [.01, .99]]
>>> log_loss(y_true, y_pred)    
0.1738..

7777777777777777777777777	
The function roc_curve computes the receiver operating characteristic curve, or ROC curve. Quoting Wikipedia :

“A receiver operating characteristic (ROC), or simply ROC curve, is a graphical plot which illustrates the performance of a binary classifier system as its discrimination threshold is varied. It is created by plotting the fraction of true positives out of the positives (TPR = true positive rate) vs. the fraction of false positives out of the negatives (FPR = false positive rate), at various threshold settings. TPR is also known as sensitivity, and FPR is one minus the specificity or true negative rate.”

This function requires the true binary value and the target scores, which can either be probability estimates of the positive class, confidence values, or binary decisions. Here is a small example of how to use the roc_curve function:	
>>> import numpy as np
>>> from sklearn.metrics import roc_curve
>>> y = np.array([1, 1, 2, 2])
>>> scores = np.array([0.1, 0.4, 0.35, 0.8])
>>> fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)
>>> fpr
array([ 0. ,  0.5,  0.5,  1. ])
>>> tpr
array([ 0.5,  0.5,  1. ,  1. ])
>>> thresholds
array([ 0.8 ,  0.4 ,  0.35,  0.1 ])	
	
777777777777777777777777777777777777777
The brier_score_loss function computes the Brier score for binary classes. Quoting Wikipedia:

    “The Brier score is a proper score function that measures the accuracy of probabilistic predictions. It is applicable to tasks in which predictions must assign probabilities to a set of mutually exclusive discrete outcomes.”

This function returns a score of the mean square difference between the actual outcome and the predicted probability of the possible outcome. The actual outcome has to be 1 or 0 (true or false), while the predicted probability of the actual outcome can be a value between 0 and 1.

The brier score loss is also between 0 to 1 and the lower the score (the mean square difference is smaller), the more accurate the prediction is. It can be thought of as a measure of the “calibration” of a set of probabilistic predictions.

BS = \frac{1}{N} \sum_{t=1}^{N}(f_t - o_t)^2

where : N is the total number of predictions, f_t is the predicted probablity of the actual outcome o_t.

http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration.html#sphx-glr-auto-examples-calibration-plot-calibration-py

Here is a small example of usage of this function::

>>> import numpy as np
>>> from sklearn.metrics import brier_score_loss
>>> y_true = np.array([0, 1, 1, 0])
>>> y_true_categorical = np.array(["spam", "ham", "ham", "spam"])
>>> y_prob = np.array([0.1, 0.9, 0.8, 0.4])
>>> y_pred = np.array([0, 1, 1, 0])
>>> brier_score_loss(y_true, y_prob)
0.055
>>> brier_score_loss(y_true, 1-y_prob, pos_label=0)
0.055
>>> brier_score_loss(y_true_categorical, y_prob, pos_label="ham")
0.055
>>> brier_score_loss(y_true, y_prob > 0.5)
0.0

def logloss (act, pred):
	""" vectorized computation of log loss """
	
	############################
	## sample usage
	##
	## pred = [1,0,1,0]
	## act = [1, 0, 1, 0]
	## print(logloss(act,pred))
	##
	############################
		
	epsilon = 1e-15
	pred = sp.maximum(epsilon, pred)
	pred = sp.minimum(1-epsilon, pred)
	
	# compute log loss func (vectorized)
	ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
	ll = ll * -1.0/len(act)
	return ll
	
def skl_logloss(size=1000):
	from sklearn.metrics import log_loss
	prediction = numpy.random.random(size=size)
	random_labels = numpy.random.choice(2, size=size)
	proba = numpy.ndarray((len(prediction), 2))
	proba[:, 0] = 1-prediction
	proba[:, 1] = prediction
	
	loss = metrics.LogLoss().fit(proba, y=random_labels, sample_weight=None)
	value = log_loss(random_labels, prediction)
	value2 = loss(random_labels, proba)
	
	print(value, value2)
	assert numpy.allclose(value, value2)
	
def nb_combiner():
	""" make predictions via nb_combiner classifier approach """
	## train_data >> numberic_variables >> missing values imputed >> guassian nb classifier >> nb combiner classifier >>
	##	>> calibrate predictions >> test_data >> prediciton



"""	
