from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
from tpot import TPOTRegressor, TPOTClassifier
import pandas as pd
import numpy as np

if __name__ == '__main__':
	train = pd.read_csv('./numerai_training_data.csv')
	test = pd.read_csv('./numerai_tournament_data.csv')

	y_train = np.asarray(train['target'])					# target to predict
	X_train = np.asarray(train.drop('target', axis=1))		# training data

	y_test = np.asarray(test['t_id'])
	X_test = np.asarray(test.drop('t_id', axis=1))

	
	estim = HyperoptEstimator( classifier=any_classifier('clf'),  
								algo=tpe.suggest, trial_timeout=300)

	estim.fit( X_train, y_train )

	print( estim.score( X_test, y_test ) )
	print( estim.best_model() )

	#pipeline_optimizer = TPOTClassifier(generations=5, population_size=20, num_cv_folds=5, random_state=42, verbosity=2)
	#pipeline_optimizer.fit(X, y)
	#result = pipeline_optimizer.score(x_test, t_id)
	#print(result)

	#pipeline_optimizer.export('tpot_exported_pipeline.py')
