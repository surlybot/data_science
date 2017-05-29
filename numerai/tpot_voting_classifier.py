from tpot import TPOTClassifier

import pandas as pd
import numpy as np
import os
from time import time

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.utils.validation import check_is_fitted

chunksize = 1000

# list of classifiers that will make up our VotingClassifier, should end up num_chunks in length
classifiers = []

training_filename = 'numerai_training_data.csv'
tournament_filename = 'numerai_tournament_data.csv'

#####################################################################################
# Overloaded?[redefined?] SciKit VotingClassifier to use prefit models rather than requiring a refit
# Found at https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4
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
########################################################################################
    
########################################################################################
# function:     process()
# parameters:   pandas.DataFrame() df
# description:  take a dataframe with a target column, assume the rest are features, run TPOT to get a good pipeline and then calibrated classifier on the fitted_pipeline
#               append fitted, calibrated pipeline to a list we'll use later for our VotingClassifier
def process(df):
    global classifiers
    df.rename(columns={'target':'class'}, inplace=True)
    features = df.drop(['class'],axis=1)
    target = df['class']

    training_features, testing_features, training_classes, testing_classes = train_test_split(features, target, random_state=42, train_size=0.85, test_size=0.15)

    eighty_p = int(len(training_features) * .8)

    # Up to eighty percent of this will be allotted for training
    X_train, y_train = training_features[:eighty_p], training_classes[:eighty_p]

    # Remaining twenty percent will go to validation
    X_valid, y_valid = training_features[eighty_p:], training_classes[eighty_p:]

    # valids contains all 100%
    X_train_valid, y_train_valid = training_features[:], training_classes[:]
    X_test, y_test = testing_features, testing_classes
    ################################################################

    ################################################################
    # Cross Validate, Calibrate, collaborate and listen, then score
    # Init tpot classifier and fit on X_train, y_train (the 80%)
    tpot = TPOTClassifier(generations=5, population_size=20, scoring='log_loss', verbosity=2)
    print(type(X_train))
    print(type(y_train))
    tpot.fit(X_train.to_dense(), y_train)
    # Get clf object
    clf = tpot._fitted_pipeline
    # Feed CalibratedClassifier with clf object
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    try:
        # Fit CalibratedClassifier on 20% for, i dunno, calibration
        sig_clf.fit(X_valid.to_dense(), y_valid)
        # Gimmi probabilities, cuz I need those
        sig_clf_probs = sig_clf.predict_proba(X_test.to_dense())
        sig_score = log_loss(y_test, sig_clf_probs)
        print('calibrated score log loss:   {}'.format(sig_score))
        #########################################

        # Put our calibrated classifier in the list for the voting classifier
        classifiers.append((str(time()), sig_clf))
        # TODO Perhaps rank these by their scores, and weight them by their ranks?
    except:
        print('could not calibrate, using best model without calibration\nthis can happen with xgboost')
        classifiers.append((str(time()), clf))
############################################################################################

if __name__ == '__main__':

    ##################################################################
    # Step 1 - Read in data and chunk it
    #
    for chunk in pd.read_csv(training_filename, chunksize=chunksize, skiprows=0):
        process(chunk)
    ##################################################################

    ##################################################################
    # Step 2 build a big voting classifier with all of our classifiers
    # For now, this shall be a democracy! TODO: In the future, we may want to weight by score or something else
    eclf = VotingClassifier(estimators=[(i,j) for i,j in classifiers],
                            voting='soft')
    ##################################################################


    ##################################################################
    # Step 5 read in tournament/prod data
    X = pd.read_csv(tournament_filename, skiprows=0)
    t_id = X['t_id']
    X = X.drop(['t_id'],axis=1)
    ##################################################################

    
    ##################################################################
    # Step 6 - Get predictions!
    out = eclf.predict_proba(X.to_dense())
    out = pd.DataFrame(out)
    out['t_id'] = t_id
    out.to_csv('calibrated_voter.csv', index=False)
    ###################################################################

