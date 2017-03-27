import os
import timeit
import numpy as np
from collections import defaultdict
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from utils import GENRE_LIST, GENRE_DIR, TEST_DIR
from utils import plot_confusion_matrix, plot_roc_curves
from ceps import read_ceps, read_ceps_test
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV

genre_list = GENRE_LIST

def train_model(X, Y, name, plot=False):
    """
        train_model(vector, vector, name[, plot=False])
        
        Trains and saves model to disk.
    """
    labels = np.unique(Y)
    print labels
    print X
    cv = ShuffleSplit(n=len(X),test_size=0.10, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
    #print cv
    train_errors = []
    test_errors = []
    train_errors1 = []
    test_errors1 = []

    scores = []
    scores1 = []
    pr_scores = defaultdict(list)
    #print pr_scores
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = []  # for the median
    
    cms = []
    ##    train=[i for i in range(500)]
##    test=[i for i in range(500)]
##    cv=[i for i in range(10)]
    for train, test in cv:
    #cv=[1]
    #for i in cv:
        #print "train huaa ",test
        X_train, y_train = X[train], Y[train]
        X_test, y_test = X[test], Y[test]
        #print "\nlen of X ",len(X_train),"\n array \n",X_train
        #print "\nlen of y ",len(y_train),"\n array y \n",y_train
        #clf = LogisticRegression()
##        clf = svm.SVC(decision_function_shape='ovo', degree=3, gamma='auto',coef0=0.5, kernel='rbf',probability=True)
##        clf = svm.NuSVC(nu=0.543, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
##        clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
##     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
##     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
##     verbose=0)
##        clf = RFECV(model,13)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=0)
        clf = clf.fit(X_train, y_train)

        
        #clf.fit(X_train, y_train)
        #clf1.fit(X_train, y_train)
        clfs.append(clf)
        #clfs1.append(clf)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        scores.append(test_score)
        print "\ntrain score ",train_score
        print "\ntest score   ",test_score
        train_errors.append(1 - train_score)
        test_errors.append(1 - test_score)

        
        
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cms.append(cm)

        
        for label in labels:
            #y_label_test = np.asarray(y_test == label, dtype=int)
            y_label_test = np.asarray(y_test == label,dtype=int)
            proba = clf.predict_proba(X_test)
            #proba = clf.decision_function(X_test)
            #print proba
            proba_label = proba[:, label]
            #print proba_label
            fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
            roc_scores[label].append(auc(fpr, tpr))
            tprs[label].append(tpr)
            fprs[label].append(fpr)

    if plot:
        for label in labels:
            scores_to_sort = roc_scores[label]
            median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
            desc = "%s %s" % (name, genre_list[label])
            plot_roc_curves(roc_scores[label][median], desc, tprs[label][median],fprs[label][median], label='%s vs rest' % genre_list[label])

    all_pr_scores = np.asarray(pr_scores.values()).flatten()
    summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    #print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    #save the trained model to disk
    joblib.dump(clf, 'saved_model\\model_ceps.pkl')
    #joblib.dump(clf1, 'saved_model\\model_ceps1.pkl')
    print train_errors
    print test_errors

    return np.mean(train_errors), np.mean(test_errors), np.asarray(cms)


if __name__ == "__main__":
    start = timeit.default_timer()
    print
    print " Starting classification \n"
    print " Classification running ... \n" 
    X, y = read_ceps(genre_list)
    train_avg, test_avg, cms = train_model(X, y, "ceps", plot=True)
    cm_avg = np.mean(cms, axis=0)
    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print " Classification finished \n"
    stop = timeit.default_timer()
    print " Total time taken (s) = ", (stop - start)
    print "\n Plotting confusion matrix ... \n"
    plot_confusion_matrix(cm_norm, genre_list, "ceps","CEPS classifier - Confusion matrix")
    print " All Done\n"
    print " See plots in 'graphs' directory \n"
