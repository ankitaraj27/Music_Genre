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
from sklearn import tree
from sklearn.preprocessing import normalize as norm
genre_list = GENRE_LIST



def train_model(X, Y, name, plot=False):
   
    labels = np.unique(Y)
    
    pr_scores = defaultdict(list)
    
    precisions, recalls, thresholds = defaultdict(list), defaultdict(list), defaultdict(list)

    roc_scores = defaultdict(list)
    tprs = defaultdict(list)
    fprs = defaultdict(list)

    clfs = []  # for the median
    
    cms = []
    
    #X = norm(X, norm='l2', axis=1, copy=True, return_norm=True)
    print len(X)
    print len(X[0])
    print X[0]
    #for i in range(500):
    X_train= []
    y_train=(Y)
    max_len = 0
    i=0
    for i in range(500):
        if len(X[i]) > max_len :
            max_len = len(X[i])
    i=0
    for i in range(500):
        if len(X[i]) < max_len:
            P=[]
            P=np.zeros(max_len-len(X[i]))
            v=[]
            v=np.concatenate([np.array(X[i]),np.array(P)])
            X_train.append(v)
        else :
            X_train.append(X[i])

    
    print len(X_train)
    print y_train

  
    maximum=max_len
    #clf = LogisticRegression()
    #clf = svm.SVC(decision_function_shape='ovr',cache_size=500, degree=3, gamma='auto',coef0=0.5, kernel='rbf',probability=True)
    #clf = svm.NuSVC(nu=0.543, kernel='rbf', degree=3, gamma=0.1, coef0=0.0, shrinking=True, probability=True, tol=0.001, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)
##        clf = svm.LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
##     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
##     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
##     verbose=0)
   
##        clf = RFECV(model,13)
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)

    
    print max_len
    train_score = clf.score(X_train, y_train)
    #test_score = clf.score(X_test, y_test)
    #scores.append(test_score)
##        
##    y_pred = clf.predict(X_test)
##    cm = confusion_matrix(y_test, y_pred)
##    cms.append(cm)

    
##    for label in labels:
##        #y_label_test = np.asarray(y_test == label, dtype=int)
##        y_label_test = np.asarray(y_test == label,dtype=int)
##        #proba = clf.predict_proba(X_test)
##        proba = clf.decision_function(X_test)
##        #print proba
##        proba_label = proba[:, label]
##        #print proba_label
##        fpr, tpr, roc_thresholds = roc_curve(y_label_test, proba_label)
##        roc_scores[label].append(auc(fpr, tpr))
##        tprs[label].append(tpr)
##        fprs[label].append(fpr)

##    if plot:
##        for label in labels:
##            scores_to_sort = roc_scores[label]
##            median = np.argsort(scores_to_sort)[len(scores_to_sort) / 2]
##            desc = "%s %s" % (name, genre_list[label])
##            plot_roc_curves(roc_scores[label][median], desc, tprs[label][median],fprs[label][median], label='%s vs rest' % genre_list[label])
##
##    all_pr_scores = np.asarray(pr_scores.values()).flatten()
##    summary = (np.mean(scores), np.std(scores), np.mean(all_pr_scores), np.std(all_pr_scores))
    #print("%.3f\t%.3f\t%.3f\t%.3f\t" % summary)

    #save the trained model to disk
    joblib.dump(clf, 'saved_model\\model_ceps.pkl')
    #joblib.dump(clf1, 'saved_model\\model_ceps1.pkl')
##    print train_errors
##    print test_errors

    return max_len

def length():
    return 53118

if __name__ == "__main__":
    start = timeit.default_timer()
    print
    print " Starting classification \n"
    print " Classification running ... \n" 
    X, y = read_ceps(genre_list)
    maximum = train_model(X, y, "ceps", plot=True)
##    cm_avg = np.mean(cms, axis=0)
##    cm_norm = cm_avg / np.sum(cm_avg, axis=0)
    print " Classification finished \n"
    stop = timeit.default_timer()
    print " Total time taken (s) = ", (stop - start)
##    print "\n Plotting confusion matrix ... \n"
##    plot_confusion_matrix(cm_norm, genre_list, "ceps","CEPS classifier - Confusion matrix")
##    print " All Done\n"
    #print " See plots in 'graphs' directory \n"
