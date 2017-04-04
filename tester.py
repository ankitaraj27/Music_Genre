import os
import timeit
import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from classifier_new import length
from utils import plot_roc_curves, plot_confusion_matrix, GENRE_DIR, GENRE_LIST, TEST_DIR

from ceps import read_ceps, create_ceps_test, read_ceps_test

from pydub import AudioSegment

genre_list = GENRE_LIST

clf = None

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the classifier script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def test_model_on_single_file(file_path):
    clf = joblib.load("C:\Users\RAJ\Desktop\matlab\genreXpose-master\genreXpose\saved_model\model_ceps.pkl")
    #clf1 = joblib.load("C:\Users\RAJ\Desktop\matlab\genreXpose-master\genreXpose\saved_model\model_ceps1.pkl")
    link = create_ceps_test(file_path)
    X_old, y = read_ceps_test(link+".npy")
    X_old = X_old[0]
    #probs = clf.decision_function(X)
    l = length()
    print l
    if l>len(X_old):
        zero = np.zeros(l-len(X_old))
        X=np.concatenate([np.array(X_old),np.array(zero)])
    else :
        X=X_old
    
    value = clf.predict(X)
    probs = clf.predict_proba(X)
    print probs
    #value1 = clf1.predict(X)
    print "\t".join(str(x) for x in traverse)
    print "\t".join(str("%.3f" % x) for x in (probs[0]))
    probs=probs[0]
    #probs=1-probs
    max_prob = max(probs)
    #print max_prob
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    
    #print max_prob_index
    predicted_genre = traverse[max_prob_index]
    pg = traverse[value]

    print "predicted genre = ", pg, " by predict by min= ",predicted_genre
    return predicted_genre

if __name__ == "__main__":

    global traverse
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection(set(GENRE_LIST)))
        break
    fi=[]
    test_file = "C:/Users/RAJ/Desktop/matlab/test_data"
    for f in os.listdir(test_file):
        path=test_file+'/'+f
        print "\n\nfor file "+f
        predicted_genre = test_model_on_single_file(path)
    # should predict genre as "ROCK"
    #predicted_genre = test_model_on_single_file(test_file)
    
