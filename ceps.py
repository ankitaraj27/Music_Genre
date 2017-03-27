import os
import glob
import sys
import numpy as np
import scipy
import scipy.io.wavfile
from scikits.talkbox.features import mfcc
from scipy.cluster.vq import whiten,vq
from sklearn.decomposition import PCA
from utils import GENRE_DIR, CHART_DIR, GENRE_LIST
from mfcc import my_mfcc

def write_ceps(ceps, fn):
    """
    Write the MFCC to separate files to speed up processing.
    """
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".ceps"
    np.save(data_fn, ceps)
    print "Written ", data_fn


def create_ceps(fn):
    """
        Creates the MFCC features. 
    """
    #print 'aaya'
##    sample_rate, X = scipy.io.wavfile.read(fn)
##    X[X==0]=1
##    ceps, mspec, spec = mfcc(X,sample_rate)
    #ceps=whiten(ceps)
    #final=vq(ceps,new_arr)
    ceps = my_mfcc(fn)
    #pca=PCA(n_components=5)
    #fit=pca.fit(ceps)
    #ceps=fit.components_
    
    write_ceps(ceps, fn)


def read_ceps(genre_list, base_dir=GENRE_DIR):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    i=0;
    for label, genre in enumerate(genre_list):
        for fn in glob.glob(os.path.join(base_dir, genre, "*.ceps.npy")):
            ceps = np.load(fn)
            #X = np.load(fn)
            num_ceps = len(ceps)
            #X = ceps.mean_
            #
            X.append(np.mean(ceps, axis=0))
            y.append(label)
            i=i+1;
    
    #print i
    #print X
    return np.array(X), np.array(y)


def create_ceps_test(fn):
    """
        Creates the MFCC features from the test files,
        saves them to disk, and returns the saved file name.
    """
    #print fn
##    sample_rate, X = scipy.io.wavfile.read(fn)
##    X[X==0]=1
##    #np.nan_to_num(X)
##    ceps, mspec, spec = mfcc(X,sample_rate)
##    #ceps = whiten(ceps)
    base_fn, ext = os.path.splitext(fn)
    print base_fn
    ceps = my_mfcc(fn)
    data_fn = base_fn + ".ceps"
    #final=vq(ceps,new_arr)
    #pca=PCA(n_components=5)
    #fit=pca.fit(ceps)
    #ceps=fit.components_
    np.save(data_fn, ceps)
    print "Written ", data_fn
    return data_fn


def read_ceps_test(test_file):
    """
        Reads the MFCC features from disk and
        returns them in a numpy array.
    """
    X = []
    y = []
    ceps = np.load(test_file)
    #X = np.load(test_file)
    num_ceps = len(ceps)
    X.append(np.mean(ceps, axis=0))
    #X = ceps.mean_
    return np.array(X), np.array(y)


if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    #print traverse
    print "Working with these genres --> ", traverse
    print "Starting ceps generation"     
    for subdir, dirs, files in os.walk(GENRE_DIR):
        for files1 in files:
            path = subdir+'/'+files1
            if path.endswith("wav"):
                tmp = subdir[subdir.rfind('\\',0)+1:]
                                
                if tmp in traverse:
                    create_ceps(path)
            
                    
    stop = timeit.default_timer()
    print "Total ceps generation and feature writing time (s) = ", (stop - start) 
