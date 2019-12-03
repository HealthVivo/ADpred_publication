#!/bin/python

import os,sys
sys.path.append(os.path.abspath('../libraries/'))
from utils import *

# data is stored in sqlite3 DB
db = '../data/data_complete.db'

# load complete dataset  
positives, lp = load_data_from_sqlite(db, 'positives') # lp = length positives
negatives, ln = load_data_from_sqlite(db, 'negatives') # ln = length negatives
ln, lp = ln[0], lp[0] # avoid having problems with this later

# keep only sequences
positives, negatives = positives[:,0], negatives[:,0]

# Let me know how many samples do we have in each set 
print('positives: {} samples\nnegatives: {} samples'.format(lp, ln))

# split into train, test, validation
IDp = {} # inidices positives
IDn = {} # indices negatives

# store indices for train,test,valid in a dictionary
for (LEN,SET) in zip([lp, ln], [IDp, IDn]):
    
    # split the data into 80,10,10
    [_10, _90] = [int(LEN * i) for i in [0.1, 0.9]]
    
    # make indices 
    np.random.seed(2)
    idx = np.random.permutation( np.arange(LEN) )
    
    # define the sets
    SET['training'] = idx[:_90]
    #SET['validation'] = idx[_80:_80+_10]
    SET['test'] = idx[_90:]


from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, make_scorer, average_precision_score
from joblib import load

#model = load('../models/LogReg_dimers.joblib')
model = load('dipeptide_model_AUPRC.joblib')

X_train = np.vstack(get_dipeptide_frequencies(np.hstack([positives[IDp['training']], 
                                                        negatives[IDn['training']]])))
y_train = np.hstack([np.ones(len(IDp['training'])), 
                    np.zeros(len(IDn['training']))])



X_test = np.vstack(get_dipeptide_frequencies(np.hstack([negatives[IDn['test']], 
                                                        positives[IDp['test']]])))

y_test = np.hstack([np.zeros(len(IDn['test'])), 
                    np.ones(len(IDp['test']))])


#model = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(average_precision_score), max_iter=1000).fit(X_train,y_train)

roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
acc = np.sum(y_test == model.predict((X_test).astype(int))) / len(y_test)
print('linear model, single aa:\nAUROC={:.3f}\naccuracy={:.3f}'.format(roc, acc)) 

original_weights = model.coef_[0].copy()


window = int(sys.argv[1]) # I am writing this some time after I wrote this code.
                          # I think that this is to split the calculation into 40
                          # to paralellize the calculations.  
aurocs = np.zeros(2000)
accs = np.zeros(2000)
keys = []

combinations = itertools.combinations(np.arange(400), r=2)

for n,i in enumerate(combinations):
    if n < window*2000 or n >= window*2000+2000: continue # do the work for the 1000 selected ONLY  
    # flip coefficients
    j=np.array(i)
    tmp = model.coef_[0][j]
    model.coef_[0][j] = tmp[::-1]

    # save performances with flipped coefficients
    m = n-window*2000
    aurocs[m] = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])
    accs[m] = np.sum(y_test == model.predict((X_test).astype(int))) / len(y_test)
    keys.append(','.join([dipeptides[i[0]], dipeptides[i[1], str(aurocs[m]])))    

    # revert changes
    model.coef_[0] = original_weights.copy()

auroc_filename = '../notebooks/flips/aurocs_flip_coeff_' + str(window) + '.npy'
accs_filename = '../notebooks/flips/accs_flip_coeff_' + str(window) + '.npy'

report = 'report_' + str(window) + '.csv'

np.save(auroc_filename, aurocs)
np.save(accs_filename, accs)
