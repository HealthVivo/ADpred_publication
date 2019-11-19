#!/bin/python

# import libraries used throughout this notebook
import os,sys, copy, tarfile, pickle, joblib

sys.path.append(os.path.abspath('../libraries/'))
#from summary_utils_v2B import *
from utilities import *

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import chi2, linregress, spearmanr
from sklearn.metrics import precision_recall_curve, average_precision_score, log_loss, roc_auc_score, make_scorer
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout, Activation
from keras.models import Model, model_from_json
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf


# data is stored in sqlite3 DB
db = analysis_home+'/data/data_complete.db'
print('loading data from ',db)
# define seeds for reproducibility. However, these could be changed with minor effects on the results
seeds = np.array([94499, 43481, 35516, 68937, 27841, 11428, 12385, 73607, 76537, 56232])

# load complete dataset  
positives, lp = load_data_from_sqlite(db, 'positives') # lp = length positives
negatives, ln = load_data_from_sqlite(db, 'negatives') # ln = length negatives
ln, lp = ln[0], lp[0] # avoid having problems with this later


# get a table with counts and scores
def load_data_for_scores(database, table):
    '''
        function extract data from a database file.
        INPUT: database filename
               table= 'positives' or 'negatives'. This argument is a string!
        OUTPUT: data array with seq and ss and labels
                length of the table
    '''
    # create connection to database
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    # extract positives and negatives into different variables
    cursor.execute('SELECT seq,bg,bin1,bin2,bin3,bin4 FROM ' + table)
    samples = np.array( cursor.fetchall() )

    # get length of the table 
    cursor.execute('SELECT COUNT(*) FROM ' + table)
    length = cursor.fetchone()

    # return data in a table with named columns and replace the indices assigned by sqlite
    return samples, length


def get_enrichment_scores_table():
    positives, lp = load_data_for_scores(db, 'positives') # lp = length positives
    negatives, ln = load_data_for_scores(db, 'negatives')
    scores = pd.DataFrame(np.vstack([positives, negatives]))
    scores.columns = ['sequence', 'bg','bin1','bin2','bin3','bin4']
    scores.set_index('sequence', inplace=True)
    scores = scores.astype(int)

    return scores




# try with positives whose sequence is found ONLY in bins2,3,4
scores = get_enrichment_scores_table()
pidx = scores[(scores.bin2>0) | (scores.bin3>0) | (scores.bin4>0)].index
positives = np.array([i for i in positives if i[0] in pidx])
lp = len(positives)


# Let me know how many samples do we have in each set 
if '--verbose' in sys.argv:
    print('positives: {} samples\nnegatives: {} samples'.format(lp, ln))


# For the following analysis 1- and 2-mers and NN, I will use the same validation set and will train with the same train+test sets scrumbling them often.
# Here are the permutated indices
np.random.seed(seeds[0])
p = np.random.permutation( np.arange(lp) )
np.random.seed(seeds[0])
n = np.random.permutation( np.arange(ln) ) 

# These are the exact sets used in the manuscript. This is ment for reproducibility issues and can be changed by the user.
# define validation set as 10% of the complete dataset
posit10 = p.shape[0]//10
negat10 = posit10 #n.shape[0]//10

# define inidices for split data
ptest = p[:posit10]
ntest = n[:posit10] 

ptrain = p[posit10:] #-posit10]
ntrain = n[posit10:] #-negat10]

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc


def acc(y1,y2):

    if len(y1)!=len(y2):
        print('error')
        return
    
    y1,y2 = y1>=0.5, y2>=0.5

    return np.sum(y1==y2)/len(y1)


def scramble_indices(X,y, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(np.arange(len(X)))
    return X[idx], y[idx]


def open_prediction(predictionFile):
    tar = tarfile.open(predictionFile, "r:gz")
    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            content = f.read().decode('ascii').split('\n')
    return content

def open_iupred(iupredFile):
    content = open_prediction(iupredFile)
    iupred_short={}
    for i in range(0, len(content)-1,4):
        try:
            [k,v] = content[i+1:i+3]
            iupred_short[k] = v
        except Exception as e:
            print(str(e))
    return iupred_short


def make_ohe(seq, aa):
    '''
        one hot encode a tensor
        INPUT: sequence and list of all possible values in the sequence(aa, ss, etc..)
        OUTPUT: ohe tensor
    '''
    # initialize results tensor
    res = np.zeros(shape = (len(seq), len(aa)))
    # get positions (that correspond to aa) to assign 1 along the 30mer
    for position, residue in enumerate(seq):
        res[position][aa.index(residue)]=1
    return res

# all models
def make_model(model_features, weights_filename=None):

    if model_features == 'model_seq': input_shape =  (30, 20, 1)
    elif model_features == 'model_seq_iupred': input_shape = (30, 22, 1)
    elif model_features == 'model_seq_ss': input_shape = (30,23,1)
    elif model_features == 'model_seq_iupred_ss': input_shape = (30, 25, 1)

    K.clear_session()
    inputs = Input(shape=input_shape)
    x = Conv2D(input_shape[0]-1, (4,input_shape[1]), activation=softplus)(inputs) # initializers are default in all layers.
    x = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)
    x = Dense(300, activation=softplus)(x)
    x = Dropout(0.3)(x)
    x = Dense(30, activation=softplus, kernel_regularizer=regularizers.l2(0.01))(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])

    if isinstance(weights_filename, str):
        with open(weights_filename, 'rb') as f:
            w = pickle.load(f)
        model.set_weights(w)

    return model


def train_model(model, n_train, p_train, batch_size=250):

    #ptrain, p_filename = store_data_numpy(p_train)   
    #ntrain, n_filename = store_data_numpy(n_train)

    __10 = len(p_train)//10

    pvalid = p_train[:__10]
    ptrain = p_train[__10:]
    nvalid = n_train[:__10]
    ntrain = n_train[__10:]

    X_valid = np.vstack([pvalid, nvalid])
    y_valid = np.hstack([np.ones(__10), np.zeros(__10)])
    X_valid, y_valid = scramble_indices(X_valid, y_valid)

    model.fit_generator(epoch_generator(ntrain, ptrain, batch_size=batch_size),
                            validation_data=(X_valid.reshape(np.insert(X_valid.shape,3,1)), y_valid),
                            steps_per_epoch=int(len(ptrain)/batch_size),
                            epochs=100,
                            callbacks=[EarlyStopping(patience=3)],
                            verbose=0)
    return model


##########################################################################


# Collect iupred results
iupred_short_file = analysis_home + '/data/30mers_ALL.IUPred_short_disorder.fasta.tar.gz'
iupred_short = open_iupred(iupred_short_file)
iupred_short_arr_pos = np.array([iupred_short[i] for i in positives[:,0]])
iupred_short_arr_neg = np.array([iupred_short[i] for i in negatives[:,0]])
ohe_p_disS = np.array([make_ohe(i,['-','D']) for i in iupred_short_arr_pos])
ohe_n_disS = np.array([make_ohe(i,['-','D']) for i in iupred_short_arr_neg])

#========================================================================================================================================

#####################################################################
# complete set for SD eccept for init 0 where held out test is used #
#####################################################################

# first have a pool of seq, ss, iupred to merge upon need.
seqp = np.array([make_ohe(i,aa) for i in positives[:,0]]).astype(np.int8)
seqn = np.array([make_ohe(i,aa) for i in negatives[:,0]]).astype(np.int8)
ssp = np.array([make_ohe(i,ss) for i in positives[:,1]]).astype(np.int8)
ssn = np.array([make_ohe(i,ss) for i in negatives[:,1]]).astype(np.int8)
iupredp = ohe_p_disS.astype(np.int8)
iupredn = ohe_n_disS.astype(np.int8)

# define the dictionaries to store the data
def define_dict():
    d = {
        'single':[],
        'dipept':[],
        'ADPred':[],
        'NNseq':[],
        'NNseq_iupred':[],
        'NNseq_iupred_ss':[]
    }
    return d

results = {
    'balanced'  : {'auroc':define_dict(), 'auprc':define_dict(), 'acc':define_dict()},
    'unbalanced': {'auroc':define_dict(), 'auprc':define_dict(), 'acc':define_dict()},
}

# define indices for balanced and for unbalanced 
rangep, rangen = len(positives), len(negatives)
_10p = rangep // 10
_10n = rangen // 10


idxp = np.random.permutation(rangep)
idxn = np.random.permutation(rangen)

for init in range(10):

    for (i,j) in zip([_10p, _10n],['balanced', 'unbalanced']):

        print(' iteration: {}\ntype{}'.format(init, j))

        # generate indices for train, test split        
        idxp10 = idxp[_10p*init:_10p*(init+1)]
        idxn10 = idxn[i*init:i*(init+1)]
        xp,xn = positives[idxp10], negatives[idxn10]
        x_test = np.vstack([xp, xn])
        y_test = np.hstack([np.ones(len(idxp10)), np.zeros(len(idxn10))])
        x_test, y_test = scramble_indices(x_test, y_test)
        del xp,xn

        idxp90 = np.hstack([ idxp[0:_10p*init], idxp[_10p*(init+1):] ])
        idxn90 = np.hstack([ idxn[0:i*init], idxn[i*(init+1):] ])
        xp,xn = positives[idxp90], negatives[idxn90]
        x_train = np.vstack([xp,xn])
        y_train = np.hstack([np.ones(len(idxp90)), np.zeros(len(idxn90))])
        x_train, y_train = scramble_indices(x_train, y_train)
        del xp,xn

        # train all models on training set and test them on test set. Both balanced and Unbalanced included            

        if 'single' in sys.argv:
            # single
            Xtest  = get_aa_frequencies(x_test[:,0]);  Xtest  = np.hstack(Xtest.T).T
            Xtrain = get_aa_frequencies(x_train[:,0]); Xtrain = np.hstack(Xtrain.T).T
            model_single_aa_composition = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score)).fit(Xtrain,y_train)
            y_hat = model_single_aa_composition.predict_proba(Xtest)[:,1]
            results[j]['auroc']['single'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['single'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['single'].append(acc(y_test, y_hat))
            del Xtest, Xtrain, model_single_aa_composition, y_hat
        
        elif 'dipeptides' in sys.argv:
            #dipept
            Xtest  = get_dipeptide_frequencies(x_test[:,0]); Xtest  = np.vstack(Xtest)
            Xtrain = get_dipeptide_frequencies(x_train[:,0]); Xtrain = np.vstack(Xtrain)
            model_dipeptides = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score), max_iter=700).fit(Xtrain,y_train)
            y_hat = model_dipeptides.predict_proba(np.vstack(Xtest))[:,1]
            results[j]['auroc']['dipept'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['dipept'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['dipept'].append(acc(y_test, y_hat))
            del Xtest, Xtrain, model_dipeptides, y_hat

        elif 'seq' in sys.argv and not 'ss' in sys.argv and not 'iupred' in sys.argv:
            # NN sequence only
            Xtest  = np.vstack([seqp[idxp10], seqn[idxn10]])
            Xtest  = Xtest.reshape(np.append(Xtest.shape,1)).astype(np.int8)
            NN_seq = make_model('model_seq') #, analysis_home+'/models/new_train_fix_split_and_redefine_positives/balanced-testweigths.pkl')
            NN_seq = train_model(NN_seq, seqn[idxn90], seqp[idxp90])
            y_hat = NN_seq.predict(Xtest)
            results[j]['auroc']['NNseq'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['NNseq'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['NNseq'].append(acc(y_test, y_hat))
            del Xtest, NN_seq, y_hat

        elif 'seq' in sys.argv and 'ss' in sys.argv and 'iupred' not in sys.argv:
            # NN sequence + ss 
            a = np.vstack([seqp.T, ssp.T]).T
            b = np.vstack([seqn.T, ssn.T]).T
            Xtest  = np.vstack([a[idxp10], b[idxn10]])
            NN_seq_ss = make_model('model_seq_ss') #ADPred = make_ADPred()
            NN_seq_ss = train_model(NN_seq_ss, b[idxn90], a[idxp90])
            Xtest = Xtest.reshape(np.append(Xtest.shape,1)).astype(np.int8)
            y_hat = NN_seq_ss.predict(Xtest)
            results[j]['auroc']['ADPred'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['ADPred'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['ADPred'].append(acc(y_test, y_hat))
            del a,b, Xtest, NN_seq_sa, y_hats

        elif 'seq' in sys.argv and not 'ss' in sys.argv and 'iupred' in sys.argv:
            # NN sequence + iupred
            a = np.vstack([seqp.T, iupredp.T]).T
            b = np.vstack([seqn.T, iupredn.T]).T
            Xtest = np.vstack([a[idxp10],b[idxn10]])
            Xtest = Xtest.reshape(np.append(Xtest.shape,1)).astype(np.int8)
            NN_seq_iupred = make_model('model_seq_iupred') #, analysis_home+'/models/new_train_fix_split_and_redefine_positives/iupred_short_--balanced-testweigths.pkl')
            NN_seq_iupred = train_model(NN_seq_iupred, b[idxn90], a[idxp90])
            y_hat = NN_seq_iupred.predict(Xtest)
            results[j]['auroc']['NNseq_iupred'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['NNseq_iupred'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['NNseq_iupred'].append(acc(y_test, y_hat))

        elif 'seq' in sys.argv and 'ss' in sys.argv and 'iupred' in sys.argv:
            # NN sequence + iupred + ss
            a = np.vstack([seqp.T, iupredp.T, ssp.T]).T
            b = np.vstack([seqn.T, iupredn.T, ssn.T]).T
            Xtest = np.vstack([a[idxp10],b[idxn10]])
            Xtest = Xtest.reshape(np.append(Xtest.shape,1)).astype(np.int8)
            NN_seq_iupred_ss = make_model('model_seq_iupred_ss') #, analysis_home+'/models/new_train_fix_split_and_redefine_positives/ss_iupred_short_--balanced-testweigths.pkl')
            NN_seq_iupred_ss = train_model(NN_seq_iupred_ss, b[idxn90], a[idxp90])
            y_hat = NN_seq_iupred_ss.predict(Xtest)
            results[j]['auroc']['NNseq_iupred_ss'].append(roc_auc_score(y_test, y_hat))
            results[j]['auprc']['NNseq_iupred_ss'].append(average_precision_score(y_test, y_hat))
            results[j]['acc']['NNseq_iupred_ss'].append(acc(y_test, y_hat))


# save pickle with results
with open('figure4C_'+'_'.join(sys.argv[1:]),'wb') as f: pickle.dump(results, f)
