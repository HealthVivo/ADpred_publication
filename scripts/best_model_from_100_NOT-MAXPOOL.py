#!/bin/python
import os,sys
sys.path.append(os.path.abspath('../libraries/'))
from utils import *
from joblib import dump, load
import tarfile


ARGS = sys.argv #['ss','iupred_short'] #ARGS

# data is stored in sqlite3 DB
db = '../data/data_complete.db'
print('loading data from ',db)
# define seeds for reproducibility. However, these could be changed with minor effects on the results
seeds = np.array([94499, 43481, 35516, 68937, 27841, 11428, 12385, 73607, 76537, 56232])

# load complete dataset  
positives, lp = load_data_from_sqlite(db, 'positives') # lp = length positives
negatives, ln = load_data_from_sqlite(db, 'negatives') # ln = length negatives
ln, lp = ln[0], lp[0] # avoid having problems with this later

# Let me know how many samples do we have in each set 
if '--verbose' in sys.argv:
    print('positives: {} samples\nnegatives: {} samples'.format(lp, ln))


# For the following analysis 1- and 2-mers and NN, I will use the same validation set and will train with the same train+test sets scrumbling them often.
# Here are the permutated indices
np.random.seed(seeds[0])
p = np.random.permutation( np.arange(lp) )
np.random.seed(seeds[0])
n = np.random.permutation( np.arange(ln) )

# define validation set as 10% of the complete dataset
posit10 = lp//10
negat10 = posit10 #n.shape[0]//10

# define indices for positive and negative training + testing sets
pvalid = p[-posit10:]
nvalid = n[-negat10:]

# define inidices for split data
ptest = p[:posit10]
ntest = n[:posit10] 

ptrain = p[posit10:-posit10]
ntrain = n[negat10:-negat10]


np.save('ptest.npy',ptest)


############################################
#### include structural predicted data  ####
############################################

iupred_short_file = '../data/30mers_ALL.IUPred_short_disorder.fasta.tar.gz'
iupred_long_file  = '../data/30mers_ALL.IUPred_short_disorder.fasta.tar.gz'
netsurf_file = '../data/30mers_ALL.netsurfp_ss_acc.fasta.tar.gz'

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

def get_charge(seq):
    def sc(res):
        if res in ['K','R']: return 1
        elif res in ['H']: return 0.5
        elif res in ['D','E']: return -1
        else: return 0
    
    return np.array([sc(i) for i in seq])

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

###################################################################
#### include features indicated in the arguments of the script ####
###################################################################
ohe_p_aa = np.array([make_ohe(i,aa) for i in positives[:,0]])
ohe_n_aa = np.array([make_ohe(i,aa) for i in negatives[:,0]])

# list of features to include only features requested in the arguments passed to the scripts
features_positives = [ohe_p_aa.T]
features_negatives = [ohe_n_aa.T]

if 'iupred_short' in ARGS:
    iupred_short = open_iupred(iupred_short_file)
    iupred_short_arr_pos = np.array([iupred_short[i] for i in positives[:,0]])
    iupred_short_arr_neg = np.array([iupred_short[i] for i in negatives[:,0]])
    ohe_p_disS = np.array([make_ohe(i,['-','D']) for i in iupred_short_arr_pos])
    ohe_n_disS = np.array([make_ohe(i,['-','D']) for i in iupred_short_arr_neg])
    features_positives.append(ohe_p_disS.T)
    features_negatives.append(ohe_n_disS.T)

if 'iupred_long' in ARGS:
    iupred_long = open_iupred(iupred_long_file)
    iupred_long_arr_pos = np.array([iupred_long[i] for i in positives[:,0]])
    iupred_long_arr_neg = np.array([iupred_long[i] for i in negatives[:,0]])
    ohe_p_disL = np.array([make_ohe(i,['-','D']) for i in iupred_long_arr_pos])
    ohe_n_disL = np.array([make_ohe(i,['-','D']) for i in iupred_long_arr_neg])
    features_positives.append(ohe_p_disL.T)
    features_negatives.append(ohe_n_disL.T)

if 'ss' in ARGS:
    ohe_p_ss = np.array([make_ohe(i,ss) for i in positives[:,1]])
    ohe_n_ss = np.array([make_ohe(i,ss) for i in negatives[:,1]])
    features_positives.append(ohe_p_ss.T)
    features_negatives.append(ohe_n_ss.T)


# positives
p = np.vstack(features_positives).T
for i in features_positives:
    del i

p_train, p_filename = store_data_numpy(p[ptrain])
p_valid = p[pvalid]
p_test = p[ptest]
del p


# negatives 
n = np.vstack(features_negatives).T
for i in features_negatives:
    del i

n_train, n_filename = store_data_numpy(n[ntrain])
n_valid = n[nvalid]
n_test = n[ntest]
del n


# make test and valid complete sets with labels
X_test = np.vstack([p_test, n_test])
y_test = np.hstack([np.ones(p_test.shape[0]), np.zeros(n_test.shape[0])])

X_valid = np.vstack([p_valid, n_valid])
y_valid = np.hstack([np.ones(p_valid.shape[0]), np.zeros(n_valid.shape[0])])


########################
#### train NN model ####
########################

import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf
from sklearn.metrics import roc_auc_score

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc

BS = 250 # samples/epoch

# define input shape
input_shape = np.append(p_train.shape[1:],1)
print('input shape is = {}'.format(input_shape))

# 100 initializations and choose the best model. Run that model 4 times to get distribution parameters (u+-sd)
best_weights = []
best_performance = 0

for init in range(100):

    K.clear_session()

    inputs = Input(shape=input_shape)
    x = Conv2D(input_shape[0]-1, (4,input_shape[1]), activation=softplus)(inputs) # initializers are default in all layers.
    #x = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(x)
    x = Flatten()(x)
    #x = GlobalMaxPooling2D()(x)
    x = Dense(300, activation=softplus)(x) 
    x = Dropout(0.3)(x)
    x = Dense(30, activation=softplus, kernel_regularizer=regularizers.l2(0.01))(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])
    model.fit_generator(epoch_generator(n_train, p_train, batch_size=BS), 
                        validation_data=(X_valid.reshape(np.insert(X_valid.shape,3,1)), y_valid),
                        steps_per_epoch=int(len(p_train)/BS),
                        epochs=100,
                        callbacks=[EarlyStopping(patience=3)],
                        verbose=0)

    # validation
    y_hat = model.predict(X_test.reshape( np.insert(X_test.shape, 3, 1) ))
    roc = roc_auc_score(y_test, y_hat)

    if roc > best_performance:
        best_performance = roc
        best_weights = model.get_weights()

    print('{}, DaTa, val={}, test={}'.format(init, roc, roc_auc_score(y_test, model.predict(X_test.reshape( np.insert(X_test.shape, 3, 1) )))))

#acc
acc = np.sum(np.hstack(y_hat>0.5)==y_valid) / y_valid.shape[0]

# some random identifier
import uuid
filename = '_'.join(ARGS[1:]) + 'weigths.pkl'

import pickle
with open(filename, 'wb') as f:
    pickle.dump(best_weights, f)


#fp = open('seed_test_validation_IDp.csv', 'w')
#fn = open('seed_test_validation_IDn.csv', 'w')

#np.save(fp, np.hstack([IDp['validation'], IDp['test']]))
#np.save(fn, np.hstack([IDn['validation'], IDn['test']]))

#fp.close()
#fn.close()

#witt open(filename, 'w') as f:
#    f.write('auc,acc,{},{}'.format(roc,acc))




#ss
#iupred_short
#iupred_long
