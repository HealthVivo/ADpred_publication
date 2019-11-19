#!/bin/python

# import libraries used throughout this notebook
import os,sys, matplotlib
from matplotlib import rc, rcParams, font_manager

analysis_home=os.path.abspath('./')
sys.path.append(os.path.abspath(analysis_home))
from libraries.utils import *

import joblib 
import seaborn as sns 
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output
import ipywidgets as widgets
import pickle
from scipy.stats import chi2, linregress, spearmanr
import copy
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
posit10 = p.shape[0]//10
negat10 = posit10 #n.shape[0]//10

# define indices for positive and negative training + testing sets
#negat = n[:-nvalid]
#posit = p[:-pvalid]
pvalid = p[-posit10:]
nvalid = n[-negat10:]

# define inidices for split data
ptest = p[:posit10]
ntest = n[:posit10] 

ptrain = p[posit10:-posit10]
ntrain = n[posit10:-negat10]

# load results from linear models in case the user doesn't have the time to run 
# the regression analysis's optimization and can focus straight on the results.
with open(analysis_home+'/results/single_aa_performances_pos.pickle', 'rb') as handle:
    single_aa_performances_pos = pickle.load(handle)
    
with open(analysis_home+'/results/dipept_aa_performances.pickle', 'rb') as handle:
    dipept_aa_performances = pickle.load(handle)    
    
with open(analysis_home+'/results/single_aa_performances.pickle', 'rb') as handle:
    single_aa_performances = pickle.load(handle) 


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


# load ADPred #
#deep_file = open('models/deep_model.json','r')
#ADPred = deep_file.read(); deep_file.close()
#ADPred = model_from_json(ADPred)
#ADPred.load_weights('models/deep_model.h5')

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc


BS = 250 # samples/epoch

K.clear_session()

inputs = Input(shape=(30,23,1))
x = Conv2D(29, (4,23), activation=softplus)(inputs)
#x = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(x)
x = Flatten()(x)
x = Dense(300, activation=softplus)(x)  
x = Dropout(0.3)(x)
x = Dense(30, activation=softplus, kernel_regularizer=regularizers.l2(0.01))(x)
#output = Dense(1, activation='sigmoid')(x)
x = Dense(1)(x)
output = (Activation('sigmoid'))(x)
ADPred = Model(inputs=inputs, outputs=output)
ADPred.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc])

#print(ADPred.summary())

# get optimized weigths of models
#f = open(analysis_home+'/scripts/ss.weigths.pkl','rb')

##f = open(analysis_home+'/scripts/ss.weigths.pkl','rb')
##weights = pickle.load(f)
#ADPred.set_weights(weights)


#f = open(analysis_home+'/modelsB/ss.weigths.pkl','rb')
#f = open(analysis_home+'/models/new_train_models/with_max_pooling/ss_balanced.tweigths.pkl', 'rb')
#f = open(analysis_home+'/models/new_train_models/no_max_pooling/ss.weigths.pkl', 'rb')
f = open(analysis_home+'/models/new_train_fix_split/ss_--balanced-testweigths.pkl', 'rb')
weights = pickle.load(f)
ADPred.set_weights(weights)
#ADPred.load_weights('models/deep_model.h5')

HELP='''
    \x1b[1;37;40mAvailable methods\x1b[0m:
    \x1b[1;37;42mget_enrichment_scores_table\x1b[0m: pandas table with sequences, counts/bin and enrichment scores
    \x1b[1;37;42mADPred\x1b[0m: Keras model to predict ADs probability in a 30mer.
    Test and Validation sets have 50% positives and 50% negatives.
'''

print(HELP)
