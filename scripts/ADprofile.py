#!/bin/python

# Load libraries, data and pseudorandom nambers for reproducibility
# the number of positive and negatives samples will be displayed
import warnings, uuid
warnings.filterwarnings('ignore')
# import libraries used throughout this notebook
import os,sys, matplotlib, copy, tarfile, pickle, joblib
from matplotlib import rc, rcParams, font_manager

analysis_home=os.path.abspath('./')
sys.path.append(os.path.abspath(analysis_home))
from libraries.utils import *

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, clear_output
from scipy.stats import chi2, linregress, spearmanr
from sklearn.metrics import precision_recall_curve, average_precision_score, log_loss, roc_auc_score, make_scorer
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout, Activation
from keras.models import Model, model_from_json
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf

for n,i in enumerate(sys.argv):
    if i in ['--sequence','i']:
        sequence = sys.argv[n+1].upper()

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc

def make_ADPred():
    with open('models/ADpred.json', 'r') as f:
        model_definition = f.read()
    ADPred = model_from_json(model_definition)
    with open('models/ADpred_weights.pkl', 'rb') as f:
        weights = pickle.load(f)
    ADPred.set_weights(weights)
    return ADPred

ADPred = make_ADPred()

# function to run pripred
from subprocess import Popen, PIPE, call

def psipred(sequence):
    name = str(uuid.uuid4())
    fq = open(name, 'w')
    fq.write(sequence)
    fq.close()

    p = ['bash', 'run_psipred', name]
    res = Popen(p, stdout=PIPE).communicate()[0].decode('utf-8').strip().replace('C','-')
    os.remove(name)
    
    return res


def find_AD_lenghts(result):
    r=[]
    
    long = 0
    for i in result:
        if i<0.8:
            if long>=1: #5:
                r.append(long)
                long = 0
            else:
                continue
        else:
            long +=1
    return r


# predict secondary structure and proceed to ohe
struct = psipred(sequence)
ohe = prepare_ohe([sequence, struct])


adpred_complete_sequence = []  #adpred score on sliding 30mers

# IMPORTANT!!! --> here position 0 is actual position 0 not 15.
# n is the start of the 30mer!!
for i in range(len(ohe)-30):
    adpred = ADPred.predict(ohe[i:i+30].reshape(1,30,23,1))[0][0]
    adpred_complete_sequence.append(adpred)

print(','.join([str(i) for i in adpred_complete_sequence]))

