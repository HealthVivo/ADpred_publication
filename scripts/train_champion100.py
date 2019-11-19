#!/bin/python

# import libraries
import pickle, sys, os 
import numpy as np
sys.path.append(os.path.abspath('libraries'))
from summary_utils_v2B import *
import keras.backend as K
from keras.layers import Input, Dense, Conv2D, Flatten, GlobalMaxPooling2D, AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.activations import softmax, softplus, softsign, relu
from keras.callbacks import EarlyStopping
from keras import regularizers
import tensorflow as tf
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import ParameterGrid


# list of paramters to consider in this run
for n,i in enumerate(sys.argv):
    if i in ['--parameter-keys', '--paramters', '-p']:
        parameter_ids = np.array( sys.argv[n+1].split(',') ).astype(int) # start,end of the portion of params to analyze
    if i in ['--features','-f']:
        features = sys.argv[n+1].split(',')
    if i in ['--gpu', '-gpu', '_GPU','--GPU']:         # configure tensorflow to use GPUs
        config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} ) 
        sess = tf.Session(config=config)
        K.set_session(sess)


def main():
    # load data
    Path = '/fh/scratch/delete90/hahn_s/aerijman/'
    with open(Path + 'complete_dataset.pkl', 'rb') as f:
        data = pickle.load(f)

    print('data loaded!')

    # one hot encode data and save to disk with representation in memory
    ohe_data_positives, ohe_data_negatives = get_data(data, features)

    # is everything ok?
    print(ohe_data_positives.shape, ohe_data_negatives.shape)

    # split into 10 parts. pidx and nidx contain 10 arrays of indices for data['positives'] and data['negatives']
    pidx, nidx = split_indices(data)
    del data    # Not needed anymore 

    # test and valid are the indices of pidx and nidx for the CV
    np.random.seed(0)
    test = np.random.permutation(np.arange(10))

	parameter = {'activation':'softplus',
				  'conv_filter_shape0':6,
				  'dense1':30,
				  'dense2':100,	
				  'dropout':0.5,
				  'l2':0.001
				 }

	# define X and y, based on the current split
	train_idx = np.array([0,1,3,4,5,6,7,8,9])
	X,y = define_Xy(ohe_data_positives, ohe_data_negatives, pidx, nidx, train_idx=train_idx, test_idx=2)
	X_test = np.vstack([
						X['test']['positives'],
						X['test']['negatives']
						])


	# optimize parameters # ============================================================================================ 
	BS = 250
	epochs = 100
	input_shape = np.hstack([X['test']['positives'].shape[1:],1]) 
	last_auprc = 0      # hold performance to compare current weight-initialization

	last_auprc_valid = 0      # hold performance to compare current parameters 

	for init in range(100):: 
		# 10 random weight initializations for each set of parameters


		# redefine the model again with same parameters just to re-run the K.close_session()
		# so that I start with a new set of initialization weights.
		model = define_model(
						input_shape, 
						parameter['conv_filter_shape0'], 
						parameter['dense1'], 
						parameter['dense2'], 
						parameter['activation'], 
						parameter['dropout'], 
						parameter['l2']
						)

		model.fit_generator(
				epoch_generator(X['train']['negatives'], X['train']['positives'], batch_size=BS), 
				validation_data=(X_test.reshape(np.hstack([X_test.shape,1])), y['test']),
				steps_per_epoch=int(len(X['train']['positives'])/BS),
				epochs=epochs,
				callbacks=[EarlyStopping(patience=3)],
				verbose=0
				)

		y_hat = model.predict( X_test.reshape( np.hstack([X_test.shape,1]) ) )
		auprc = average_precision_score(y['test'], y_hat)

		if auprc > last_auprc:
			last_auprc = auprc
			weights = model.get_weights()[:]
	
	model.set_weights(weights)

	model_json = model.to_json()
	with open("ADPred_architecture.json", "w") as json_file:
		json_file.write(model_json)
	with open('ADPred_weights.pkl','wb') as f:
		pickle.dump(weights,f)
	
	

###### LIBRARY OF FUNCTIONS SECTION ============================================================================================

def get_data(data, features):
    global aa 
    global ss 
    global dis 
    
    ohe_data = {'positives':{},
                'negatives':{}
    }
    for ftrn, ftr in enumerate(['aa','ss','dis']):
        if ftr in features:
            for Set in data.keys():
                ohe_data[Set][ftr] = np.array( [ohe(i, eval(ftr)) for i in data[Set][:,ftrn]] )

    # stack into two arrays in a dictionary                
    ohe_data['positives'] = np.vstack([v.T for k,v in ohe_data['positives'].items()]).T
    ohe_data['negatives'] = np.vstack([v.T for k,v in ohe_data['negatives'].items()]).T
    # finally data is composed of positives and negatives arrays in disk
    ohe_data_positives, filename = store_data_numpy(ohe_data['positives'])
    ohe_data_negatives, filename = store_data_numpy(ohe_data['negatives'])

    return ohe_data_positives, ohe_data_negatives


def split_indices(data):
    np.random.seed(42)
    pidx = np.random.permutation( np.arange(len(data['positives'])) )
    pidx = np.array( np.array_split(pidx, 10) )

    np.random.seed(42)
    nidx = np.random.permutation( np.arange(len(data['negatives'])) )
    nidx = np.array( np.array_split(nidx, 10) )

    return pidx, nidx


def define_Xy(ohe_data_positives, ohe_data_negatives, pidx, nidx, train_idx, test_idx):
    X = {
        'train': {
            'positives': ohe_data_positives[np.hstack(pidx[train_idx])],
            'negatives': ohe_data_negatives[np.hstack(nidx[train_idx])]
        },
        'test': {
            'positives': ohe_data_positives[np.hstack(pidx[test_idx])],
            'negatives': ohe_data_negatives[np.hstack(nidx[test_idx])][:len(np.hstack(pidx[test_idx]))]   # stratified
        },
    }

    # append to negative-train the part of  negative-valid and negative-test that are not used
    X['train']['negatives'] = np.vstack([
                                    X['train']['negatives'],
                                    ohe_data_negatives[np.hstack(nidx[test_idx])][len(np.hstack(pidx[test_idx])):],
                                    ]) 

	y = {
        'train': np.hstack( [np.ones(len(X['train']['positives'])), 
                            np.zeros(len(X['train']['negatives']))] 
                            ),
        'test': np.hstack( [np.ones(len(X['test']['positives'])), 
                            np.zeros(len(X['test']['negatives']))] 
                        ),
    }
    return X, y


def set_split_idxs(test, split):
    test_idx = test[split]
    train_idx = np.array([i for i in range(10) if i not in test_idx])
    return test_idx, valid_idx, train_idx


def define_model(input_shape, 
                 conv_filter_shape0, 
                 dense1, 
                 dense2, 
                 activation, 
                 dropout, 
                 l2, 
                 optimizer='adam'):
	
	K.clear_session()
	
	inputs = Input(shape = input_shape)
	# convolutional layer
	x = Conv2D(input_shape[0]-1, (conv_filter_shape0, input_shape[1]), activation=activation)(inputs) # initializers are default in all layers.
	x = Flatten()(x)
	# first dense layer... if needed	
	#if dense0>0:
	#	x = Dense(dense0, activation=activation, kernel_regularizer=regularizers.l2(l2))(x)	
	#	x = Dropout(dropout)(x)
	# second/first dense layer
	if dense1>0:
		x = Dense(dense1, activation=softplus, kernel_regularizer=regularizers.l2(l2))(x)
		x = Dropout(dropout)(x)
	# third/second dense layer
	x = Dense(dense2, activation=softplus, kernel_regularizer=regularizers.l2(l2))(x)
	x = Dropout(dropout)(x)
	# output
	output = Dense(1, activation='sigmoid')(x)
	# concatenate layers and compile
	model = Model(inputs=inputs, outputs=output)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[auc_pr])
	
	return model


def auc_roc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    K.get_session().run(tf.local_variables_initializer())
    return auc


def auc_pr(y_true, y_pred):
    #auc = tf.metrics.auc(y_true, y_pred)[1]  # using defaults parameters --> num_thresholds=200
    auc = tf.metrics.auc(
        y_true,
        y_pred,
        curve= 'PR', #'ROC'
        summation_method='careful_interpolation'
    )[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


if __name__=='__main__': main()
