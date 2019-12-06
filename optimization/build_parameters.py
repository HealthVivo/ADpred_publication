#!/bin/python

import pickle
import numpy as np
from sklearn.model_selection import ParameterGrid

# dense_0 = np.arange(0,500,100)
# dense_1 = np.arange(0,500,100)
# dense_2 = np.arange(10,50,10)
# activation = ['relu','softplus']
# dropout = [0, 0.2, 0.3, 0.8]
# l2 = [0.001, 0.01, 0.1]
#optimizer = ['adam','rmsprop']
#epochs = [10,50,100]


conv_filter_shape0 = [2,4,6]
#dense_0 = [0] #np.arange(0,50,10)
dense_1 = [100, 300] #np.arange(0,50,10)
dense_2 = [10, 30] #np.arange(10,50,10)
activation = ['softplus'] #relu']
dropout = [0.3, 0.5]
l2 = [0.001, 0.01]

parameters = list(ParameterGrid({
					'conv_filter_shape0': conv_filter_shape0,
					# 'dense0':dense_0,
					'dense1':dense_1,
					'dense2':dense_2,
					'activation':activation,
					'dropout':dropout,
					'l2':l2
			}))

with open('/fh/scratch/delete90/hahn_s/aerijman/parameters.pkl','wb') as f:
	pickle.dump(parameters, f)

