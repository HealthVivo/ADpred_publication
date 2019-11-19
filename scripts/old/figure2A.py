#!/bin/python

from initialize_notebook import *

if '--plot-intermediates' in sys.argv: 
    flag=True
else:
    flag=False

# generate amino-acicd frequencies for p(positives) and n(negatives)
p = np.hstack(get_aa_frequencies(positives)[:,0]*30).T
n = np.hstack(get_aa_frequencies(negatives)[:,0]*30).T

# scrumble the data
#p = p[np.random.permutation( np.arange(lp) )]
#n = n[np.random.permutation( np.arange(ln) )]

# hold out a validation set
#valid = int(p.shape[0]/10)
#p_valid = p[-valid:]
#n_valid = n[-valid:]

# obtain X and y to train and test
ptt = np.hstack([ptest,ptrain])
ntt = np.hstack([ntest,ntrain[:len(ptrain)]]) 
X = np.vstack([n[ntt],p[ptt]])
y = np.hstack([np.zeros(len(ntt)), np.ones(len(ptt))])

idx = np.arange(len(X)) # index to be used for splitting train test in the 5 fold CV 
LL_av = [] # Initialize list of coefficients to average over the 5 models.
    
''' compute the log-likelihood of aa to be in the positive set over the negative and plot distribution
    of log-likelihood scores of all samples of our library.
'''
if flag: plt.figure(figsize=(10,15))

for counter, (train, test) in enumerate(KFold(n_splits=5, shuffle=True).split(idx)): 

    if '--only-one-example' in sys.argv and counter>0:
        continue 
   
    # define sets to build the log-likelihood factors 
    train_pos, train_neg = X[train][y[train]==1], \
                           X[train][y[train]==0]
        
    test_pos, test_neg = X[test][y[test]==1], \
                         X[test][y[test]==0]
    
    # log-likelihood coefficients calculation on training data
    LL = np.log2( np.mean(train_pos, axis=0) / np.mean(train_neg, axis=0) )
    LL_av.append(LL)

    # calculate log-likelihood scores for all samples
    _X = np.sum(X * LL, axis=1)
    
    if flag:
        # generate plot on the train set
        plt.subplot(3,2,counter+1)
        plt.hist(_X[train][y[train]==0], bins=100, color='r', alpha=0.4, density=True)
        plt.hist(_X[train][y[train]==1], bins=100, color='g', alpha=0.4, density=True)
    
        # calculate ROC on test set
        roc = roc_auc_score(y[test], _X[test])
        plt.title('AUROC = {:.3f}'.format(roc))


# validation on an averaged model    
LL_av = np.vstack(LL_av).mean(axis=0)
X_valid = np.hstack([np.sum(p[pvalid] * LL_av, axis=1), 
                     np.sum(n[nvalid] * LL_av, axis=1)])
y_valid = np.hstack([np.ones(len(pvalid)), np.zeros(len(nvalid))])

if flag:
    plt.subplot(3,2,6)
    plt.hist(X_valid[y_valid==1], bins=100, color='g', alpha=0.4, density=True)
    plt.hist(X_valid[y_valid==0], bins=100, color='r', alpha=0.4, density=True)
    roc = roc_auc_score(y_valid, X_valid)
    plt.title('AUROC = {:.3f}'.format(roc))


## plot validation set on averaged log-likelihood coefficients for the figure.
a1,b1 = np.histogram(X_valid[y_valid==1], bins=100, density=True)
a0,b0 = np.histogram(X_valid[y_valid==0], bins=100, density=True)
[a0, a1] = [np.convolve(i, np.ones(10)/10, 'same') for i in [a0,a1]]
[b0,b1] = [np.array([np.mean(i[j:j+2]) for j in range(len(i)-1)]) for i in [b0,b1]]

#plt.figure(figsize=(5,4))
plt.figure(figsize=(10,5))
plt.fill(b0,a0, alpha=0.5, c='r', label='AD negative')
plt.fill(b1,a1, alpha=0.5, c='g', label='AD positive')
plt.xlabel('log-likelihood score', fontweight='bold', fontsize=20)
plt.ylabel('frequency', fontweight='bold', fontsize=20)
plt.legend(loc=2, fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
