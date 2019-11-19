#!/bin/python

from initialize_notebook import *

# Will keep performances on a hash table to compare later to other models.
single_aa_performances  = {'auroc': {'test':[], 'validation':[]}, \
                             'acc': {'test':[], 'validation':[]}, \
                           'auprc': {'test':[], 'validation':[]} 
                           }

# We can already define the validation set
Xtest = np.hstack(get_aa_frequencies( np.hstack([positives[ptest,0], negatives[ntest,0]]))).T    
ytest = np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])                    

# list of models to choose the best to plot
models = []
best_roc = 0
for counter, seed in enumerate(seeds):
    
    
    if '--only-one-example' in sys.argv and counter>0:
        continue
    
    # make indices 
    np.random.seed(seed)
    #print("random seed = ", np.random.get_state()[1][0])
    idp = np.random.permutation(ptrain)
    np.random.seed(seed)
    idn = np.random.permutation(ntrain)[:len(idp)]

    #Ptrain, Ptest = idp[:], idp[-test:]
    #Ntrain, Ntest = idn[:len(Ptrain)], idn[-test:]
    
    Xtrain = np.hstack(get_aa_frequencies(np.hstack([positives[idp,0], negatives[idn,0]]))).T
    ytrain = np.hstack([np.ones(len(idp)), np.zeros(len(idn))])
        
    Xtrain, ytrain = scrumble_index(Xtrain, ytrain)
    
    _10 = len(ytrain)//10
    Xtrain, ytrain = Xtrain[_10:], ytrain[_10:]
    Xvalid, yvalid = Xtrain[:_10], ytrain[:_10]

    if seed==seeds[0]: print(len(ptrain)," positives and ",len(ntrain)," negatives") # print info only @biginning
    
    ### train the logistic regression model
    model = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score)).fit(Xtrain,ytrain)
    models.append(model)
    
    # evaluate performance on test
    roc_test = roc_auc_score(ytest, model.predict_proba(Xtest)[:,1])
    acc_test = np.sum(ytest == model.predict(Xtest).astype(int)) / len(ytest)
    prc_test = average_precision_score(ytest, model.predict_proba(Xtest)[:,1])
    print('test\tAUROC={:.4f}\taccuracy={:.4f}\tAUPRC={:.4f}'.format(roc_test, acc_test, prc_test))

    # evaluate performance on validation 
    roc_valid = roc_auc_score(yvalid, model.predict_proba(Xvalid)[:,1])
    acc_valid = np.sum(yvalid == model.predict(Xvalid).astype(int)) / len(yvalid)
    prc_valid = average_precision_score(yvalid, model.predict_proba(Xvalid)[:,1])
    print('valid\tAUROC={:.4f}\taccuracy={:.4f}\tAUPRC={:.4f}'.format(roc_valid, acc_valid, prc_valid))

    # save performance results
    single_aa_performances['auroc']['test'].append(roc_test)
    single_aa_performances['acc']['test'].append(acc_test)
    single_aa_performances['auroc']['validation'].append(roc_valid)
    single_aa_performances['acc']['validation'].append(acc_valid)
    single_aa_performances['auprc']['test'].append(prc_test)
    single_aa_performances['auprc']['validation'].append(prc_valid)

    if roc_test > best_roc:
        best_roc = roc_test
        model_single_aa_composition = copy.copy(model)

print('single_aa_composition available')

# Plot Figure
sns.set_style('ticks')
f,ax = plt.subplots(1, figsize=(7,4.5))
cols = ['b']*3+['r']*2+['c']*4+['g']*8+['y']*3
ax.bar(np.arange(20), model_single_aa_composition.coef_[0], color=cols, linewidth=1, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.6)

ax.set_xticks(np.arange(20))
ax.set_xticklabels(aa, fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(-30,30)

plt.grid(axis='y')
plt.ylabel('Logistic Regression coefficients', fontsize=16)
plt.tight_layout()
