#!/bin/python

from initialize_notebook import *

# Will keep performances on a hash table to compare later to other models.
single_aa_performances  = {'auroc': {'test':[], 'validation':[]}, \
                             'acc': {'test':[], 'validation':[]}}

# model training in train_logReg_single_aa.py
# model = load('../models/logReg_single_aa.joblib')

# We can already define the validation set
Xvalid = np.vstack([np.hstack(get_aa_frequencies(i)).T for i in (positives[pvalid,0], negatives[nvalid,0])])    
yvalid = np.hstack([np.ones(len(pvalid)), np.zeros(len(nvalid))])                    

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
    
    Xtrain = np.vstack([np.hstack(get_aa_frequencies(i)).T for i in (positives[idp,0], negatives[idn,0])])
    Xtest = np.vstack([np.hstack(get_aa_frequencies(i)).T for i in (positives[ptest,0], negatives[ntest,0])])
    ytrain, ytest = np.hstack([np.ones(len(idp)), np.zeros(len(idn))]), np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])
    
    if seed==seeds[0]: print(len(ptrain)," positives and ",len(ntrain)," negatives") # print info only @biginning
    
    ### train the logistic regression model
    model = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score)).fit(Xtrain,ytrain)
    models.append(model)
    
    # evaluate performance on test
    roc_test = roc_auc_score(ytest, model.predict_proba(Xtest)[:,1])
    acc_test = np.sum(ytest == model.predict(Xtest).astype(int)) / len(ytest)
    print('test\tAUROC={:.4f}\taccuracy={:.4f}'.format(roc_test, acc_test))

    # evaluate performance on validation 
    roc_valid = roc_auc_score(yvalid, model.predict_proba(Xvalid)[:,1])
    acc_valid = np.sum(yvalid == model.predict(Xvalid).astype(int)) / len(yvalid)
    print('valid\tAUROC={:.4f}\taccuracy={:.4f}'.format(roc_valid, acc_valid))

    # save performance results
    single_aa_performances['auroc']['test'].append(roc_test)
    single_aa_performances['acc']['test'].append(acc_test)
    single_aa_performances['auroc']['validation'].append(roc_valid)
    single_aa_performances['acc']['validation'].append(acc_valid)

    if roc_test > best_roc:
        best_roc = roc_test
        model_single_aa_composition = copy.copy(model)

print('single_aa_composition available')

# Plot Figure
f,ax = plt.subplots(1, figsize=(7,5))
cols = ['b']*3+['r']*2+['c']*4+['g']*8+['y']*3
ax.bar(np.arange(20), model.coef_[0], color=cols)
ax.set_xticks(np.arange(20))
ax.set_xticklabels(aa, fontsize=14)
plt.grid(axis='y')
plt.ylabel('Logistic Regression coefficients', fontsize=20)
plt.tight_layout()

