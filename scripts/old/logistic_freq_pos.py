#!/bin/python



# Will keep performances on a hash table to compare later to other models.
single_aa_performances_pos  = {'auroc': {'test':[], 'validation':[]}, \
                             'acc': {'test':[], 'validation':[]}}

# We can already define the validation set NEGATIVE-VALIDATION SET TO SAME NUMBER OF POSITIVES only for computation issues
Xvalid = np.vstack([np.hstack(get_aa_freq_pos(i)).T for i in np.hstack([positives[pvalid,0], negatives[nvalid,0]])])
yvalid = np.hstack([np.ones(len(pvalid)), np.zeros(len(nvalid))])

for counter,seed in enumerate(seeds):

    if '--only-one-example' in sys.argv and counter>0:
        continue
    
    # make indices 
    np.random.seed(seed)
    #print("random seed = ", np.random.get_state()[1][0])
    idp = np.random.permutation(ptrain)
    np.random.seed(seed)
    idn = np.random.permutation(ntrain)[:len(idp)]
    
    Xtrain = np.vstack([np.hstack(get_aa_freq_pos(i)).T for i in np.hstack([positives[idp,0], negatives[idn,0]])])
    Xtest = np.vstack([np.hstack(get_aa_freq_pos(i)).T for i in np.hstack([positives[ptest,0], negatives[ntest,0]])])
    ytrain, ytest = np.hstack([np.ones(len(idp)), np.zeros(len(idn))]), np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])

    if '--verbose' in sys.argv and n==0:
        print(len(ptrain)," positives and ",len(ntrain)," negatives") # print info only @biginning
    
    ### train the logistic regression model
    model = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score), max_iter=500).fit(Xtrain,ytrain)

    # evaluate performance on test
    roc_test = roc_auc_score(ytest, model.predict_proba(Xtest)[:,1])
    acc_test = np.sum(ytest == model.predict(Xtest).astype(int)) / len(ytest)
    print('test\tAUROC={:.4f}\taccuracy={:.4f}'.format(roc_test, acc_test))

    # evaluate performance on validation 
    roc_valid = roc_auc_score(yvalid, model.predict_proba(Xvalid)[:,1])
    acc_valid = np.sum(yvalid == model.predict(Xvalid).astype(int)) / len(yvalid)
    print('valid\tAUROC={:.4f}\taccuracy={:.4f}'.format(roc_valid, acc_valid))

    # save performance results
    single_aa_performances_pos['auroc']['test'].append(roc_test)
    single_aa_performances_pos['acc']['test'].append(acc_test)
    single_aa_performances_pos['auroc']['validation'].append(roc_valid)
    single_aa_performances_pos['acc']['validation'].append(acc_valid)


f,ax = plt.subplots(1, figsize=(5,4))
ax.pcolor(model.coef_[0].reshape(30,20), cmap='jet')
ax.set_xticks(np.arange(20)+0.5)
ax.set_xticklabels(aa)
ax.set_xlabel('amino acid', fontsize=20)
ax.set_ylabel('position', fontsize=20);

