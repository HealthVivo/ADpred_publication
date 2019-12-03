#!/bin/python

from initialize_notebook import *

# Will keep performances on a hash table to compare later to other models.
dipept_aa_performances  = {'auroc': {'test':[], 'validation':[]}, \
                             'acc': {'test':[], 'validation':[]}, \
                           'auprc': {'test':[], 'validation':[]}
                           }

# We can already define the validation set
Xtest = np.vstack(get_dipeptide_frequencies( np.hstack([positives[ptest,0], negatives[ntest,0]])))
ytest = np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])

# keep track of models to choose the best
models_dipept = []
best_roc = 0
# This is very time consuming and the results are ready somewhere else.
for counter, seed in enumerate(seeds): #[7:8]: # I run this in the cluster but get the model parameters from the best performer.

    if '--only-one-example' in sys.argv and counter>0:
        continue

    # make indices 
    np.random.seed(seed)
    #print("random seed = ", np.random.get_state()[1][0])
    idp = np.random.permutation(ptrain)
    np.random.seed(seed)
    idn = np.random.permutation(ntrain)[:len(idp)]

    Xtrain = np.vstack(get_dipeptide_frequencies(np.hstack([positives[idp,0], negatives[idn,0]])))
    ytrain = np.hstack([np.ones(len(idp)), np.zeros(len(idn))]) 
        
    Xtrain, ytrain = scrumble_index(Xtrain, ytrain)
    
    _10 = len(ytrain)//10
    Xtrain, ytrain = Xtrain[_10:], ytrain[_10:]
    Xvalid, yvalid = Xtrain[:_10], ytrain[:_10]

    if seed==seeds[0]: print(len(ptrain)," positives and ",len(ntrain)," negatives") # print info only @biginning

    ### train the logistic regression model
    model = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), cv=5, scoring = make_scorer(roc_auc_score), max_iter=700).fit(Xtrain,ytrain)
    models_dipept.append(model)
    
    # evaluate performance on test
    roc_test = roc_auc_score(ytest, model.predict_proba(Xtest)[:,1])
    acc_test = np.sum(ytest == model.predict(Xtest).astype(int)) / len(ytest)
    prc_test = average_precision_score(ytest, model.predict_proba(Xtest)[:,1])
    print('test\tAUROC={:.4f}\taccuracy={:.4f}\tAUPRC={:.4f}'.format(roc_test, acc_test, prc_test))

    roc_valid = roc_auc_score(yvalid, model.predict_proba(Xvalid)[:,1])
    acc_valid = np.sum(yvalid == model.predict(Xvalid).astype(int)) / len(yvalid)
    prc_valid = average_precision_score(yvalid, model.predict_proba(Xvalid)[:,1])
    print('valid\tAUROC={:.4f}\taccuracy={:.4f}\tAUPRC={:.4f}'.format(roc_valid, acc_valid, prc_valid)) 
    
    # save performance results
    dipept_aa_performances['auroc']['test'].append(roc_test)
    dipept_aa_performances['acc']['test'].append(acc_test)
    dipept_aa_performances['auroc']['validation'].append(roc_valid)
    dipept_aa_performances['acc']['validation'].append(acc_valid)

    if roc_test > best_roc:
        best_roc = roc_test
        model_dipeptides = copy.copy(model)

print('model_dipeptides now available')
best = np.argmax(dipept_aa_performances['auroc']['validation'])
#mpl.rcParams.update(mpl.rcParamsDefault)
#mpl.style.use('seaborn-colorblind')
#f, ax = plt.subplots(1, figsize=(10,5))
#im = ax.pcolor(models_dipept[best].coef_.reshape(20,20), cmap=cm.jet)

#ax.set_xticks(np.arange(20)+0.5)
#ax.set_yticks(np.arange(20)+0.5)

#ax.set_xticklabels(aa, fontsize=14)
#ax.set_yticklabels(aa, fontsize=14)

#plt.colorbar(im, shrink=0.5)
#plt.savefig('/Users/aerijman/Desktop/figure3.png')


# LIKELIHOOD RATIO

if '--calculate-likelihood-ratio' in sys.argv:
    # dataset to train alt and null models
    #id_DW = np.where(dipeptides=='DW')[0][0]
    p_values = []

    for NULL in range(400):

        id_alt = np.ones(400).astype(np.bool)
        id_null = np.ones(400).astype(np.bool)
        id_null[NULL]=0

        Xtrain_null =  Xtrain[:,id_null]
        Xtrain_alt  =  Xtrain[:,id_alt]

        Xtest_null = Xtest[:,id_null] 
        Xtest_alt   = Xtest[:,id_alt]

        df = Xtest_alt.shape[1] - Xtest_null.shape[1] # degrees of freedom

        # train both models
        model_null = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), 
                                          cv=5, 
                                          scoring = make_scorer(roc_auc_score),                                   
                                          max_iter=700).fit(Xtrain_null,ytrain)

        model_alt = LogisticRegressionCV(Cs = np.linspace(1e-4, 1e4, 40), 
                                          cv=5, 
                                          scoring = make_scorer(roc_auc_score),                                   
                                          max_iter=700).fit(Xtrain_alt,ytrain)


        # get probabilities for each model
        alt_prob = model_alt.predict_proba(Xtest_alt)
        null_prob = model_null.predict_proba(Xtest_null)

        # log-loss from complete (alternate) model from the previous cells
        alt_log_likelihood = -log_loss(ytest, alt_prob, normalize=False)
        null_log_likelihood = -log_loss(ytest, null_prob, normalize=False)

        G = 2 * (alt_log_likelihood - null_log_likelihood)
        p_value = chi2.sf(G, df)
        p_values.append(p_value)

    print('likelihood ratio test for DW_coeff=0\nG={:.16f}\np_value={:.3f}'.format(G,p_value))

else:
    filename = analysis_home+'/results/likelihood_ratio.results'
    pvals = [i.strip().split(',') for i in open(filename,'r')][1:]
    print('to save your time, using saved results from ' + filename)
    dips = [i[:2] for i in pvals]
    pvals = np.array(pvals)[:,-2:].astype(float)


    ps = [np.mean(pvals[i:i+10]) for i in range(0,pvals.shape[0],10)]

    small_ps = np.where(np.array(ps)<0.001)[0]
    int(small_ps[0]/20), small_ps[0]%20

    best = np.argmax(dipept_aa_performances['auroc']['validation'])
    #mpl.rcParams.update(mpl.rcParamsDefault)
    matplotlib.style.use('seaborn-colorblind')

    f, ax = plt.subplots(1, figsize=(18,15))
    im = ax.pcolor(models_dipept[best].coef_.reshape(20,20), cmap='seismic') #jet_r') #Spectral', )
    ax.set_xticks(np.arange(20)+0.5)
    ax.set_yticks(np.arange(20)+0.5)

    ax.set_xticklabels(aa, fontsize=28)
    ax.set_yticklabels(aa, fontsize=28)

    small_ps = np.where(np.array(ps)<0.001)[0]

    for i in small_ps:
        ax.text(i%20+0.1, int(i/20)+0.2, int(np.log10(ps[i])), fontsize=25, color='white', weight='bold')

    from matplotlib import ticker
    cb = plt.colorbar(im, shrink=0.5)
    tick_locator = ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    cb.ax.tick_params(labelsize=25)




