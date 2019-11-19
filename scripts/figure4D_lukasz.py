#!/bin/python

#plt.style.available

from initialize_notebook import *
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib

def roc_curve(X,y):
    X,y = np.array(X), np.array(y)
    # thresholds to use for defining fpr and tpr
    scale = np.linspace(X.min(), X.max(), 50)
    
    # define fpr and tpr functions to use with different thresholds
    def tpr(y_hat, y):
        a,b = set(np.hstack(np.where(y==1))), set(np.hstack(np.where(y_hat==1)))
        return(len(a & b) / len(a))
    def fpr(y_hat, y):
        a,b = set(np.hstack(np.where(y==0))), set(np.hstack(np.where(y_hat==1)))
        return(len(a & b) / len(a))
    
    # Core of the function where TPR and FPR are calculated
    TPR, FPR = np.zeros(50), np.zeros(50)
    for n,s in enumerate(scale):
        # threshold of variable defines what is + and -
        y_hat = np.array([1 if x>=s else 0 for x in X])
        TPR[n] = tpr(y_hat, y)
        FPR[n] = fpr(y_hat, y)
    
    # the resulting array will contain thresholds, fpr and tpr
    roc = np.vstack([scale, FPR, TPR])
    return(roc)

plt.style.use('tableau-colorblind10')
#matplotlib.style.use('ggplot')

deep_samples = np.vstack([positives[ptest,:2], negatives[ntest,:2]])
ohe_aa = np.array([ohe(i,aa) for i in deep_samples[:,0]])
ohe_ss = np.array([ohe(i,ss) for i in deep_samples[:,1]])

deep_X = np.vstack([ohe_aa.T, ohe_ss.T]).T
deep_y = np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])
deep_idx = np.random.permutation(np.arange(len(deep_y)))

#f, ax = plt.subplots(1, figsize=(7,7))
f, (ax,ax2) = plt.subplots(2,1, figsize=(10.5,12), sharex=True, gridspec_kw={'height_ratios':[3,1], 'hspace':0.05})

plt.setp(ax.spines.values(), linewidth=5)
plt.setp(ax2.spines.values(), linewidth=5)

Qs = charge(deep_samples[:,0]) / 30
cols = ['b' if i==0 else 'r' for i in deep_y[deep_idx]]

Xtest = np.vstack([np.vstack(get_dipeptide_frequencies(i)) for i in (positives[ptest,0], negatives[ntest,0])])
ytest = np.hstack([np.ones(len(ptest)), np.zeros(len(ntest))])
model = joblib.load(analysis_home+'/models/LogReg_dimers.joblib')
y_hat_linear = model.predict_proba(Xtest)[:,1]
scale_linear, fpr_linear, tpr_linear = roc_curve( y_hat_linear, ytest)

y_hat_deep = ADPred.predict(deep_X.reshape(np.append(deep_X.shape,1)))
ax.scatter(y_hat_deep[deep_idx], Qs[deep_idx], color=cols, s=10)


ax.scatter([0.5,0.5],[0.25,0.31], s=40, c=['b','r'])
ax.text(0.55,0.288, "AD positive", fontsize=24)
ax.text(0.55,0.228, "AD negative", fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=30)

plt.rc('axes', facecolor='#E6E6E6')
ax.grid(color='w', linestyle='solid')
ax2.grid(color='w', linestyle='solid')
ax2.tick_params(axis='both', which='major', labelsize=30)
y_hat_deep = ADPred.predict(deep_X.reshape(np.append(deep_X.shape,1)))

xn = np.array([n for n,i in enumerate(cols) if i=='r'])
xp = np.array([n for n,i in enumerate(cols) if i=='b'])

ax2.hist(y_hat_deep[deep_idx[xn]], density=True, bins=100, color='r', alpha=0.5);
ax2.hist(y_hat_deep[deep_idx[xp]], density=True, bins=100, color='b', alpha=0.5);



ax2.set_xticks(np.linspace(0,1,6))
ax2.tick_params(axis='x', which='major', labelsize=30)
#ax.tick_params(top='off', bottom='off', left='on', right='off', labelleft='off', labelbottom='off')
ax.tick_params(bottom='off')
ax2.tick_params(labelleft='off', left='off')

plt.xlabel('ADpred score', fontsize=34, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=30)
#plt.ylabel('Avg charge/residue', fontsize=34, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_4D.png', dpi=600)
