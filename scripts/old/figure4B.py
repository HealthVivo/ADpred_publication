#!/bin/pypthon

from initialize_notebook import positives, negatives, ohe, aa, ss, ADPred


deep_samples = np.vstack([positives[pvalid,:2], negatives[nvalid,:2]])
ohe_aa = np.array([ohe(i,aa) for i in deep_samples[:,0]])
ohe_ss = np.array([ohe(i,ss) for i in deep_samples[:,1]])

deep_X = np.vstack([ohe_aa.T, ohe_ss.T]).T
deep_y = np.hstack([np.ones(len(pvalid)), np.zeros(len(nvalid))])
deep_idx = np.random.permutation(np.arange(len(deep_y)))



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

'''
from keras.models import model_from_json

deep_file = open(analysis_home+'/models/ADPred.json','r')
ADPred = deep_file.read(); deep_file.close()
ADPred = model_from_json(ADPred)
ADPred.load_weights(analysis_home+'/models/ADPred.h5')
'''

#with open('models/ssweigths.pkl', 'rb') as f:
#    w = pickle.load(f)
#ADPred.set_weights(w)


# ohe encode validation set

Xvalid = np.vstack([np.vstack(get_dipeptide_frequencies(i)) for i in (positives[pvalid,0], negatives[nvalid,0])])
yvalid = np.hstack([np.ones(len(pvalid)), np.zeros(len(nvalid))])
model = load(analysis_home+'/models/LogReg_dimers.joblib')
y_hat_linear = model.predict_proba(Xvalid)[:,1]
scale_linear, fpr_linear, tpr_linear = roc_curve( y_hat_linear, yvalid)

y_hat_deep = ADPred.predict(deep_X.reshape(np.append(deep_X.shape,1)))
scale_NN, fpr_NN, tpr_NN = roc_curve(y_hat_deep[deep_idx], deep_y[deep_idx])


df = pd.DataFrame(np.vstack([deep_y, y_hat_linear, np.hstack(y_hat_deep)]).T)
df.columns = ['label', 'y_hat_linear', 'y_hat_deep']
print(df.sample(5))

print(average_precision_score(deep_y, y_hat_linear))
print(average_precision_score(deep_y, y_hat_deep))

precision_linear, recall_linear, _ = precision_recall_curve(deep_y, y_hat_linear)
precision_deep, recall_deep, _ = precision_recall_curve(deep_y, y_hat_deep)

plt.style.use('ggplot')
plt.figure(figsize=(7,7))
plt.plot(recall_linear, precision_linear)
plt.plot(recall_deep, precision_deep)
plt.legend(['Linear models', 'Deep learning'], fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=22)

np.save('label_linear_deep', np.vstack([deep_y, y_hat_linear, np.hstack(y_hat_deep)]).T)

df.shape
