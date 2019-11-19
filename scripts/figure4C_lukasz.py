from initialize_notebook import *


# Performances were measured in a high-computing cluster in parallel
# each file in the following directory contains a 2D array with 
# values of y_hat and y_test

# get filenames
Dir = analysis_home+'/data/performances/'
files = {
    'single': [i for i in os.listdir(Dir) if re.search('_single_',i)],
    'dipeptides': [i for i in os.listdir(Dir) if re.search('_dipept_',i)],
    'seq': [i for i in os.listdir(Dir) if re.search('seq_--random-seed_', i)],
    'seq_iupred': [i for i in os.listdir(Dir) if re.search('seq_iupred_short_--random-seed_', i)],
    'seq_ss': [i for i in os.listdir(Dir) if re.search('seq_ss_--random-seed_', i)],
    'seq_ss_iupred': [i for i in os.listdir(Dir) if re.search('seq_ss_iupred_short_--random-seed_', i)]
}

# extract content of files into a dictionary with mean+/- SD of performance of models
performances = {i: {'auprc':{}, 'auroc':{}, 'acc':{}} for i in files.keys()}

for i in files.keys():
    
    # first get SD for each metric
    raw_values, metrics = [], {'auprc':[], 'auroc':[], 'acc':[]}

    for f in files[i]:
        tmp = np.load(Dir+f)
        raw_values.append(tmp)
        metrics['auprc'].append(average_precision_score(tmp[:,1],tmp[:,0]))
        metrics['auroc'].append(roc_auc_score(tmp[:,1],tmp[:,0]))
        metrics['acc'].append(accuracy(tmp[:,0], tmp[:,1].astype(int)))
    
    X = np.vstack(raw_values)
    performances[i]['auprc'] = np.hstack([average_precision_score( X[:,1], X[:,0] ), np.std( np.hstack(metrics['auprc']) )])    
    performances[i]['auroc'] = np.hstack([roc_auc_score( X[:,1], X[:,0] ),           np.std( np.hstack(metrics['auroc']) )])
    performances[i]['acc']   = np.hstack([accuracy( X[:,0], X[:,1].astype(int) ),    np.std( np.hstack(metrics['acc']) )])
    
    
#matplotlib.style.use('ggplot') #seaborn-colorblind')
fig = plt.figure(figsize=(30, 10))
ax = fig.add_subplot(111)
plt.setp(ax.spines.values(), linewidth=5)

keys = ['dipeptides', 'single', 'seq_ss_iupred', 'seq_iupred', 'seq_ss', 'seq'][::-1]
vals = np.vstack([performances[i]['auprc'] for i in keys])
plt.tick_params(axis='both', which='major', labelsize=30)
plt.xlabel('AUPRC score', fontsize=34, fontweight='bold')
plt.ylabel('models', fontsize=34, fontweight='bold')
plt.errorbar(vals[:,0], keys, xerr=vals[:,1], fmt='o', mfc='r', mec='r', ecolor='k', lw=5, markersize=14)
plt.savefig('figure_4C.png', dpi=600)
