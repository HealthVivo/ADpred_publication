# Performances were measured in a high-computing cluster in parallel
# each file in the following directory contains a 2D array with 
# values of y_hat and y_test

# get filenames
#Dir = analysis_home+'/data/performances/'
#performances = {}
#with open(Dir+'performances.csv', 'r') as f:
#    while True:
#        try:
#            model, perform = next(f).strip().split(',')[:2]
#            if model in performances:
#                performances[model].append(perform)
#            else:
#                performances[model] = [perform]
#        except StopIteration:
#            break
files = {
    'single': [i for i in os.listdir(Dir) if re.search('_single_',i)],
    'dipeptides': [i for i in os.listdir(Dir) if re.search('_dipept_',i)],
#    'seq': [i for i in os.listdir(Dir) if re.search('seq_--random-seed_', i)],
#    'seq_iupred': [i for i in os.listdir(Dir) if re.search('seq_iupred_short_--random-seed_', i)],
#    'seq_ss': [i for i in os.listdir(Dir) if re.search('seq_ss_--random-seed_', i)],
#    'seq_ss_iupred': [i for i in os.listdir(Dir) if re.search('seq_ss_iupred_short_--random-seed_', i)]
}

# extract content of files into a dictionary with mean+/- SD of performance of models
performances = {i: {'AUPRC':{}, 'AUROC':{}, 'ACC':{}} for i in files.keys()}

for i in files.keys():
    
    # first get SD for each metric
    raw_values, metrics = [], {'AUPRC':[], 'AUROC':[], 'ACC':[]}

    for f in files[i]:
        tmp = np.load(Dir+f)
        raw_values.append(tmp)
        metrics['AUPRC'].append(average_precision_score(tmp[:,1],tmp[:,0]))
        metrics['AUROC'].append(roc_auc_score(tmp[:,1],tmp[:,0]))
        metrics['ACC'].append(accuracy(tmp[:,0], tmp[:,1].astype(int)))
    
    X = np.vstack(raw_values)
    performances[i]['AUPRC'] = np.hstack([average_precision_score( X[:,1], X[:,0] ), np.std( np.hstack(metrics['AUPRC']) )])    
    performances[i]['AUROC'] = np.hstack([roc_auc_score( X[:,1], X[:,0] ),           np.std( np.hstack(metrics['AUROC']) )])
    performances[i]['ACC']   = np.hstack([accuracy( X[:,0], X[:,1].astype(int) ),    np.std( np.hstack(metrics['ACC']) )])
    
    
#matplotlib.style.use('ggplot') #seaborn-colorblind')
#plt.figure(figsize=(10,5))
#names = ['dipeptides', 'single', 'seq_ss_iupred', 'seq_iupred', 'seq_ss', 'seq'][::-1]
#keys = ["linear_dip", "linear_aa", "aa_ss_dis", "aa_dis", "aa_ss", "aa"][::-1]

#vals = np.vstack([performances[i]['auprc'] for i in keys])
#vals = np.vstack([performances[i] for i in keys])
#plt.errorbar(vals[:,0], names, xerr=vals[:,1], fmt='o', mfc='r', mec='r', ecolor='k', lw=3, markersize=10)


#Dir = analysis_home+'/data/' #performances/'
#performances = {}
#with open(Dir+'performances.csv', 'r') as f:
#    while True:
#        try:
#            model, perform = next(f).strip().split(',')[:2]
#            perform = float(perform)
#            if model in performances:
#                performances[model].append(perform)
#            else:
#                performances[model] = [perform]
#        except StopIteration:
#            break



# DEEP LEARNING MODELS WERE WRONG AND ARE CORRECTED HERE
df = pd.read_csv(analysis_home + '/data/performances.publication_with_CVnumber.tsv', sep='\t', header=None)
df.columns = ['CV','features','AUPRC','AUROC','ACC','parameters']


for i in ["aa_ss_dis", "aa_dis", "aa_ss", "aa"]:
    performances[i] = {}
    for k in ['AUPRC', 'AUROC', 'ACC']:
        tmp = [ (df[(df.CV==j) & (df.features==i)][k].max(axis=0)) for j in range(10) ]        
        performances[i][k] = [ np.mean(tmp), np.std(tmp)/np.sqrt(10) ]


#performances = {}
#for i in ["aa_ss_dis", "aa_dis", "aa_ss", "aa"]:
#    performances[i] = [ (df[(df.CV==j) & (df.features==i)]['AUPRC'].max(axis=0)) for j in range(10) ]

names = ['dipeptides', 'single', 'seq_ss_iupred', 'seq_iupred', 'seq_ss', 'seq'][::-1]
#keys = ["linear_dip", "linear_aa", "aa_ss_dis", "aa_dis", "aa_ss", "aa"][::-1]

keys = ["dipeptides", "single", "aa_ss_dis", "aa_dis", "aa_ss", "aa"][::-1]
vals  = np.array([performances[i]['AUPRC'] for i in keys])

#vals=[]
#for i in keys:
#    av = np.mean(performances[i])
#    sdm = np.std(performances[i]) / np.sqrt(10)
#    vals.append([av,sdm])
    #print("{:<10}: {:.3f} \u00B1 {:.5f}".format(i,av,sdm))
#vals = np.array(vals)

print('feature         AUPRC              AUROC              ACC\n', ''.join(['-']*63))
for k,v in performances.items():
    print('{:<10} {:.4f} \u00B1 {:.4f}    {:.4f} \u00B1 {:.4f}    {:.4f} \u00B1 {:.4f}'.format(
        k, 
        v['AUPRC'][0], v['AUPRC'][1],
        v['AUROC'][0], v['AUROC'][1],
        v['ACC'][0], v['ACC'][1]
    ))

matplotlib.style.use('seaborn-colorblind')
plt.figure(figsize=(20,10))    

plt.scatter(vals[:,0], np.arange(6), s=300)
for i in range(6):
    plt.plot([vals[i,0]-vals[i,1], vals[i,0]+vals[i,1]], [i,i], c='k', lw=5)
plt.yticks(np.arange(6), names);
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid()
