#################################################################################
# load table with sequence, secondary structure, disorder and prediction scores #
#################################################################################

# forgot to add the confidence of secondary_structure_oredictions...
path = analysis_home + '/data/secondary_structure/'
tmp, tnp = [], []
for f in [i for i in os.listdir(path) if i[-11:]==".output.csv"]:
    predsName = f + ".predictions.npz"

    df = pd.read_csv(path + f, index_col=0)
    tmp.append(df[['sequence','secondStruct','disorder']])
    nf = np.load(path + predsName, allow_pickle=True)
    tnp.append(nf[nf.files[0]])

predictions = np.hstack(tnp)
df = pd.concat(tmp)

# finally join all fields into a single data structure to facilitate further analysis
df['predictions'] = predictions

# Apparently what I needed here are the disorder probabilities rather than yes/no
Dir = analysis_home+'/data/disorder_yeast/'
files = [i for i in os.listdir(Dir) if re.search(".results$",i)]

disorders = {}
for i in files:
    d = read_iupred_results(Dir+i)
    disorders.update(d)

df2 = pd.DataFrame([disorders]).T
idx = df2.index.intersection(df.index)
df2 = df2.loc[idx]

df = pd.concat([df.loc[idx],df2], axis=1)
df.columns = ['sequence', 'secondStruct', 'disorder', 'predictions', 'iupred']

del(df2)





# include disorder in df
fixed_disorder = []
for n,i in enumerate(df.iupred.values):
    i = [t for t in i if t!=""]
    fixed_disorder.append(np.array(i).astype(float))
    
df.iupred = fixed_disorder





# include probability of a-helix in df
helicity  = {}
helicity2 = {}

#for i in files:
d, d2  = read_psipred_results(analysis_home + '/data/secondary_structure/total.results') #+i)

for i in d2.keys():
    if re.search(",", i):
        print('oh oh!!!')

helicity.update(d)
helicity2.update(d2)


# joined datasets
df2 = pd.DataFrame([helicity2]).T
idx = df2.index.intersection(df.index)
df2 = df2.loc[idx]

df = pd.concat([df.loc[idx],df2], axis=1)
df.columns = ['sequence', 'secondStruct', 'disorder', 'predictions', 'iupred', 'helicity_proba']

del(df2)


                                        ## SGD ##
# collect data from SGD 
SGD = pd.read_csv('https://downloads.yeastgenome.org/curation/chromosomal_feature/SGD_features.tab', index_col=3, sep='\t', header=None)
SGD = SGD[SGD[1]=='ORF'][4]

                                        ## TF ##
# Steve's list of TFs
# long list including potential NON-TF
tf_full = pd.read_csv(analysis_home + '/data/TFs.csv')
tf_full = tf_full['Systematic name'].values

# short list excluding potential False TF
tf_short = pd.read_csv(analysis_home + '/data/TFs_small.csv')
tf_short = tf_short['Systematic name'].values

                                        ## Nuclear ##
# Are tf enriched in the Nucleus?
localization = pd.read_csv(analysis_home + '/data/proteome_localization.csv', index_col=0)
X = localization.iloc[:,1] 
nuclear = [i for i in set(X) if re.search("nucl",i)]
X = pd.DataFrame([1 if i in nuclear else 0 for i in X], index=localization.index, columns=['loc'])
nuclear = X[X['loc']==1].index


total_idx = df.index.intersection(X.index)
nuclear_idx = nuclear.intersection(total_idx)
tf_full_idx = set(tf_full).intersection(total_idx)
tf_short_idx = set(tf_short).intersection(total_idx)

print('{} in tf_full\n{} in tf_short\n{} in total\n{} in nuclear\n'.format(
    len(tf_full_idx), len(tf_short_idx), len(total_idx), len(nuclear_idx)))


# set cutoff to predict TADs in the proteome
cutoff=0.8 

results = np.zeros(shape=(df.shape[0],4))
for n,prot in enumerate(df.predictions):
    results[n] = predict_motif_statistics(prot, cutoff)
    
results = pd.DataFrame(results, index=df.index, columns = ['length', 'start_position', 'gral_mean', 'mean_longest_region'])

# first question
from scipy.stats import hypergeom

def enrichment(M,n,N, x):
    '''
        Calculates the enrichment of genes in a Pugh group
        
        M = total number of proteins (population size)
        n = group with score>cutoff longer than 5 residues (# of successes in population) 
        N = total of group taken into account (e.g. tf_full) (sample size)
        x = number of successes (withTAD) in that group (drawn successes)
    '''
    if N==0: N=N+0.5
    #enriched = (x / (n-x+0.0001)) / ( (N-x+0.0001)/(M-n+0.0001) ) 
    enriched = (x/N)/(n/M)
    p_val = hypergeom.sf(x-1, M, n, N)
    
    return enriched, p_val


M = len(total_idx)
n = len(tf_full_idx)
N = len(nuclear_idx)
x = len(nuclear_idx.intersection(tf_full_idx))
e,p = enrichment(M,n,N,x)
print(M,n,N,x,e,p)


enrich, pval = [],[]

for l in range(70):
    withTADs = results[results.length>=l].index
    nuclear_TADs = withTADs.intersection(nuclear_idx)
    
    M = len(results.index)
    n = len(withTADs) 
    N = len(tf_short_idx) #.intersection(nuclear_idx))
    x = len(tf_short_idx.intersection(withTADs)) 

    e,p = enrichment(M,n,N,x)
    enrich.append(e)
    pval.append(p)

    
enrich2 = np.convolve(enrich, np.ones(4)/4, 'same')
pval2 = np.convolve(pval, np.ones(4)/4,'same')
    
    
ys = [np.mean(enrich2[i:i+10]) for i in np.arange(5, len(enrich2)-10, 10)]
ys2 = [np.mean(pval2[i:i+10]) for i in np.arange(5, len(pval2)-10, 10)]

####################
# plot enrichments #
####################

f,(ax1,ax2) = plt.subplots(2, sharex=True, figsize=(7,7))
ax1.bar(np.arange(len(ys)), ys, alpha=0.4, color='b')
ax1.set_xticks(np.arange(6))
ax1.set_xticklabels(np.arange(10,70,10))
ax2.scatter(np.arange(len(ys2)), ys2, alpha=0.4, color='r', s=100, marker='s')
ax1.set_ylabel('enrichment', color='b', fontsize=16)
ax1.plot(np.arange(6)-0.5, [1]*6, ls='--', alpha=0.4, color='k', lw=3)
ax2.plot(np.arange(len(ys2)), ys2, alpha=0.4, color='r')
ax2.set_ylabel('p value', color='r', fontsize=16)
ax2.set_yscale('log')
ax1.set_xlim(-0.5,4.5)

#print(ax2.get_ylim())
ax2.set_ylim((0.001,1)[::-1])
matplotlib.rcParams.update({'font.size': 16})
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)


###############################################################################
# plot percentiles of disorder and secondary structure for the while proteome #
###############################################################################

lenCutoff=5
get_percentiles(feature='iupred', df=df, results=results, lenCutoff=lenCutoff)

plt.figure(figsize=(10,5))

_25,_50,_75 = get_percentiles(df=df, results=results, feature='iupred')
ax = plt.subplot(1,2,1)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.01, "pre-tad", fontsize=17)
ax.text(47, -0.01, "TAD", fontsize=17)
ax.text(75, -0.01, "post-TAD", fontsize=17)
ax.set_ylabel('predicted disorder', fontsize=18)
ax.set_title('Disorder', fontsize=18);


_25,_50,_75 = get_percentiles(df=df, results=results, feature='helicity_proba')
ax = plt.subplot(1,2,2)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.1, "pre-tad", fontsize=17)
ax.text(47, -0.1, "TAD", fontsize=17)
ax.text(75, -0.1, "post-TAD", fontsize=17)
ax.set_ylabel('predicted helicity', fontsize=18)
ax.set_title('Helicity', fontsize=18);

ax.tick_params(axis='both', which='major', labelsize=17)
plt.tight_layout()


######################################################################################
# plot percentiles of disorder and secondary structure for 132 transcription factors #
######################################################################################


plt.figure(figsize=(10,5))

_25,_50,_75 = get_percentiles(feature='iupred', df = df.loc[tf_short_idx], results=results)
ax = plt.subplot(1,2,1)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.01, "pre-tad", fontsize=17)
ax.text(47, -0.01, "TAD", fontsize=17)
ax.text(75, -0.01, "post-TAD", fontsize=17)
ax.set_ylabel('predicted disorder', fontsize=18)
ax.set_title('Disorder', fontsize=18);


_25,_50,_75 = get_percentiles(feature='helicity_proba', df = df.loc[tf_short_idx], results=results)
ax = plt.subplot(1,2,2)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.1, "pre-tad", fontsize=17)
ax.text(47, -0.1, "TAD", fontsize=17)
ax.text(75, -0.1, "post-TAD", fontsize=17)
ax.set_ylabel('predicted helicity', fontsize=18)
ax.set_title('Helicity', fontsize=18);

ax.tick_params(axis='both', which='major', labelsize=17)
plt.tight_layout()


#############################################################################
# plot percentiles of disorder and secondary structure for nuclear proteins #
#############################################################################

plt.figure(figsize=(10,5))

_25,_50,_75 = get_percentiles(feature='iupred', df = df.loc[nuclear_idx], results=results)
ax = plt.subplot(1,2,1)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.01, "pre-tad", fontsize=17)
ax.text(47, -0.01, "TAD", fontsize=17)
ax.text(75, -0.01, "post-TAD", fontsize=17)
ax.set_ylabel('predicted disorder', fontsize=18)
ax.set_title('Disorder', fontsize=18);


_25,_50,_75 = get_percentiles(feature='helicity_proba', df = df.loc[nuclear_idx], results=results)
ax = plt.subplot(1,2,2)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.1, "pre-tad", fontsize=17)
ax.text(47, -0.1, "TAD", fontsize=17)
ax.text(75, -0.1, "post-TAD", fontsize=17)
ax.set_ylabel('predicted helicity', fontsize=18)
ax.set_title('Helicity', fontsize=18);

ax.tick_params(axis='both', which='major', labelsize=17)
plt.tight_layout()

########################################################################
# plot percentiles of disorder and secondary structure for nuclear TF  #
########################################################################

special_idx = set(nuclear_idx).intersection(set(tf_short_idx))
plt.figure(figsize=(10,5))

_25,_50,_75 = get_percentiles(feature='iupred', df = df.loc[special_idx], results=results)
ax = plt.subplot(1,2,1)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.01, "pre-tad", fontsize=17)
ax.text(47, -0.01, "TAD", fontsize=17)
ax.text(75, -0.01, "post-TAD", fontsize=17)
ax.set_ylabel('predicted disorder', fontsize=18)
ax.set_title('Disorder', fontsize=18);


_25,_50,_75 = get_percentiles(feature='helicity_proba', df = df.loc[special_idx], results=results)
ax = plt.subplot(1,2,2)
ax.fill_between( np.arange(len(_25)),_75,_25, alpha=0.2, color='gray')
ax.plot(_50, label="50%", lw=3, c='k')
ax.set_xticks([0,50,55,100])
ax.set_xticklabels([])
ax.text(5, -0.1, "pre-tad", fontsize=17)
ax.text(47, -0.1, "TAD", fontsize=17)
ax.text(75, -0.1, "post-TAD", fontsize=17)
ax.set_ylabel('predicted helicity', fontsize=18)
ax.set_title('Helicity', fontsize=18);

ax.tick_params(axis='both', which='major', labelsize=17)
plt.tight_layout()
