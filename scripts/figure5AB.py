matplotlib.rcParams.update({'font.size': 24})

#GCN4 sequences
name = 'gcn4'
seq = read_fasta('data/' + name + '.fasta')
ss = read_horiz('data/' + name + '.horiz')

min_range = 107
max_range = 137
seq = seq[min_range:max_range]
ss = ss[min_range:max_range]


def predict_seq(ohe_data, ADPred):
    '''
        function outputs probability of TAD for a 30 AA long segment
        INPUT: ohe_data including fasta and ss
        OUTPUT: predictions array (1D) of probabilities over the length of the protein
        NOTE: This functions slides along 30aa long windows and predict the center of the protein.
              The first and last 15 residues are repeated on purpose.
    '''
    # exit if data is not correctly shaped.
    if ohe_data.shape[1] != 23:
        print('shape should be (30,23), wrong amino acid number')
        return
    if ohe_data.shape[0] != 30:
        print('shape should be (30,23), sequence is not 30 long')
        return

    seq = ohe_data.reshape(1,30,23,1)
    prediction = ADPred.predict(seq)

    return prediction[0][0]

def predict_single(A):
    # all selected positions to mutate

    positions = np.arange(0,30)

    # all these combinations get to 20*42=840 single mutants


    mutant_info = []
    predictions = []

    # go over each of the positions to be mutated
    for pos in positions:
        residues = {pos:A} #aa}

        # go over each mutant and assume ALL keep the same secondary structure
        for mutant in mutate_protein(seq, residues=residues):

            # keep track of the mutant 
            mutant_info.append([mutant])

            # prepare data for deep model and make prediction
            data = prepare_ohe([mutant,ss])
            prediction = predict_seq(data, ADPred)

            # predict TAD over the whole sequence and sum it up
            predictions.append( np.mean(prediction) )

    return predictions, mutant_info



predictions = []
for i in aa:
    pred, _ = predict_single(i)
    predictions.append(pred)

predictions = pd.DataFrame(predictions, index=aa, columns=list(seq))

orig_score = predict_seq(prepare_ohe([seq,ss]), ADPred)
print(orig_score)

'''
# create the colormap to be called under "salma_cmap"
salma_suggestion = {'blue':  ((0.0, 1.0, 1.0),
                              (0.97, 1.0, 1.0), # red 
                              (1.0, 0.0, 0.0)), # blue

                    'green': ((0.0, 0.0, 0.0),
                              (0.97, 1.0, 1.0),
                              (1.0, 0.0, 0.0)),

                    'red':   ((0.0, 0.0, 0.0),
                              (0.97, 1.0, 1.0),
                              (1.0, 1.0, 1.0))
                     }
                     
plt.register_cmap(name='salma_cmap', data=salma_suggestion)
'''
import matplotlib.colors as mcolors
divnorm = mcolors.DivergingNorm(vmin=predictions.min().min(), vcenter=orig_score, vmax=predictions.max().max()) #prediction is a dataframe here


sns.set_style('ticks')
f,ax = plt.subplots(1,1, figsize=(20,12))
#sns.heatmap(predictions, ax=ax, cmap='salma_cmap', square=True, cbar_kws={'shrink': 0.7, 'ticks': (0, 0.2, 0.4, 0.6, 0.8, 0.957563, 1)}) # viridis cmap='RdBu_r'
sns.heatmap(predictions, ax=ax, cmap='RdBu_r', square=True, cbar_kws=dict(ticks=(0,0.2,0.4,0.6,0.8,orig_score,1), shrink=0.8)) #, linewidths=0.2, linecolor='black') #,norm=divnorm)

plt.xlabel('GCN4 cAD', fontsize=22)
plt.ylabel('Mutations', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=40)
plt.text(35.5, 17.4, '(cAD)', weight = 'bold', fontsize=28)
plt.plot([12.05, 12.05], [0,20], lw=5,c='k')
plt.plot([17.05, 17.05], [0,20], lw=5,c='k')

ax.set_ylim(-0.5, 20.5)
ax.spines['bottom'].set_position(('data',0))


plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

cols = ['g','k','g','k','k','g'] + 6*['k'] + ['r','k','k','r','r'] + ['k']*13
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), cols):
    ticklabel.set_color(tickcolor)

fontProperties = {'family':'sans-serif','sans-serif':['Oswald'],
    'weight' : 'normal', 'size' : 35}
ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    size=35, weight='normal', stretch='normal')

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)

# figure 5B

df = pd.read_csv(analysis_home+'/data/pnas2015.tsv', sep='\t', header=None, index_col=0)
df = df[1]
df.loc['AVWESLFSS']=7.41

heatmap = np.zeros(shape=(20,9))

for i in df.index:
    for n,(j,k) in enumerate(zip(i,'AVWESLFSS')):
        missmatch=0
        if j != k:
            if missmatch==0: missmatch=n
            else: print('not good')
        heatmap[aa.index(j),n] = df.loc[i]
        
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
heatmap = pd.DataFrame(heatmap, columns=['A','V','W','E','S','L','F','S','S'], index=aa)
ids = np.hstack([np.arange(51), np.arange(52,180)])

x,y = np.hstack(predictions.iloc[:,10:19].values)[ids], np.hstack(heatmap.values)[ids]
y_hat = [np.log(p/(1-p)) for p in x]
slope, intercept, r_value, p_value, std_err = linregress(np.log(y),y_hat)

matplotlib.rcParams.update({'font.size': 14})

f,ax = plt.subplots(1, figsize=(5,5))
ax.scatter(y_hat, np.log(y), c='k', alpha=0.6, s=10, marker='o')
ys = [np.min(np.log(y)), np.max(np.log(y))]
ax.plot([intercept+slope*i for i in ys], ys, ls='--', lw=2)
ax.set_yticks(np.log([0.03,0.1,0.3,1]))
ax.set_yticklabels([0.03,0.1,0.3,1])
ax.set_xticks([-2,2,6])
ax.set_title('Rs={:.2f}, p={}'.format(r_value, p_value))
ax.set_ylabel('Warfield et al', fontsize=24)
ax.set_xlabel('- logit(predictions)', fontsize=24)
ax.grid(c='grey',ls='--', alpha=0.4)

matplotlib.rcParams.update(matplotlib.rcParamsDefault)




# supplementary figures
matplotlib.rcParams.update({'font.size': 24})
#INO2 sequences
name = 'gal4'
seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')

min_range = 847
max_range = 877
seq = seq[min_range:max_range]
ss = ss[min_range:max_range]

predictions = []
for i in aa:
    pred, _ = predict_single(i)
    predictions.append(pred)
    
orig_score = predict_seq(prepare_ohe([seq,ss]), ADPred)
print(orig_score)
predictions = pd.DataFrame(predictions, index=aa, columns=list(seq))


import matplotlib.colors as mcolors
divnorm = mcolors.DivergingNorm(vmin=predictions.min().min(), vcenter=orig_score, vmax=predictions.max().max()) #prediction is a dataframe here


sns.set_style('ticks')
f,ax = plt.subplots(1,1, figsize=(20,12))
#sns.heatmap(predictions, ax=ax, cmap='salma_cmap', square=True, cbar_kws={'shrink': 0.7, 'ticks': (0, 0.2, 0.4, 0.6, 0.8, 0.957563, 1)}) # viridis cmap='RdBu_r'
sns.heatmap(predictions, ax=ax, cmap='RdBu_r', square=True, cbar_kws=dict(ticks=(0,0.2,0.4,0.6,0.8,orig_score,1), shrink=0.8)) #, linewidths=0.2, linecolor='black',norm=divnorm)

plt.xlabel('GAL4 nAD', fontsize=22)
plt.ylabel('Mutations', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.text(35.5,17.5, '(Gal4 cAD)', weight = 'bold', fontsize=28)
#plt.plot([12.05, 12.05], [0,22], lw=5,c='k')
#plt.plot([17.05, 17.05], [0,22], lw=5,c='k')
ax.set_ylim(-0.5, 20.5)
ax.spines['bottom'].set_position(('data',0))

plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

cols = ['g','k','g','k','k','g'] + 6*['k'] + ['r','k','k','r','r'] + ['k']*13
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), cols):
    ticklabel.set_color(tickcolor)

fontProperties = {'family':'sans-serif','sans-serif':['Oswald'],
    'weight' : 'normal', 'size' : 35}
ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    size=35, weight='normal', stretch='normal')

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)






matplotlib.rcParams.update({'font.size': 24})
#INO2 sequences
name = 'ino2'
seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')

min_range = 11
max_range = 41
seq = seq[min_range:max_range]
ss = ss[min_range:max_range]

predictions = []
for i in aa:
    pred, _ = predict_single(i)
    predictions.append(pred)
    
orig_score = predict_seq(prepare_ohe([seq,ss]), ADPred)
    
predictions = pd.DataFrame(predictions, index=aa, columns=list(seq))

import matplotlib.colors as mcolors
divnorm = mcolors.DivergingNorm(vmin=predictions.min().min(), vcenter=orig_score, vmax=predictions.max().max()) #prediction is a dataframe here

sns.set_style('ticks')
f,ax = plt.subplots(1,1, figsize=(20,12))
sns.heatmap(predictions, ax=ax, cmap='RdBu_r', square=True, cbar_kws=dict(ticks=(0,0.2,0.4,0.6,0.8,orig_score,1), shrink=0.8)) #, linewidths=0.2, linecolor='black',norm=divnorm)
plt.xlabel('INO2 nAD', fontsize=22)
plt.ylabel('Mutations', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.text(35.4, 17.0, '(Ino2 nAD)', weight = 'bold')
#plt.plot([12.05, 12.05], [0,22], lw=5,c='k')
#plt.plot([17.05, 17.05], [0,22], lw=5,c='k')
ax.set_ylim(-0.5, 20.5)
ax.spines['bottom'].set_position(('data',0))

plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

cols = ['g','k','g','k','k','g'] + 6*['k'] + ['r','k','k','r','r'] + ['k']*13
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), cols):
    ticklabel.set_color(tickcolor)

fontProperties = {'family':'sans-serif','sans-serif':['Oswald'],
    'weight' : 'normal', 'size' : 35}
ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    size=35, weight='normal', stretch='normal')

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)






import matplotlib.colors as mcolors
matplotlib.rcParams.update({'font.size': 24})

# for figure S2

#INO2 sequences
name = 'ino2'
seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')

min_range = 114
max_range = 144
seq = seq[min_range:max_range]
ss = ss[min_range:max_range]

predictions = []
for i in aa:
    pred, _ = predict_single(i)
    predictions.append(pred)
    
orig_score = predict_seq(prepare_ohe([seq,ss]), ADPred)
print(orig_score)

predictions = pd.DataFrame(predictions, index=aa, columns=list(seq))
sns.set_style('ticks')
divnorm = mcolors.DivergingNorm(vmin=predictions.min().min(), vcenter=orig_score, vmax=predictions.max().max()) #prediction is a dataframe here

f,ax = plt.subplots(1,1, figsize=(20,12))
sns.heatmap(predictions, ax=ax, cmap='RdBu_r', square=True, cbar_kws=dict(ticks=(0,0.2,0.4,0.6,0.8,orig_score,1), shrink=0.8)) #, linewidths=0.2, linecolor='black',norm=divnorm)
plt.xlabel('INO2 cAD', fontsize=22)
plt.ylabel('Mutations', fontsize=22)
ax.tick_params(axis='both', which='major', labelsize=20)
plt.text(35.5,3.8, '(Ino2 cAD)', weight = 'bold')
#plt.plot([12.05, 12.05], [0,22], lw=5,c='k')
#plt.plot([17.05, 17.05], [0,22], lw=5,c='k')
ax.set_ylim(-0.5, 20.5)
ax.spines['bottom'].set_position(('data',0))

plt.rc('font', weight='bold')
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

cols = ['g','k','g','k','k','g'] + 6*['k'] + ['r','k','k','r','r'] + ['k']*13
for ticklabel, tickcolor in zip(plt.gca().get_xticklabels(), cols):
    ticklabel.set_color(tickcolor)

fontProperties = {'family':'sans-serif','sans-serif':['Oswald'],
    'weight' : 'normal', 'size' : 35}
ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    size=35, weight='normal', stretch='normal')

for label in ax.get_yticklabels():
    label.set_fontproperties(ticks_font)
for label in ax.get_xticklabels():
    label.set_fontproperties(ticks_font)
