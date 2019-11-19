#!/bin/python

# correspondence with Lukaz and Johannesd on Mon 2/5/2018 9:27 AM:
#             ARG1         ILV6       ARG3
#    bin1 =   0-200       0-200      120- 400
#    bin2 = 370-570     300-400      400- 640
#    bin3 = 580-850     410-500      640-1000
#    bin4 = 850-inf     510-inf     1010-7500

# binning data in Canto_1/052716
# counts from data in db?

ARG1_FACS_FU  = [[0, 200], [370, 570], [580, 850], [850, None]]
ARG1_FACS_RFU = [1.0, 4.7, 7.15, 9.85]

ILV6_FACS_FU  = [[0, 200], [300, 400], [410, 500], [510, None]]
ILV6_FACS_RFU = [1.0, 3.5, 4.55, 5.55]

ARG3_FACS_FU  = [[120, 400], [400, 640], [640, 1000], [1010, None]]
ARG3_FACS_RFU = [1.0, 2.0, 3.154, 4.577]

# get validation sequences and mean of fluorescence from this file
exp_valid = analysis_home+'/data/activity-measurements-for-18-peptide-seqs-26Jan2018.txt'
gfp = pd.read_csv(exp_valid, index_col=0, sep='\t') #['GFP fluo']

# get counts in bg and bins1-4 from the database
conn = sqlite3.connect(db)
cursor = conn.cursor()
k,v = [],[]
for sequence in gfp.index:
    cursor.execute('SELECT bg,bin1,bin2,bin3,bin4 from positives WHERE seq="' + sequence + '"')
    v.append(cursor.fetchall()[0])
    k.append(sequence)
counts = pd.DataFrame(v, index=k)

# merge the two tables
df = pd.concat([gfp['GFP fluo'],counts], axis=1)
df['score'] = np.sum(df.iloc[:,2:]*ARG3_FACS_RFU, axis=1) / df.loc[:,1]
X = df[['score','GFP fluo']].values
x,y = np.log(X[:,0]), X[:,1]

# fit line to points
slope, intercept, r_value, p_value, std_err = linregress(x,y)

sns.set_style('ticks')
plt.figure(figsize=(4,3.7))
fl = [min(x)-1,max(x)+1]
plt.plot(fl, [slope*i+intercept for i in fl], ls='--', c='r')
plt.scatter(x,y, c='k')
plt.xlabel('AD Enrichment score', fontsize=14, fontweight='bold')
plt.ylabel('Fluorescence intensity', fontsize=14, fontweight='bold')
plt.text(1.5,1200, 'R = {:.2f}\np-value = {:.4f}'.format(r_value, p_value), fontsize=14)
plt.xticks(fontsize=14)
plt.xlim(fl)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('figure_1C.png', dpi=600)
