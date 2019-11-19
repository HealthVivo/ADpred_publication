from initialize_notebook import *
from scipy.stats import linregress

Dir = analysis_home + '/data/same_aa_dif_scores/'
ss2_files= [Dir+i for i in os.listdir(Dir) if i[-3:]=='ss2']

filesP = [i for i in os.listdir(Dir) if i[-5:]=='+.ss2']
filesN = [i for i in os.listdir(Dir) if i[-5:]=='-.ss2']
x = []

def open_prot(prefix, which='ss'):
    ss,seq=[],[]
    
    f = open(Dir + prefix + '.' + which)
    for i in f:
        if i[0]=='#' or len(i)<3: continue
        ss.append(i.strip().replace('  ',' ').replace('C','-').split(' ')[2])
    f.close()
    
    f=open(Dir + prefix + '.fasta')
    seq = [i for i in f if i[0]!=">"][0]
    
    return np.array([seq, ''.join(ss)])

def make_ohe(SEQ,SS):
    ohe = np.zeros(shape=(30,23))
    for n,i in enumerate(SEQ):
        ohe[n, aa.index(i)]=1    
    for n,i in enumerate(SS):
        ohe[n, ss.index(i)+20]=1

    return ohe

for n,i in enumerate(filesN+filesP):
    a,b = open_prot(i[:-4])
    x.append(make_ohe(a,b))
            
x = np.array(x)
y = np.hstack(ADPred.predict(x.reshape(len(x),30,23,1)))

idxp = ['A1+','B1+','C1+','D1+','E1+','A2+','B2+','C2+','D2+','E2+']
idxn = ['A1-','B1-','C1-','D1-','E1-','A2-','B2-','C2-','D2-','E2-']

expP = np.zeros(len(idxp))
expN = np.zeros(len(idxn))

theP = np.zeros(len(idxp))
theN = np.zeros(len(idxn))



#ARG3
NN_scores = y
filesN+filesP
experimental = [0.112876623, 0.15547652, 0.11527561, 0.185727112, 0.210297121, 0.161050794, 0.111915757, 0.14484829, 0.107696827,
                0.229597571, 0.589282217, 0.165045984, 0.556443746, 0.281672194, 0.205324748, 0.233521384, 0.54408126, 0.155961526]

f,ax1 = plt.subplots(1,figsize=(5,5))

ax1.scatter (np.log(NN_scores), np.log(experimental))
for xcoord,ycoord,text in zip(np.log(NN_scores), np.log(experimental), filesN+filesP):
    ax1.text(xcoord,ycoord,text[8:11])
    
slope, intercept, r_value, p_value, std_err = linregress(np.log(NN_scores), np.log(experimental))
plt.title('ARG3, R={:.2f}, p={:.3f}'.format(r_value, p_value))
plt.xlabel('log-scores')
plt.ylabel('log-experimental');

#HIS4
NN_scores = y
filesN+filesP
experimental = [0.022730735, 0.038230405, 0.041479216, 0.038881456, 0.087696036, 0.023620793, 0.046929963, 0.048461607, 0.052217961,
                0.192587102, 0.485438934, 0.089917386, 0.505061115, 0.247023568, 0.104519027, 0.307483538, 0.379600415, 0.189996959]

f,ax1 = plt.subplots(1,figsize=(5,5))

ax1.scatter (np.log(NN_scores), np.log(experimental))
for xcoord,ycoord,text in zip(np.log(NN_scores), np.log(experimental), filesN+filesP):
    ax1.text(xcoord,ycoord,text[8:11])

slope, intercept, r_value, p_value, std_err = linregress(np.log(NN_scores), np.log(experimental))
plt.title('HIS4, R={:.2f}, p={:.3f}'.format(r_value, p_value))
plt.xlabel('log-scores')
plt.ylabel('log-experimental');
