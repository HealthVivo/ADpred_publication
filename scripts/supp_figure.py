
# supp figure
def open_prot(prefix, which='ss'):
    ss,seq=[],[]
    Dir = analysis_home + '/data/same_aa_diff_scores/'

    f = open(Dir + prefix + '.' + which)
    for i in f:
        if i[0]=='#' or len(i)<3: continue
        ss.append(i.strip().replace('  ',' ').replace('C','-').split(' ')[2])
    f.close()

    f=open(Dir + prefix + '.fasta')
    seq = [i for i in f if i[0]!=">"][0]

    return np.array([seq, ''.join(ss)])


Dir = analysis_home + '/data/same_aa_diff_scores/'
ss2_files= [Dir+i for i in os.listdir(Dir) if i[-3:]=='ss2']

filesP = [i for i in os.listdir(Dir) if i[-5:]=='+.ss2']
filesN = [i for i in os.listdir(Dir) if i[-5:]=='-.ss2']
x = []

for n,i in enumerate(filesN+filesP):
    a,b = open_prot(i[:-4])
    x.append(prepare_ohe([a,b]))

x = np.array(x)
y = np.hstack(ADPred.predict(x.reshape(len(x),30,23,1)))
