from initialize_notebook import *
from itertools import product
dipeptides = np.array([''.join(i) for i in product(aa, repeat=2)])

# result arrays for the 4 libraries // positives // negatives // random_positives // random_negatives //
di_positives = np.zeros(shape=(len(positives),400))
di_negatives = np.zeros(shape=(len(negatives),400))

# fill result arrays
for n,i in enumerate(dipeptides):
    di_positives[:,n] = [seq.count(i) for seq in np.array(positives)[:,0]]
    di_negatives[:,n] = [seq.count(i) for seq in np.array(negatives)[:,0]]

def find_combinations(seq, aa_combi, distance=1):
    results = np.zeros(400)
    
    for i in range(len(seq)-distance):
        _2mer = ''.join([seq[i], seq[i+distance]])
        results[aa_combi.index(_2mer)] +=1        

    return results    
    
def permutateSequence(seq):
    '''
        function returns a sequence with same aminoacids as the input but 
        with shuffled positions
        INPUT: seq.sequence of aminoacids
        OUTPUT: sequence of same aminoacids in different order
    '''
    idx = np.random.permutation(np.arange(len(seq)))
    return ''.join([seq[i] for i in idx])

permutated = np.hstack([[permutateSequence(i) for i in np.array(positives)[:,0]] for n in range(3)])

aa_combi = [''.join(i) for i in product(aa, repeat=2)]
for n in range(1,29):
    P_varName = 'r_' + str(n)   
    vars()[P_varName] = np.sum([find_combinations(i, aa_combi, n) for i in np.array(permutated)], axis=0)
    
def Norm(x): return x / np.sum(x)

dipeptide = 'DW'
di_inverse = ''.join([dipeptide[1], dipeptide[0]])

plt.figure(figsize=(12,7))
plt.title(dipeptide)

for n, r in zip([dipeptide, di_inverse], [np.arange(29), np.arange(29)*-1]):
    N = aa_combi.index(n)
    
    Ap, An = np.zeros(29), np.zeros(29)
    for i in range(1,29):
        namep = 'p_'+str(i)
        Ap[i] = eval(namep)[N]
        namen = 'r_'+str(i)
        An[i] = eval(namen)[N]
    
        
    plt.bar(r, Norm(Ap), color='g', alpha=0.3)
    plt.bar(r, Norm(An), color='r', alpha=0.3)
    plt.bar(r, Norm(Ap)-Norm(An), color='b')
