from initialize_notebook import *


#sys.path.append(analysis_home + '/libraries')
from null_distribution import *

# load genome sequence of 180 TF from Stark's publication
Stark_data_genome = read_fasta(analysis_home + '/data/Stark_data/GSE114387_TFbb_list180_linear.fa')

# get length of the genome
genome_length = len(Stark_data_genome)

# load bed files 
gfp_p = read_bed(analysis_home + '/data/Stark_data/GSM3140923_short-library_GFP+_1+.wig', genome_length)
gfp_m = read_bed(analysis_home + '/data/Stark_data/GSM3140921_short-library_GFP-_1+.wig', genome_length)

# Load annotations from stark data
Stark_data_annotations = pd.read_csv(analysis_home + '/data/Stark_data/TFbb_list180_linear_anno.bed', delimiter='\t', 
                                     header=None, names=['_','start','end']+['a','b','c','d','e','f','g','h'], index_col=3).iloc[:,1:3]

# exclude a problematic gene
Stark_data_annotations = Stark_data_annotations.drop('Su(var)2-10_FBtr0088576', axis=0)

# original factor and position indexes
f = list(Stark_data_annotations.index)  

# set p_dict outside of the loop since this is the most time consuming step
p_dict = make_p_dict(Stark_data_annotations, 
                     ADPred, 
                     fastas_folder=analysis_home + '/data/Stark_data/',
                     horiz_folder=analysis_home + '/data/Stark_data/')

#Stark_data_annotations.head(5)



good_samples = np.array([f[i] for i in [7, 18,  68, 82, 154]])

normalize = lambda arr1, arr2: np.vstack( [(arr1-np.min(arr1)) / (np.max(arr1)-np.min(arr1)), 
                                     (arr2-np.min(arr1)) / (np.max(arr1)-np.min(arr1))])

plt.figure(figsize=(16,48))

for n, k in enumerate(good_samples):
    plt.subplot(9,3,n+1)
    plt.fill_between(np.arange(len(p_dict[k])), p_dict[k], alpha=0.4)
    start,end = Stark_data_annotations.loc[k].values
    p,m = normalize(gfp_p[start:end+1:3], gfp_m[start:end+1:3])
    plt.plot(p, c='g')
    plt.plot(m, c='r')
    
    # plot disorder
    FileName = "analysis_home + '/data/Stark_data/"+ k +".fasta"
    
    plt.title(k)
    
    n+=1
