'''
    So far, I've been running this script with slurm, specifying the output.file. Now the output (which contains statistics about the filters) 
	could be lost if not specified a file as output: MAKE THE NECESARY MODIFICATIONS!! 
	
	NOTE: The run takes ~30 min in 2 cpus from rhino. Parallelize it with multiprocessing or even splitting the files in bash and run the subfiles
	in different cpus? (or re-write in c?)
'''

import numpy as np
import matplotlib.pyplot as plt
import sys,re,os
from termcolor import colored
import subprocess, collections
import uuid 

INFO = '\n\nrun: ' + colored('python translate.py <path> <filename>', attrs=['bold']) \
       + colored(' (optional) -o outputfile_filename (no extension! this is automatic)', 'green') + '\n' + \
       'OUTPUT: .counter and .fasta files\n\n'

# Names of these variables are self explained. These variables should have global properties
start_primer = 'ATGTCTGCA' #MSA
end_primer = 'GGCGACAATGACATT'
wt = 'ATGTCTGCAGGCGACAATGACATTCCTGCAGGCACTGACGATGTGTCATTGG'
gene_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

Num_of_proteins = 1000000 # fair assumption... I don't think I could have more than that
Not using this now. It made things slower. I am just numbering.
#unique_indexes = list(set([uuid.uuid4().hex[:10] for i in range(Num_of_proteins)]))

def main():

    if len(sys.argv)==1:
        print(INFO)
        sys.exit(1)

    # define the filters to exclude problematic samples
    noStart, noStop, frameShift, lowQuality, shortSequence, early_stops = 0,0,0,0,0,0
    # file names of input and output files
    fastq_filename = sys.argv[1]
    proteins_output_filename = '_'.join(fastq_filename.split('/')[:2])
    proteins_output_filename = [sys.argv[sys.argv.index('-o')+1] if '-o' in sys.argv else proteins_output_filename][0]
    counter_output = open(proteins_output_filename + '.counter', 'w')
    fasta_output = open(proteins_output_filename + '.fasta','w')
    # load the file iterator
    fastq = open(fastq_filename, 'r')
    aa_sequences = []

    # wc -l, etc.
    total = int(subprocess.Popen(['wc', '-l', sys.argv[1]], \
            stdout=subprocess.PIPE).communicate()[0].decode("utf-8").strip().split(' ')[0])
    percents, n = [int(total/i) for i in np.arange(100,0,-10)], 0

    # go over lines and test the functions
    while True:
        # output the situation
        n+=1
        if n in percents:
            print('{}% done'.format(percents.index(n)*10))
        try:
            name, seq, sep, qual = next(fastq), next(fastq), next(fastq), next(fastq)
            # first check if sequence is short, no Stop, no Start or frameshifts
            positions = sequence_check(seq) 
            # if something is wrong, REPORT! and NEXT, else: keep checking the relevant sub-sequence
            if len(positions) <2: 
                if positions[0]=='noStart': noStart+=1
                elif positions[0]=='noStop': noStop+=1
                elif positions[0]=='frameShift': frameShift+=1
                elif positions[0]=='shortSequence': shortSequence+=1
                #vars()[positions[0]] += 1            ### WHY ON EARTH THIS IS NOT WORKING !!!! 
                continue
            # Check the quality of the sequencing. NEXT if quality is low 
            quality = quality_check(qual[positions[0]:positions[1]])
            if not quality: 
                lowQuality +=1 
                continue
            # All right, keep the aa sequence if didn't happen the special situation in which the whole sequence
            # is composed by the 2 primers without AD sequence in it (e.g. the end_primer was found in the 
            # middle and has less mismatches than the real end_primer towards the end of the sequence) or 
            # there are Stop codons within the sequence
            sequence = seq[positions[0]:positions[1]]
            if positions[1]-positions[0]<40: continue
            translation = translate_dna(sequence)
            if '_' in translation: 
                early_stops+=1
                continue
            aa_sequences.append(translation)
        except StopIteration:
            break

    aa_counter = collections.Counter(aa_sequences)  # dictionary of {sequence count}
    aa_counter = aa_counter.most_common()           # sort them in decreasing order
    # string to write into file
    aa_counter = '\n'.join([ ','.join([k,str(v)]) for k,v in aa_counter ])
    counter_output.write( str(aa_counter) ) 
	# I had to change the uuid to this, since either there were not enough indexes for the pre_sorting library or it took to much memory?
    aa_fasta = '\n'.join(['>{}{:015d}\n{}'.format('uniqueN',n,k) for n,k in enumerate(set(aa_sequences))])
    fasta_output.write(aa_fasta)
    # report statistics
    totals = len(aa_sequences) + early_stops + noStart + noStop + frameShift + lowQuality + shortSequence   
    percents = [i*100.0/totals for i in [early_stops,noStart,noStop,frameShift,lowQuality,shortSequence]] 
    print('#, early stops, No 5-primer, no3-primer, frame shift, low quality, short sequence\n')
    print('absolute', early_stops, noStart, noStop, frameShift, lowQuality, shortSequence)
    print('%, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(*percents))

def translate_dna(dna_sequence):
    '''translate DNA to protein'''    
    seqLen = len(dna_sequence)
    # Avoid crashing if not multiple of 3 and check has only allowed nucleotides
    seqLen = seqLen - seqLen%3
    if set([i for i in dna_sequence.strip()]).issubset(set(['A','C','T','G'])):
        return(''.join([gene_code[dna_sequence[i:i+3]] for i in range(0,seqLen,3) ]))

def seq_distance(a, b):
    # define a function to look for the 3' primer, since I should allow for missmatches (NO GAPS, 
    # since I don't want frame shifts). re.search("(pattern){e<=2}", seq) is not working.
    # This fx returns the number of mismatches between the two sequences.
    return np.sum(i for i in map(lambda x, y: 0 if x == y else 1, a, b))

# return the position of the best match allowing up to 3 mismatches.
def search_distance(primer, fastq_sequence, annealing_spots):
    best_match, position = 3, 0
    for p in annealing_spots:
        match = seq_distance(primer, fastq_sequence[p:p+15])
        if match <= best_match:
            position, best_match = p, match
    return position

def sequence_check(seq, cutOff=160):
    ''' Check that start and stop are IN and there is no frame shift'''
    i90 = re.search(start_primer,seq)
    # looking for the final90 (3' primer), I don't have to slide all over the sequence since
    # I know that ot should be at least 51 nt from the start and it should be mutiple of 3
    # old version was [i.start() for i in re.finditer(end_primer, seq)] #Could be more than 1 
    annealing_1st_nt = np.arange(51,len(seq)-15,3) #Start @nt 51 (shorter fragments are non-informative, every 3 (no frameshift) until len(seq)-len(end_primer)  
    f90 = search_distance(end_primer, seq, annealing_1st_nt)
    # Discard no Start/Stop or framechift
    if not i90: return ["noStart"]
    if not f90: return ["noStop"]
    if (f90-i90.end())%3 >0: return ["frameShift"]
    # If everything works fine, the function returns a UPPER CASE char-LIST. 
    if len(seq)<cutOff: return ['shortSequence']
    return [i90.end(),f90]
    
def quality_check(gene_seq, threshold = 30):
    ''' Sanger format can encode a Phred quality score from 0 to 93 using ASCII 33 to 126
    from https://en.wikipedia.org/wiki/FASTQ_format '''
    threshold = threshold * 3 # To avoid having to divide by 3 (hence forcing to use floating numbers...etc.)    
    geneLen = len(gene_seq)
    geneLen = geneLen - geneLen%3
    # nucleotides (nt) and amino-acids(aa) scores of sequences
    nt_Phred_qual = [ord(n)-33 for n in gene_seq]
    aa_Phred_qual = np.array( [np.sum(nt_Phred_qual[i:i+3]) for i in range(0,geneLen,3)] )
    # Return False if there is an aa with score lower than the threshold
    if np.sum(aa_Phred_qual <= threshold):
        return False
    else:
        return np.mean(aa_Phred_qual)

if __name__== '__main__': main()
