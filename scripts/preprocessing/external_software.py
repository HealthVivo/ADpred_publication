# Run external software!!

import os, sys, subprocess, re
import pandas as pd
import numpy as np
from time import time

if len(sys.argv)<2:
    print('*'*60)
    print('python Step_5.py <long | short>\n')
    print('*'*60)
    sys.exit(1)

SHORT_or_LONG = sys.argv[1]

def main():
    # notice for the user
    print('='*80,'\nrunning IUpred now!')
    print('='*80)
    # decided to do it sequential instead of parallel so I don't have to move files to diff folders and there is no overlapping of files named the same that could be used by two parallel runs 
    #for param in ['long','short','glob']:
    param = SHORT_or_LONG
    # get parameters
    output_file = '/fh/scratch/delete90/hahn_s/aerijman/from_scratch/iupred_' + param + '.csv' 
    # ecexute !
    execute_all(param).to_csv(output_file)

def execute(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (result, error) = process.communicate()
    rc = process.wait()
    if rc!=0:
        print('error: failed execute command:', cmd)
        print(error)
    return(result)


def execute_all(SHORT_or_LONG):
    # makedir to store the different results
    os.mkdir(SHORT_or_LONG)
    iupred_results = []
    ids = []
    # list of fasta files to merge and merge them to get unique sequences from all experiments. We don't need them splited by experiments here!
    list_of_fastas = [i for i in os.listdir() if re.search('.fasta',i)]
    sequences = []
    for f in list_of_fastas:
        for n,i in enumerate(open(f)):
            if n%2!=0: sequences.append(i.strip())
    sequences = list(set(sequences))

    #######################################################################################
    # First, let's let aside sequences with less than 30 aa
    len_with_less = len(sequences)
    sequences = [i for i in sequences if len(i)==30]
    print('{} sequences left aside based on their short length (<30)'.format(len_with_less-len(sequences)))
    # In a later version I could see how I incorporate these sequences into the calculations
    ########################################################################################
    
    # change directory, otherwise iupred doesn't work.
    os.chdir('/app/iupred/')
    # execute line by line from the fasta 
    for seq in sequences:
        # 1- create the single fasta with identifier to track possible runtime_errors
        fasta_filename = '/fh/scratch/delete90/hahn_s/aerijman/from_scratch/'+SHORT_or_LONG+'/f' + seq + '.fa'
        fasta_sequence = '">fasta_seq\n"'+seq
        execute('echo ' + fasta_sequence + '> ' + fasta_filename)
        # 2- run the iupred long and store the results in the long matrix (to make things faster the 'long or short' will be an argument, so I can run them in parallel easily.
        try:
            tmp = execute('./iupred '+ fasta_filename + ' ' + SHORT_or_LONG)
            tmp = tmp.decode("utf-8").strip().split('\n')
            tmp = np.array([i.strip().split(' ')[6] for i in tmp[9:]], dtype=float)
        except:
            print('sequence ' + fasta_sequence + 'had an Error\n')
            execute('rm ' + fasta_filename)
            continue
        # 3- Append ids and iupred results into arrays that will be merged into df or matrix later to merge with other results based on sequences as indexes
        if len(seq)==30:                         # Already happened that it was not 30 and the dataframe creation failed and I lost all info.
            ids.append(seq)
            iupred_results.append(tmp) 
        # 4- remove the fasta file and free memory for the next file
        execute('rm ' + fasta_filename)
    # 5- tidy results into df
    return(pd.DataFrame(np.vstack(iupred_results), index=ids))


if __name__=='__main__':
    main()

