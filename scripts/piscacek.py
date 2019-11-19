most = '[MDENQSTYG]{KRHCGP}[ILVFWM]{KRHCGP}{CGP}{KRHCGP}[ILVFWM][ILVFWMAY]{KRHC}' 
less = '[MDENQSTYCPGA]X[ILVFWMAY]{KRHCGP}{CGP}{CGP}[ILVFWMAY]XX'

print('Most stringent {} and \nLess stringent {}'.format(most, less))

def generate_motif(pattern):
    seq, flag = [], False
    
    for i in pattern:
        if i == "X":
            seq.append(''.join(["["]+aa+["]"]))

        elif i =="{":
           flag=True
           tmp=[]            

        elif i == "}":
            flag=False
            tmp = [i for i in aa if i not in tmp]
            seq.append(''.join(["["]+tmp+["]"]))

        else:
            if flag:
                tmp.append(i)
            else:
                seq.append(i)

    return ''.join(seq)


def search_9aa(sequences):
    if isinstance(sequences, str):
        return('please provide a list or numpy array, not a string')

    n_most,n_less=0,0

    motif_most = generate_motif(most)
    motif_less = generate_motif(less)

    for n, seq in enumerate(sequences):
        if re.search(motif_most, seq):
            n_most+=1
        if re.search(motif_less, seq):
            n_less+=1
    
    return {'most_stringent':n_most, 'less_stringent':n_less, 'total':n}
