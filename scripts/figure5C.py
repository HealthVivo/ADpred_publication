#matplotlib.rcParams.update({'font.size': 22})

known_regions={  # information taken from Pacheco et al; PMID:29507182
    'gal4':[np.arange(840,881)],                        # order [7:106, 244:506]
    'gcn4':[np.arange(1,98), np.arange(102,134)],       # order [223:280]
    'ino2':[np.arange(1,41),np.arange(96,160)],         # order [6:41, 230:301]
    'met4':[np.arange(72,117), np.arange(126,161)],     # order [1:399]
    #'pdr1':[np.arange(901,1068)],
    'rtg3':[np.arange(1,250), np.arange(375,486)],      # order [2:266, 286:365]    
    'hap4':[np.arange(321,490)],                        # order [65:265]
    'mtf1_fly':[np.arange(500,1000)/3], #Information for this is in nucleotides => /3  # order [1:76]
    'rap1':[np.arange(595,720)]                         # order [95:213, 360:439, 446:594]
}

ordered_regions={
    #'gal4':[np.arange(7,106), np.arange(244,506)],
    'gal4':[np.arange(1,70), np.arange(233,653)], # https://toolkit.tuebingen.mpg.de/jobs/7814146_1
    #'gcn4':[np.arange(223,280)], 
    'gcn4':[np.arange(213,280)], 
    #'ino2':[np.arange(6,41),np.arange(230,301)], 
    'ino2':[np.arange(227,304)], 
    #'met4':[np.arange(1,399)],
    'met4':[np.arange(608,648)],
    #'rtg3':[np.arange(2,266), np.arange(286,365)],
    'rtg3':[np.arange(253,372)],
    #'hap4':[np.arange(65,265)],
    'hap4':[np.arange(65,80)],
    #'mtf1_fly':[np.arange(1,76)],
    'mtf1_fly':[np.arange(1,62)],
    #'rap1':[np.arange(95,213), np.arange(360,439), np.arange(446,594)]
    'rap1':[np.arange(110,207), np.arange(355,560), np.arange(748,827)]
}

def test_model(model, known_regions):
    plt.figure(figsize=(14,14))
    for n, name in enumerate(known_regions.keys()):

        # read files into lists
        seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
        ss = read_horiz(analysis_home  + '/data/'+ name + '.horiz')

        if name in ['pdr1', 'rtg3', 'hap4']:
            ss = ss[60:-60]

        # one hot encode the data
        vars()[name] = prepare_ohe([seq,ss])

        # plot predictions
        plt.subplot(3,3,n+1)
        font = {'family': 'calibry', 'color':  'black', 'weight': 'normal', 'size': 30}
        title = ['Mtf1 (fly)' if name=='mtf_fly' else name][0] 
        plt.title(name, fontdict=font, y=1.05)  
        #plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)        
        plt.xticks(np.linspace(0,len(seq),4).astype(int), fontsize=20)
        plt.yticks(np.round(np.linspace(0,1,4),2), fontsize=20)

        # predictions with the new model
        tmp = np.convolve(predict_TAD(eval(name), model), np.ones(15)/15, 'same')
        tmp = np.array([i if i>0.5 else 0 for i in tmp])
        plt.fill_between(np.arange(len(tmp)), tmp, alpha=0.3, color='b')

        # plot published tADs
        for i in known_regions[name]:
            plt.fill_between(i,np.zeros(len(i))+0.7, np.ones(len(i)), color='y', alpha=0.3)

        # plot ordered regions according to s2p2.pro
        for i in ordered_regions[name]:
            plt.fill_between(i,np.zeros(len(i)), np.ones(len(i))-0.7, color='gray', alpha=0.3)

        # plot legend 
        #if n==2:
        #    plt.legend(['predictions', 'documented ADs'], bbox_to_anchor=(5, 0.8), fontsize=25)

        # show experimentally validated regions
        if name=='gal4':
            plt.scatter(180, 0.95, marker='v', s=200, c='r')
            plt.ylabel('score')
        if name=='hap4':
            [plt.scatter(i, j, marker='v', s=200, c='r') for [i,j] in zip([150,310,537],[0.98,0.98,0.98])]
        if name=='mtf1_fly':
            plt.scatter(270, 0.95, marker='v', s=200, c='r')
            plt.xlabel('residue number')
        
            
test_model(ADPred, known_regions)
plt.tight_layout()

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.savefig('figure5C.png', dpi=300)
