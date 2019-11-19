from deepexplain.tensorflow import DeepExplain
from matplotlib import transforms
import matplotlib.patheffects
from matplotlib.font_manager import FontProperties



class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx, sy=None):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy)+affine
        renderer.draw_path(gc, tpath, affine, rgbFace)


def ohe_2_aa_analog(ohe_data):
    global ss
    seq = ohe_data[:,:20].reshape(30,20)
    SS =  ohe_data[:,20:].reshape(30,3)

    seq_list = []
    for i in seq:
        seq_list.append([(i,j) for i,j in zip(aa,i)])
        
    ss_list = []
    for i in SS:
        ss_list.append([(i,j) for i,j in zip(ss,i)])
        
    return seq_list, ss_list


# define color schemes
aa_1 = {i:'blue' for i in ['R','H','K']}
aa_2 = {i:'red' for i in ['D','E']}
aa_3 = {i:'cyan' for i in ['S','T','N','Q']}
aa_4 = {i:'green' for i in ['A','I','L','M','F','W','Y','V']}
aa_5 = {i:'yellow' for i in ['C','G','P']}

COLOR_SCHEME_AA = {**aa_1,**aa_2,**aa_3,**aa_4,**aa_5}

COLOR_SCHEME_SS = {'E': 'orange',
                   'H': 'blue',
                   '-': 'red'}

def draw_logo2(all_scores, filename, fontfamily='Arial', size=80, COLOR_SCHEME=COLOR_SCHEME_AA):
    if fontfamily == 'xkcd':
        plt.xkcd()
    else:
        matplotlib.rcParams['font.family'] = fontfamily

    fig, ax = plt.subplots(figsize=(len(all_scores), 2.5))

    font = FontProperties()
    font.set_size(size)
    font.set_weight('bold')
    
    #font.set_family(fontfamily)

    ax.set_xticks(range(1,len(all_scores)+1))    
    ax.set_yticks(range(0,6))
    ax.set_xticklabels(range(1,len(all_scores)+1), rotation=90)
    ax.set_yticklabels(np.arange(-3,3,1))    
    sns.despine(ax=ax, trim=True)
    
    trans_offset = transforms.offset_copy(ax.transData,fig=fig, x=1, y=0,units='dots')
   

    for index, scores in enumerate(all_scores):
        yshift = 0
        for base, score in scores:
            txt = ax.text(index+1, 3, base,transform=trans_offset,fontsize=80, color=COLOR_SCHEME[base],ha='center',fontproperties=font,)
            txt.set_path_effects([Scale(1.0, score)])
            fig.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            yshift = window_ext.height*score
            trans_offset = transforms.offset_copy(txt._transform, fig=fig, y=yshift, units='points')
        
        trans_offset = transforms.offset_copy(ax.transData, fig=fig, x=1, y=0, units='points')    

    plt.axis('off')
    plt.savefig(filename)
    return fig


def make_figure(name, name_seq, name_ss, min_range, max_range):

    ohe_name = prepare_ohe(np.vstack([[i for i in name_seq[min_range:max_range]], 
                                      [i for i in name_ss[min_range:max_range]]]))

    with DeepExplain(session=K.get_session()) as de:
        input_tensor = ADPred.layers[0].input
        fModel = Model(inputs=input_tensor, outputs = ADPred.layers[-2].output)
        target_tensor = fModel(input_tensor)

        xs = ohe_name.reshape(1,30,23,1)
        ys = np.array([1]).reshape(1,1)

        attributions_gi = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        #attributions_sv = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys)
        #attributions_dl = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        #attributions_s  = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)


    #for name, j in zip(['_grad_int_','_shapley_vals_','_saliency_'],[attributions_gi, attributions_sv,attributions_s]):
    for i in range(len(attributions_gi)):
        ALL_SCORES1, aSS1 = ohe_2_aa_analog(attributions_gi[i])
        fig = draw_logo2(ALL_SCORES1, name+'_AA.png', 'Verdana', COLOR_SCHEME=COLOR_SCHEME_AA)
        #draw_logo2(aSS1, name+'gcn4_SS.png', 'Verdana', COLOR_SCHEME=COLOR_SCHEME_SS)
    return fig



# draw Gcn4        
name='gcn4'
gcn4_seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
gcn4_ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')
min_range = 107
max_range = 137


make_figure('gcn4', gcn4_seq, gcn4_ss, min_range, max_range)


# draw Ino2 nAD
name='ino2'
ino2_seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
ino2_ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')
min_range = 11
max_range = 41
make_figure('Ino2-nAD', ino2_seq, ino2_ss, min_range, max_range)


# draw Ino2 cAD
ino2_seq = read_fasta(analysis_home + '/data/'+ name + '.fasta')
ino2_ss = read_horiz(analysis_home + '/data/'+ name + '.horiz')
min_range = 114
max_range = 144
make_figure('Ino2-cAD', ino2_seq, ino2_ss, min_range, max_range)

# draw Gal4
name='gal4'
gal4_seq = read_fasta('data/'+ name + '.fasta')
gal4_ss = read_horiz('data/'+ name + '.horiz')
min_range = 847
max_range = 877
make_figure('gal4', gal4_seq, gal4_ss, min_range, max_range)
