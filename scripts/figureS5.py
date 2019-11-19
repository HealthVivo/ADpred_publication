from matplotlib import transforms
import matplotlib.patheffects
from matplotlib.font_manager import FontProperties
from deepexplain.tensorflow import DeepExplain

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
    plt.show()
    plt.close()


df = get_enrichment_scores_table()
df['score'] = np.sum(df.iloc[:,1:]*ARG3_FACS_RFU, axis=1) / (df.iloc[:,0] + 1)

best_20_enrichment = list(df.sort_values('score')[::-1][:20].index)
best_20_ADscore = positives[np.argsort([ADPred.predict(prepare_ohe(sample).reshape(1,30,23,1))[0][0] for sample in positives])[-20:],0]

ids_enrichment = [np.where(positives[:,0]==sample)[0][0] for sample in best_20_enrichment]
ids_adpred = [np.where(positives[:,0]==sample)[0][0] for sample in best_20_ADscore]

ohes1 = np.array([prepare_ohe(i) for i in positives[ids_enrichment]])
ohes2 = np.array([prepare_ohe(i) for i in positives[ids_adpred]])

preds1 = np.hstack([ADPred.predict(i.reshape(1,30,23,1))[0][0] for i in ohes1])
preds2 = np.hstack([ADPred.predict(i.reshape(1,30,23,1))[0][0] for i in ohes2])


with DeepExplain(session=K.get_session()) as de:
    input_tensor = ADPred.layers[0].input
    fModel = Model(inputs=input_tensor, outputs = ADPred.layers[-2].output)
    target_tensor = fModel(input_tensor)
    
    s0 = ohes2.shape[0]
    xs = ohes2.reshape(s0,30,23,1)
    ys = preds2.reshape(s0,1)
    
    attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
    #attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)
    #attributions_dl     = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
    
#for i,j in zip(attributions_dl, preds2):
#    f,ax1 = plt.subplots(1, figsize=(20,5))
#    logo(i, j, ax1)
    
for i in range(len(attributions_gradin)):
    ALL_SCORES1, aSS1 = ohe_2_aa_analog(attributions_gradin[i])
    draw_logo2(ALL_SCORES1, 'lala_AA.png', 'Verdana', COLOR_SCHEME=COLOR_SCHEME_AA)
