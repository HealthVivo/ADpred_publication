#!/bin/python

from initialize_notebook import *


matplotlib.style.use('default')

flip_dip = [i.strip().split(',') for i in open(analysis_home+'/results/dimer.flip.results.csv', 'r')]
flip_dip_header = [i.replace(' ','') for i in flip_dip[0]]
vals = flip_dip[1:]

print(flip_dip_header)

combs = [i[:2] for i in vals if len(i)>1]
vals = np.vstack([i[2:] for i in vals if len(i[2:])>0]).astype(np.float16)

def getit(dip, measure='test_auc'):
    global flip_dip_header
    tmp = np.array([n for n,i in enumerate(combs) if dip in i])
    return vals[tmp,flip_dip_header.index(measure)-2]
    
wd = getit('WD')
dw = getit('DW')
we = getit('WE')
ew = getit('EW')
ve = getit('VE')
ev = getit('EV')
vd = getit('VD')
dv = getit('DV')
ld = getit('LD')
dl = getit('DL')
fd = getit('FD')
df = getit('DF')
yd = getit('YD')
dy = getit('DY')

lt = getit('LT')
tl = getit('TL')
ww = getit('WW')
dd = getit('DD')

All = vals[:,0]

#f,ax = plt.subplots(1, figsize=(7,5))
#plt.fill_between(x=(-2,20),y1=(0.941,0.941),y2=(0.945,0.945), alpha=0.3, color='grey')
#plt.plot((-2,20),(0.941,0.941), ls='--', lw=2, alpha=0.6, c='k')
#sns.set(style="ticks", palette="pastel")
#sns.boxplot(data=[All,wd,dw,we,ew,ve,ev,vd,dv,ld,dl,fd,df,yd,dy,lt,tl,ww,dd], ax=ax)
#ax.set_xticklabels(['All','WD','DW','WE','EW','VE','EV','VD','DV','LD','DL','FD','DF','YD','DY','LT','TL','WW','DD'])
#sns.despine(offset=10, trim=True)


#labels = [[i.upper()]*eval(i).shape[0] for i in ['All','dw','dw','ew','ew','ev','ev','dv','dv','dl','dl','df','df','dy','dy','tl','tl','ww','dd']]
#values = [eval(i) for i in ['All','wd','dw','we','ew','ve','ev','vd','dv','ld','dl','fd','df','yd','dy','lt','tl','ww','dd']]
#labels = np.hstack(labels)
#values = np.hstack(values)

#hues = ['fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','fwd','fwd']
#hues = [[i] * eval(j).shape[0] for i,j in zip(hues, ['All','wd','dw','we','ew','ve','ev','vd','dv','ld','dl','fd','df','yd','dy','lt','tl','ww','dd'])]


labels = [[i.upper()]*eval(i).shape[0] for i in ['dw','dw','ew','ew','ev','ev','dv','dv','dl','dl','df','df','dy','dy','tl','tl','ww','dd']]
values = [eval(i) for i in ['wd','dw','we','ew','ve','ev','vd','dv','ld','dl','fd','df','yd','dy','lt','tl','ww','dd']]
labels = np.hstack(labels)
values = np.hstack(values)

hues = ['rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','rev','fwd','fwd','fwd']
hues = [[i] * eval(j).shape[0] for i,j in zip(hues, ['wd','dw','we','ew','ve','ev','vd','dv','ld','dl','fd','df','yd','dy','lt','tl','ww','dd'])]
hues = np.hstack(hues)

df = pd.DataFrame(data=np.vstack([values, labels, hues])).T
df.columns = ['AUROC','dipeptides','direction']
df.AUROC = df.AUROC.astype(float)


matplotlib.style.use('seaborn-white')
f,ax = plt.subplots(1,figsize=(5,5))
sns.boxplot(data=df, x='dipeptides',y='AUROC',hue='direction', palette="Set3", fliersize=1.5)
sns.set(style="ticks", palette="pastel")
sns.despine(offset=10, trim=True)
ax.fill_between(x=(-2,20),y1=(0.941,0.941),y2=(0.945,0.945), alpha=0.3, color='grey')
ax.tick_params(axis='both', labelsize=11)
ax.set_ylabel("AUROC", fontsize=11)

ax.set_xticks(np.arange(0.5,10.5), minor=True)
plt.grid(axis='x', which='minor', ls='--', lw=1, color='grey')
