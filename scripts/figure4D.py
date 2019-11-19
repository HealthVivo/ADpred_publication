#!/bin/python

#plt.style.available
#plt.style.use('tableau-colorblind10')


matplotlib.style.use('tableau-colorblind10')

#f, ax = plt.subplots(1, figsize=(7,7))
f, (ax,ax2) = plt.subplots(2,1, figsize=(7,8), sharex=True, gridspec_kw={'height_ratios':[3,1], 'hspace':0.05})
Qs = charge(deep_samples[:,0]) / 30
cols = ['b' if i==0 else 'r' for i in deep_y[deep_idx]]
ax.scatter(y_hat_deep[deep_idx], Qs[deep_idx], color=cols, s=3)


ax.scatter([0.5,0.5],[0.25,0.31], s=40, c=['b','r'])
ax.text(0.55,0.288, "AD positive", fontsize=24)
ax.text(0.55,0.228, "AD negative", fontsize=24)

ax.tick_params(axis='both', which='major', labelsize=18)
plt.rc('axes', facecolor='#E6E6E6')
ax.grid(color='w', linestyle='solid')
ax2.grid(color='w', linestyle='solid')

xn = np.array([n for n,i in enumerate(cols) if i=='r'])
xp = np.array([n for n,i in enumerate(cols) if i=='b'])
ax2.hist(y_hat_deep[deep_idx[xn]], density=True, bins=100, color='r', alpha=0.5);
ax2.hist(y_hat_deep[deep_idx[xp]], density=True, bins=100, color='b', alpha=0.5);


ax2.set_xticks(np.linspace(0,1,6))
ax2.tick_params(axis='x', which='major', labelsize=18)
#ax.tick_params(top='off', bottom='off', left='on', right='off', labelleft='off', labelbottom='off')
ax.tick_params(bottom='off')
ax2.tick_params(labelleft='off', left='off')


ax.plot([-0.05,1.05],[0,0], ls='--', lw=3, c='k')

plt.xlim(-0.02,1.02)

plt.tight_layout()
