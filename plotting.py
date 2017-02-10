import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from players import *
from mpl_toolkits.mplot3d import Axes3D

execfile('load.py')

pitches = atbats.merge(pitches, on=['game_path', 'ab_num'])
pitches.loc[pitches.des == 'inplay','des'] = pitches.loc[pitches.des == 'inplay', 'event']

kinsler_ab = atbats[atbats.batter_id == 435079]
kinsler = pitches[pitches.batter_id == 435079]
porcello_ab = atbats[atbats.pitcher_id == 519144]
porcello = pitches[pitches.pitcher_id == 519144]

## Pitcher Pitch Distribution

print "Porcello Pitch Distribution: \n", porcello.type.value_counts()

xbound = np.percentile(pitches.x, [1,99])
ybound = np.percentile(pitches.y, [1,99])


plocs,_,_ = np.histogram2d(porcello.x.values, porcello.y.values, range=[xbound, ybound]) 

plt.imshow(plocs)
plt.show()

plt.pcolormesh(plocs)
plt.show()

## Batter Heatmap

strike = kinsler[kinsler.des == 'strike']
ball = kinsler[kinsler.des == 'ball']
foul = kinsler[kinsler.des == 'foul']
hit = kinsler[kinsler.des == 'hit']
out = kinsler[kinsler.des == 'out']

slocs,_,_ = np.histogram2d(strike.x.values, strike.y.values, range=[xbound, ybound])
blocs,_,_ = np.histogram2d(ball.x.values, ball.y.values, range=[xbound, ybound])
flocs,_,_ = np.histogram2d(foul.x.values, foul.y.values, range=[xbound, ybound])
hlocs,_,_ = np.histogram2d(hit.x.values, hit.y.values, range=[xbound, ybound])
olocs,_,_ = np.histogram2d(out.x.values, out.y.values, range=[xbound, ybound])

total = slocs + blocs + flocs + hlocs + olocs
slocs /= total; slocs[total == 0] = 0
blocs /= total; blocs[total == 0] = 1
flocs /= total; flocs[total == 0] = 0
hlocs /= total; hlocs[total == 0] = 0
olocs /= total; olocs[total == 0] = 0

plt.imshow(hlocs)
plt.show()

## Kernel Density Estimation Stuff

xy = [porcello.x.values, porcello.y.values]
x,y = zip(*[(x,y) for x in np.arange(xbound[0], xbound[1],5) for y in np.arange(ybound[0], ybound[1], 5)])
locs = np.vstack([x,y])

for bw in [0.07,0.10,0.15,0.25]:
	kde = stats.gaussian_kde(xy, bw_method=bw)
	z = kde(locs)
	
	fig = plt.figure(figsize=[5,4])
	ax = fig.add_subplot(1,1,1, projection='3d')
	ax.plot_trisurf(x,y,z,cmap=cm.Spectral) # PDF of Rick Porcello's pitch locations
	plt.xlabel('x'); plt.ylabel('y');
	plt.suptitle('Pitch Location Distribution: Bandwidth=' + str(bw))
	plt.savefig('figures/pitch-' + str(bw) + '.png')

for ptype, data in porcello.groupby('type'):
	if(len(data) <= 10): continue
	xy = [data.x.values, data.y.values]
	kde = stats.gaussian_kde(xy)
	z = kde(locs)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1, projection='3d')
	ax.plot_trisurf(x,y,z,cmap=cm.Spectral) # PDF of Rick Porcello's pitch locations
	plt.xlabel('x'); plt.ylabel('y');
	plt.suptitle('Pitch Location Distribution: Pitch Type = ' + ptype)
	plt.savefig('figures/' + ptype + '.pdf')



## Conditional KDE

# P(Hit | x,y)= f(x,y | Hit) P(Hit) / f(x,y)

# kde_all = stats.gaussian_kde([kinsler.x.values, kinsler.y.values])
kdes = {}
for des, data in kinsler.groupby('des'):
	kde = stats.gaussian_kde([data.x.values, data.y.values])
	kdes[des] = kde(locs) * len(data)
	
for des in kdes.keys():
	kde = kdes[des] / sum(kdes.values())
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1,projection='3d')
	ax.plot_trisurf(x,y,kde,cmap=cm.Spectral)
	plt.xlabel('x'); plt.ylabel('y')
	plt.suptitle(des)
	plt.savefig('figures/' + des + '.pdf')

