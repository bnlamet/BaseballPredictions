import pandas as pd
import numpy as np
from random import random
from math import log
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import pdb
import numpy.linalg as lg
from multiprocessing import Pool, cpu_count
from functools import partial


class Player:
	def __init__(self):
		self.atbats = None # at bats data frame
		self.pitches = None # pitches data frame
		self.stats = None # at bat outcome proportions
		self.pstats = pd.Series({'hit':0,'out':0,'walk':0,'ball':0,'strike':0,'foul':0})	
		self.pdfs = None # Pitcher PDF 
		self.cpdfs = None # Conditional PDFs
		self.hists = {} # histograms/2d arrays for pitch locations
		self.mlearn = {}
		self.ptable = None # data frame grouped by (balls, strikes), des
		self.counts = {}
		self.absplits = None # data frame of atbat probabilities where index is (day/night, home/away, L/R, L/R) 

	def setABs(self, atbats):
		self.atbats = atbats
		self.stats = atbats.event.value_counts(normalize=True)
		bayes = MultinomialNB(alpha=2.5, fit_prior=True, class_prior=None)
		forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=25)
		logistic = LogisticRegression()
		cols = ['home','b_stand2','p_throws2']
		bayes.fit(atbats[cols], atbats.event2)
		forest.fit(atbats[cols], atbats.event2)
		logistic.fit(atbats[cols], atbats.event2)
		self.mlearn['ab_bayes'] = bayes 
		self.mlearn['ab_forest'] = forest 
		self.mlearn['ab_logistic'] = logistic 
#		pdb.set_trace()		
		G = atbats.groupby(cols).apply(lambda d: d.event.value_counts())
		if(isinstance(G, pd.Series)): G = G.unstack().fillna(0)
		self.absplits = weight(G)
		
	
	def ml_predict(self, algo, data):
		cols = ['home','b_stand2','p_throws2']
		ans = self.mlearn[algo].predict_proba([data[cols]])
		return pd.Series(ans[0,:], index=['out','hit','walk'])

	
	def setPitches(self, pitches):
		self.pitches = pitches
		self.pstats += pitches.des.value_counts(normalize=True)
		self.pstats.replace(np.nan, 0, inplace=True)
		
		# For use in markov3
		self.counts['normal'] = weight(pitches.groupby(['balls','strikes']).apply(lambda d: d.des.value_counts()).unstack().fillna(0))
		self.counts['type'] = pitches.groupby(['balls','strikes']).apply(lambda d: d.type.value_counts(normalize=True))
		self.counts['type-des'] = weight(pitches.groupby(['balls','strikes','type']).apply(lambda d: d.des.value_counts()).unstack().fillna(0))

		X,Y = np.meshgrid(np.linspace(36,177,num=145), np.linspace(99,216,num=118))
		xy = np.vstack([X.ravel(), Y.ravel()])

		# For use in markov4
		'''
		try:
			print 'Building KDE Mesh'
			self.pdf = stats.gaussian_kde(np.vstack([pitches.x, pitches.y])).evaluate(xy)
			stub = pd.Series(index=['ball','strike','foul','hit','out','walk']).fillna(0)
			self.cpdfs = (stub + pitches.groupby('des').apply(lambda data: 0 if len(data) < 5 else stats.gaussian_kde(np.vstack([data.x, data.y])).evaluate(xy) * len(data))).fillna(0)
			
			total = self.cpdfs.sum()
			self.cpdfs = self.cpdfs.map(lambda kde: kde/total)
		except:
			pdb.set_trace()	
		'''

# Evaluate predictions using quadratic loss scoring metric
# Equivalent to mean squared error 
def quadloss(pred, real):
	f = lambda h: (real.map(lambda x: 1 if x==h else 0) - pred[h])**2
	return sum(f('hit') + f('out') + f('walk')) / len(pred)

def brier(pred, real):
	out = pd.Series({'out':1,'hit':0,'walk':0})
	hit = pd.Series({'out':0,'hit':1,'walk':0})
	walk = pd.Series({'out':0,'hit':0,'walk':1})
	A = real.apply(lambda c: out if c=='out' else hit if c=='hit' else walk)
	E = pred

	E['hit2'] = E.hit.map(lambda x: int(x*100))
	E['out2'] = E.out.map(lambda x: int(x*100))

	N = len(E)
	o = A.sum() / N
	REL = 0
	RES = 0 
	for _, data in E.groupby(['hit2','out2']):
		data = data[['hit','out','walk']]
		nk = len(data)
		fk = data.sum() / nk
		ok = A.loc[data.index].sum() / nk
		REL += nk * (fk - ok)**2
		RES += nk * (ok - o)**2
	REL = REL.div(N).sum()
	RES = RES.div(N).sum()
	UNC = o.mul(1 - o).sum()
	return REL, RES, UNC

def logloss(pred, real):
	events = real
	log2 = lambda p: -float('inf') if p == 0 else log(p)
	hit = pred[events == 'hit']['hit'].map(log2)
	walk = pred[events == 'walk']['walk'].map(log2)
	out = pred[events == 'out']['out'].map(log2)
	return -(sum(hit) + sum(walk) + sum(out)) / len(pred)

# Probabilities are completely random
def simple0(ab_test):
	out, walk, hit = random(), random(), random()
	total = out+walk+hit
	out /= total; walk /= total; hit /= total
	return pd.Series({ 'hit' : hit, 'walk' : walk, 'out' : out })

# Probabilities are always 1 for 'out' (most common occurance)
def simple1(ab_test):
	return pd.Series({ 'hit' : 0.0, 'walk' : 0.0, 'out' : 1.0 })

# Probabilities only dependent on batter stats
def simple2(ab_test):
	batter = batters[ab_test.batter_id]
	return batter.stats

# Probabilities only dependent on pitcher stats
def simple3(ab_test):
	pitcher = pitchers[ab_test.pitcher_id]
	return pitcher.stats

# Probabilities are calculated with the log5 formula
def simple4(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	evts = batter.stats * pitcher.stats / average.stats
	return evts / sum(evts)

def weight1(ab_test):
	cols = ['home','b_stand2', 'p_throws2']
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	X = tuple(ab_test[cols])
#	pdb.set_trace()
	b = batter.absplits.loc[X]
	p = pitcher.absplits.loc[X]
	a = average.absplits.loc[X]
	A = b*p/a
	return A/A.sum()	

# Use Machine Learning to predict At Bat outcomes (ignoring pitch data)
def mlearn(method, ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	#pdb.set_trace()
	bprob = batter.ml_predict(method, ab_test)
	pprob = pitcher.ml_predict(method, ab_test)
	aprob = average.ml_predict(method, ab_test)
	ans = bprob*pprob / aprob
	return ans / ans.sum()

def bayes(ab_test):
	return mlearn('ab_bayes', ab_test)

# Use Machine Learning to predict At Bat outcomes (ignoring pitch data)
def forest(ab_test):
	return mlearn('ab_forest', ab_test)	

def logistic(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	X = ['out','hit','walk']
	ans = logit.predict_proba([np.append(batter.stats[X],pitcher.stats[X])])
	return pd.Series(ans[0,:], index=[X])

def markov(P):
	T = np.zeros([15,15])
	T[12,12] = T[13,13] = T[14,14] = 1.0
	s,b,f,h,o,w = 'strike','ball','foul','hit','out','walk'
	# Build Markov Chain
	T[0,1], T[0,3], T[0,12], T[0,13], T[0,14] = P[0,0][s] + P[0,0][f], P[0,0][b], P[0,0][h], P[0,0][w], P[0,0][o]
	T[1,2], T[1,4], T[1,12], T[1,13], T[1,14] = P[0,1][s] + P[0,1][f], P[0,1][b], P[0,1][h], P[0,1][w], P[0,1][o]
	T[2,2], T[2,5], T[2,12], T[2,13], T[2,14] = P[0,2][f], P[0,2][b], P[0,2][h], P[0,2][w], P[0,2][o] + P[0,2][s]
	T[3,4], T[3,6], T[3,12], T[3,13], T[3,14] = P[1,0][s] + P[1,0][f], P[1,0][b], P[1,0][h], P[1,0][w], P[1,0][o]
	T[4,5], T[4,7], T[4,12], T[4,13], T[4,14] = P[1,1][s] + P[1,1][f], P[1,1][b], P[1,1][h], P[1,1][w], P[1,1][o]
	T[5,5], T[5,8], T[5,12], T[5,13], T[5,14] = P[1,2][f], P[1,2][b], P[1,2][h], P[1,2][w], P[1,2][o] + P[1,2][s]
	T[6,7], T[6,9], T[6,12], T[6,13], T[6,14] = P[2,0][s] + P[2,0][f], P[2,0][b], P[2,0][h], P[2,0][w], P[2,0][o]
	T[7,8], T[7,10], T[7,12], T[7,13], T[7,14] = P[2,1][s] + P[2,1][f], P[2,1][b], P[2,1][h], P[2,1][w], P[2,1][o]
	T[8,8], T[8,11], T[8,12], T[8,13], T[8,14] = P[2,2][f], P[2,2][b], P[2,2][h], P[2,2][w], P[2,2][o] + P[2,2][s]
	T[9,10], T[9,12], T[9,13], T[9,14] = P[3,0][s] + P[3,0][f], P[3,0][h], P[3,0][b] + P[3,0][w], P[3,0][o]
	T[10,11], T[10,12], T[10,13], T[10,14] = P[3,1][s] + P[3,1][f], P[3,1][h], P[3,1][b] + P[3,1][w], P[3,1][o]
	T[11,11], T[11,12], T[11,13], T[11,14] = P[3,2][f], P[3,2][h], P[3,2][b] + P[3,2][w], P[3,2][o] + P[3,2][s]

	# Compute the absorbing state probabilities
	T = lg.matrix_power(T,64)
	return pd.Series({ 'out' : T[0,14], 'hit' : T[0,12], 'walk' : T[0,13] })

# Transition probability does not depend on current count
def markov1(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	outcomes = pd.Series(index=['strike','ball','foul','hit','out','walk'])
	for pt in ['strike','ball','foul','hit','out','walk']:
		outcomes[pt] = batter.pstats[pt]*pitcher.pstats[pt] / average.pstats[pt] # log5 formula
	outcomes /= outcomes.sum()
	
	P = np.array([[outcomes.to_dict() for s in [0,1,2]] for b in [0,1,2,3]])	
	return markov(P)

# Transition probability DOES depend on the current count, and is approximated using
# batter/pitcher frequencies using the log5 formula
def markov2(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	
	A = average.counts['normal']
	B = batter.counts['normal']
	P = pitcher.counts['normal']

	Q = B * P / A # log5 formula
	Q.fillna(0, inplace=True)
	tot = Q.sum(axis=1)
	Q = Q.apply(lambda c: c/tot)
	Q = np.array([[Q.loc[b,s].to_dict() for s in [0,1,2]] for b in [0,1,2,3]])
	ans = markov(Q)
#	pdb.set_trace()
	if(np.isnan(ans).any()):
		pdb.set_trace()
	return ans

# Takes into account pitch type
def markov3(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	
	PP = pitcher.counts['type']
	A = average.counts['type-des'].copy()
	B = batter.counts['type-des']
	P = pitcher.counts['type-des']

	try:
		Q = B*P/A
	except:
		pdb.set_trace()
	tot = Q.sum(axis=1)
	Q = Q.apply(lambda C: C/tot).fillna(A)
	Q2 = Q.apply(lambda C: PP*C).fillna(0).sum(level=[0,1])
	Q2 = np.array([[Q2.loc[b,s].to_dict() for s in [0,1,2]] for b in [0,1,2,3]])
	ans = markov(Q2)
	if(np.isnan(ans).any()):
		pdb.set_trace()
#	pdb.set_trace()
	return ans

# Takes into account pitch location distribution	
# Unfortunately, we cannot split by count here and we cannot take into account pitch type either
def markov4(ab_test):
	batter = batters[ab_test.batter_id]
	pitcher = pitchers[ab_test.pitcher_id]
	pdf = pitcher.pdf
	zero = pdf * 0
	cpdfs = batter.cpdfs * pitcher.cpdfs / average.cpdfs
	total = cpdfs.sum()
	comp = cpdfs.map(lambda kde: (kde/total * pdf).sum()).fillna(0) # just in case on fillna
	comp /= comp.sum()
	P = np.array([[comp.to_dict() for s in [0,1,2]] for b in [0,1,2,3]])	
	if(comp.isnull().any()): 
		pdb.set_trace()
	return markov(P)


# generic evaluation function that consumes a function and a data
# frame of test instances and computes the score
def evaluate(function, test):
	# Remove test data for uncommon players before processing
	test = test[test.pitcher_id.map(lambda x: x in pitchers.keys())]
	test = test[test.batter_id.map(lambda x: x in batters.keys())]
	predictions = test.apply(function, axis=1)		
	loss1 = logloss(predictions, test.event)
	loss2 = quadloss(predictions, test.event)
	REL, RES, UNC = brier(predictions, test.event)
	print ('Test Set', 'Log Loss: ', loss1, 'Quad Loss: ', loss2)
	print ('Brier', REL, RES, UNC)
#	train = average.atbats.copy()
#	train = train[train.pitcher_id.map(lambda x: x in pitchers.keys())]
#	train = train[train.batter_id.map(lambda x: x in batters.keys())]
#	predictions = train.apply(function, axis=1)		
#	loss1 = logloss(predictions, train.event)
#	loss2 = quadloss(predictions, train.event)
#	print ('Train Set', 'Log Loss: ', loss1, 'Quad Loss: ', loss2)
#	best = test.loc[predictions.hit.argmax()]
#	print (best)

def compute(fun, data):
	return data.apply(fun, axis=1)

## Parallel evaluation
def par_eval(pool, function, test):
	# Remove test data for uncommon players before processing
	test = test[test.pitcher_id.map(lambda x: x in pitchers.keys())]
	test = test[test.batter_id.map(lambda x: x in batters.keys())]
	
	split = np.array_split(test, cpu_count())
	foo = partial(compute, function)
	predictions = pd.concat(pool.map(foo, split))
	loss1 = logloss(predictions, test.event)
	loss2 = quadloss(predictions, test.event)
	REL, RES, UNC = brier(predictions, test.event)
	print  'Log Loss: ', loss1, 'Quad Loss: ', loss2
	print 'Brier', REL, RES, UNC


#	train = average.atbats.copy()
#	train = train[train.pitcher_id.map(lambda x: x in pitchers.keys())]
#	train = train[train.batter_id.map(lambda x: x in batters.keys())]
	
#	split = np.array_split(train, cpu_count())
#	predictions = pd.concat(pool.map(foo, split))
#	loss1 = logloss(predictions, train.event)
#	loss2 = quadloss(predictions, train.event)
	#print ('Train Set', 'Log Loss: ', loss1, 'Quad Loss: ', loss2)

#	best = test.loc[predictions.hit.argmax()]
#	print (best)


## Globals
batters = {}
pitchers = {}
average = Player()
logit = LogisticRegression()

def build(atbats, pitches):
	global average, batters, pitchers, logit

	average.setABs(atbats)
	average.setPitches(pitches)

	S = atbats.groupby('batter_id').filter(lambda x: len(x) > 250).groupby('batter_id').apply(lambda d: d.event.value_counts(normalize=True))
	T = atbats.groupby('pitcher_id').filter(lambda x: len(x) > 250).groupby('pitcher_id').apply(lambda d: d.event.value_counts(normalize=True))
	if(isinstance(S,pd.Series)): S = S.unstack().fillna(0)
	if(isinstance(T,pd.Series)): T = T.unstack().fillna(0)
	S = S.reset_index()
	T = T.reset_index()
	
	AB = atbats.merge(S,on='batter_id').merge(T, on='pitcher_id')

	logit.fit(AB[['out_x', 'hit_x','walk_x','out_y','hit_y','walk_y']].values, AB.event2)

	# League Averages
	#pdb.set_trace()
	# Can these loops be paralellized?	
	for id, ab in atbats.groupby('batter_id'):
		if(len(ab) >= 250):
			batters[id] = Player()
			batters[id].setABs(ab)
	for id, ab in atbats.groupby('pitcher_id'):
		if(len(ab) >= 250):
			pitchers[id] = Player()
			pitchers[id].setABs(ab)
	for id, p in pitches.groupby('batter_id'):
		if(id in batters):
			batters[id].setPitches(p)
	for id, p in pitches.groupby('pitcher_id'):
		if(id in pitchers):
			pitchers[id].setPitches(p)
	

def build_player(data):
	pid, atbats, pitches = data
	ans = Player()
	try:
		ans.setABs(atbats)
		ans.setPitches(pitches)
	except:
		pdb.set_trace()
	return { pid : ans }

def par_build(atbats, pitches):
	global batters, pitchers, average, logit 

	S = atbats.groupby('batter_id').filter(lambda x: len(x) > 250).groupby('batter_id').apply(lambda d: d.event.value_counts(normalize=True))
	T = atbats.groupby('pitcher_id').filter(lambda x: len(x) > 250).groupby('pitcher_id').apply(lambda d: d.event.value_counts(normalize=True))
	if(isinstance(S,pd.Series)): S = S.unstack().fillna(0)
	if(isinstance(T,pd.Series)): T = T.unstack().fillna(0)
	S = S.reset_index()
	T = T.reset_index()
	
	AB = atbats.merge(S,on='batter_id').merge(T, on='pitcher_id')
	logit.fit(AB[['out_x', 'hit_x','walk_x','out_y','hit_y','walk_y']].values, AB.event2)


#	pdb.set_trace()	
	atbat1 = atbats.groupby('batter_id')
	atbat2 = atbats.groupby('pitcher_id')
	pitch1 = pitches.groupby('batter_id')
	pitch2 = pitches.groupby('pitcher_id')
	batter  = [(g, ab, pitch1.get_group(g)) for g, ab in atbat1 if len(ab) >= 250]
	pitcher = [(g, ab, pitch2.get_group(g)) for g, ab in atbat2 if len(ab) >= 250]
	pool = Pool()
	bans = pool.map(build_player, batter)
	pans = pool.map(build_player, pitcher)
	batters = { k : v for d in bans for k,v in d.items() }	
	pitchers = { k : v for d in pans for k,v in d.items() }	

	#average.setABs(batters[batters.keys()[2]].atbats)
	#average.setPitches(batters[batters.keys()[2]].pitches)

	average.setABs(atbats)
	average.setPitches(pitches)
	


# Computed based on sample data (see plotting.py)
bounds = [[34, 170], [96, 203]] # domain of pitches we will consider

# less important features should be earlier in the multi-index
def weight(D):
	k = 50
	idx = list(D.index.names)
	stub = pd.DataFrame(index=D.index).reset_index()
	#M = pd.merge(A.reset_index(), B.reset_index(), on=idx)
	T = pd.concat([D.sum(axis=1) for _ in D.columns],axis=1)
	T.columns = D.columns
	A = D / T
	P = T
	for i in range(1,len(idx)):
		P2 = stub.merge(T.sum(level=idx[i:]).reset_index(), on=idx[i:]).set_index(idx)
		C =  stub.merge(D.sum(level=idx[i:]).reset_index(), on=idx[i:]).set_index(idx)/P2
		A = (A*P + k*C) / (P + k)
		P = P2
	return (A*P + k*D.sum()/D.sum().sum()) / (P + k)
