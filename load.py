import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cross_validation import train_test_split

games = pd.read_csv('games.csv')
atbats = pd.read_csv('atbats.csv')
pitches = pd.read_csv('pitches.csv')

# Do binning here, put binning in lambda?
ptypes = ['FF','SL','FT','SI','CH','CU','FC','KC','FS','KN']
pitches = pitches[pitches.type.map(lambda t: t in ptypes)]
pitches = pitches[pitches.balls < 4]

out = ['Strikeout', 'Groundout', 'Flyout', 'Pop Out', 'Lineout', 'Forceout', 'Grounded Into DP', 'Field Error', 'Sac Bunt', 'Sac Fly', 'Bunt Groundout', 'Fielders Choice Out', 'Bunt Pop Out', 'Strikout - DP', 'Fielders Choice', 'Sac Fly DP', 'Bunt Lineout', 'Triple Play', 'Sacrifice Bunt DP']
hit = ['Single', 'Double', 'Triple', 'Home Run']
walk = ['Walk', 'Hit By Pitch', 'Intent Walk']
# bad = ['Runner Out', 'Fan interference', 'Batter Interference', 'Catcher Interference']

ball = ['Ball', 'Ball In Dirt', 'Intent Ball', 'Pitchout', 'Automatic Ball']
strike = ['Called Strike', 'Swinging Strike', 'Foul Tip', 'Swinging Strike (Blocked)', 'Foul Bunt', 'Missed Bunt', 'Swinging Pitchout', 'Automatic Strike']
foul = ['Foul', 'Foul (Runner Going)', 'Foul Pitchout']
inplay = ['In play, out(s)', 'In play, no out', 'In play, run(s)', 'Hit By Pitch']

des = lambda x: 'ball' if x in ball else 'strike' if x in strike else 'foul' if x in foul else 'inplay' if x in inplay else 'error'
evt = lambda x: 'out' if x in out else 'hit' if x in hit else 'walk' if x in walk else 'error'

atbats.event = atbats.event.map(evt)
atbats = atbats[atbats.event != 'error']

pitches.des = pitches.des.map(des)
pitches = pitches[pitches.des != 'error']
pitches = pitches[(pitches.x > 0) & (pitches.y > 0)] # is this necessary?
# pitches.loc[pitches.des == 'inplay','des'] = pitches.loc[pitches.des == 'inplay', 'event']

games['night'] = games.time_et.map(lambda s: dt.datetime.strptime(s, '%I:%M %p').time() > dt.time(17,0))
games['month'] = games.date.map(lambda s: dt.datetime.strptime(s, '%Y-%m-%d').month)

atbats['event2'] = atbats.event.map(lambda s: {'out':0,'hit':1,'walk':2}[s])
atbats['b_stand2'] = atbats.b_stand.map(lambda s: 0 if s=='L'else 1)
atbats['p_throws2'] = atbats.p_throws.map(lambda s: 0 if s=='L' else 1)
atbats['home'] = atbats.inning_half.map(lambda s: 0 if s=='bottom' else 1)

pitches['type2'] = pitches.type.map(lambda t: ptypes.index(t))
pitches['des2'] = pitches.des.map(lambda d: {'inplay':0,'strike':4,'foul':5,'ball':6}[d])


# only consider regular season games in 2014
games = games[(games.year == 2014) & (games.game_type == 'R')] 
pitches = pitches[pd.notnull(pitches.type)]
atbats = games.merge(atbats, on='game_path')

ab_train, ab_test = train_test_split(atbats, random_state=1234321)
p_train = ab_train.merge(pitches, on=['game_path', 'ab_num'])
p_train.loc[p_train.des == 'inplay','des2'] = p_train.loc[p_train.des == 'inplay', 'event2']
p_train.loc[p_train.des == 'inplay','des'] = p_train.loc[p_train.des == 'inplay', 'event']


