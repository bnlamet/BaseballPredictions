import re
import pandas as pd
import urllib
import xml.etree.ElementTree
import datetime
import multiprocessing

games = pd.DataFrame(columns=('id','date','time','location','away_team', 'home_team'))
atbats = pd.DataFrame(columns=('id', 'game_id', 'pitcher_id', 'batter_id', 'event', 'b_stand', 'p_throws', 'inning_half'))
pitches = pd.DataFrame(columns=('id', 'ab_id', 'type', 'x', 'y', 'des', 'balls', 'strikes'))

gameID = 0
abID = 0
pitchID = 0

inning_half = 'top'
root = 'http://gd2.mlb.com/components/game/mlb/'

# Player: ID, Name, Team?
# Game: ID, Date, Time, Location, Away Team, Home Team, Year
# At Bat: ID, Game ID, Pitcher ID, Batter ID, Inning, Event, BatterStand, PitcherThrows
# Pitch: ID, AtBat ID, type, x, y, des, balls, strikes

def parse_pitch(xml):
	global pitchID
	global abID
	global pitches
	pitches.loc[pitchID] = [pitchID, abID, xml.get('pitch_type'), xml.get('x'), xml.get('y'), xml.get('des'), 0, 0]
	pitchID += 1

def parse_ab(xml):
	global abID
	global gameID
	global inning_half
	global atbats
	bid = xml.get('batter')
	pid = xml.get('pitcher')
	pthrows = xml.get('p_throws')
	bstand = xml.get('stand')
	event = xml.get('event')
	atbats.loc[abID] = [abID, gameID, pid, bid, event, bstand, pthrows, inning_half]
	for xml2 in xml.findall('pitch'):
		parse_pitch(xml2)
	abID += 1

def parse_game(xml, day):
	global gameID
	global games
	global inning_half
	away = xml[0].get('away_team')
	home = xml[0].get('home_team')
	for inning in xml.findall('inning'):
		top = inning.find('top')
		inning_half = 'top'
		for xml2 in top.findall('atbat'):
			parse_ab(xml2)
			
		inning_half = 'bottom'	
		bottom = inning.find('bottom')
		if(bottom is None):
			continue
		for xml2 in bottom.findall('atbat'):
			parse_ab(xml2)	
	games.loc[gameID] = [gameID, day, 0, 0, away, home]
	gameID += 1	
		
def parse_day(day):
	global root
	print(day)
	path = root + day.strftime('year_%Y/month_%m/day_%d/') 
	file = urllib.request.urlopen(path).read().decode('utf-8')

	folders = re.compile('\"(gid_.*mlb.*mlb.*)\"').findall(file)
	for folder in folders:
		try:
			# print(path + folder + 'inning/inning_all.xml')
			webdata = urllib.request.urlopen(path + folder + 'inning/inning_all.xml').read()
		except:
			continue
		if(webdata == 'GameDay - 404 Not Found'):
			continue
		game = xml.etree.ElementTree.fromstring(webdata)	
		parse_game(game, day)

if __name__ == '__main__':
	for yr in range(2014, 2015):
		start = datetime.date(year=yr, month=1, day=1)
		end = datetime.date(year=yr, month=12, day=31)
		for day in pd.date_range(start, end):
			parse_day(day)
	games.to_csv('games.csv')	
	atbats.to_csv('atbats.csv')
	pitches.to_csv('pitches.csv')



