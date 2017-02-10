import re
import pandas as pd
import urllib
import xml.etree.ElementTree
import datetime
import multiprocessing


# Player: ID, Name, Team?
# Game: ID, Date, Time, Location, Away Team, Home Team, Year
# At Bat: ID, Game ID, Pitcher ID, Batter ID, Inning, Event, BatterStand, PitcherThrows
# Pitch: ID, AtBat ID, type, x, y, des, balls, strikes

if __name__ == '__main__':
	root = 'http://gd2.mlb.com/components/game/mlb/'
	start = datetime.date(year=2014, month=1, day=1)
	end = datetime.date(year=2014, month=12, day=31)
	games = pd.DataFrame(columns=('path','date','time','away_team', 'home_team'))
	atbats = pd.DataFrame(columns=('num', 'game_path', 'pitcher_id', 'batter_id', 'event', 'b_stand', 'p_throws', 'inning_half'))
	pitches = pd.DataFrame(columns=('id', 'ab_num', 'game_path', 'type', 'x', 'y', 'des', 'balls', 'strikes'))
	gameID = 0
	abID = 0
	pitchID = 0
	
	for day in pd.date_range(start, end):
		print(day)
		inning_half = 'top'
		path = root + day.strftime('year_%Y/month_%m/day_%d/') 
		file = urllib.request.urlopen(path).read().decode('utf-8')
		folders = re.compile('\"(gid_.*mlb.*mlb.*)\"').findall(file)
		for folder in folders:
			try:
				fullpath = path + folder
				#print(fullpath + 'inning/inning_all.xml')
				webdata = urllib.request.urlopen(fullpath + 'inning/inning_all.xml').read()
			except:
				continue
			if(webdata == 'GameDay - 404 Not Found'):
				continue
			xml1 = xml.etree.ElementTree.fromstring(webdata)	
			away = xml1[0].get('away_team')
			home = xml1[0].get('home_team')
			for inning in xml1.findall('inning'):
				top = inning.find('top')
				for xml2 in top.findall('atbat'):
					num = xml2.get('num')
					bid = xml2.get('batter')
					pid = xml2.get('pitcher')
					pthrows = xml2.get('p_throws')
					bstand = xml2.get('stand')
					event = xml2.get('event')
					atbats.loc[abID] = [num, fullpath, pid, bid, event, bstand, pthrows, 'top']
					balls = 0
					strikes = 0
					for xml3 in xml2.findall('pitch'):
						des = xml3.get('des')
						pitches.loc[pitchID] = [xml3.get('id'), num, fullpath, xml3.get('pitch_type'), xml3.get('x'), xml3.get('y'), des, balls, strikes]
						if(des in ['Swinging Strike', 'Called Strike', 'Foul Tip']):
							strikes += 1
						elif(des in ['Ball', 'Ball in Dirt']):
							balls += 1
						elif(des == 'Foul' and strikes < 2):
							strikes += 1	
						pitchID += 1
					abID += 1
				
				bottom = inning.find('bottom')
				if(bottom is None):
					continue
				for xml2 in bottom.findall('atbat'):
					num = xml2.get('num')
					bid = xml2.get('batter')
					pid = xml2.get('pitcher')
					pthrows = xml2.get('p_throws')
					bstand = xml2.get('stand')
					event = xml2.get('event')
					atbats.loc[abID] = [num, fullpath, pid, bid, event, bstand, pthrows, 'bottom']
					for xml3 in xml2.findall('pitch'):
						pitches.loc[pitchID] = [xml3.get('id'), num, fullpath, xml3.get('pitch_type'), xml3.get('x'), xml3.get('y'), xml3.get('des'), 0, 0]
						if(des in ['Swinging Strike', 'Called Strike', 'Foul Tip']):
							strikes += 1
						elif(des in ['Ball', 'Ball in Dirt']):
							balls += 1
						elif(des == 'Foul' and strikes < 2):
							strikes += 1	
						pitchID += 1
					abID += 1
	
			games.loc[gameID] = [fullpath, day, 0, away, home]
			gameID += 1	
	
	games.to_csv('games.csv')	
	atbats.to_csv('atbats.csv')
	pitches.to_csv('pitches.csv')



