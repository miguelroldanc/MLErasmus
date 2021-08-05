from itertools import product
from scipy.stats import entropy
import pandas as pd
import random
from math import floor


random.seed()
opponents = ['Team '+chr(ord('A') + i) for i in range(6)]
stadiums = ['Home', 'Away']
games = pd.DataFrame(list(product(opponents, stadiums))*2, columns=['opponent', 'stadium'])
games['result'] = random.choices(["Win", "Loss", "Draw"], k=len(games))
games.head()


# 1. Entropy of the result (ignoring all other variables)
number_games = len(games)
count_win = 0
count_loss = 0
count_draw = 0
for a in games['result']:
    if a == 'Win':
        count_win += 1
    if a == 'Loss':
        count_loss += 1
    if a == 'Draw':
        count_draw += 1

p_win = count_win / number_games
p_loss = count_loss / number_games
p_draw = count_draw / number_games

entropyglobal = entropy([p_win]*count_win + [p_loss]*count_loss + [p_draw]*count_draw, base=2)
print('The entropy for the result is %f' %(entropyglobal))


# 2. Average conditional entropies H(result|stadiums) and H(result|opponent)
# 2.1. Conditional entropy H(result|stadiums)
half_games = floor(len(games['result']) / 2)
count_home = [0 for i in range(half_games)]
count_away = [0 for i in range(half_games)]
is_home = games['stadium'] == 'Home'
is_away = games['stadium'] == 'Away'
games_home = games[is_home]
games_away = games[is_away]

for a in games_home['result']:
    if a == 'Win':
        count_home[0] += 1
    if a == 'Loss':
        count_home[1] += 1
    if a == 'Draw':
        count_home[2] += 1

for a in games_away['result']:
    if a == 'Win':
        count_away[0] += 1
    if a == 'Loss':
        count_away[1] += 1
    if a == 'Draw':
        count_away[2] += 1

# 2.1.1. Conditional entropy for Home matches
p_win = count_home[0]/(half_games)
p_draw = count_home[1]/(half_games)
p_loss = count_home[2]/(half_games)

entropy_home = entropy([p_win]*count_home[0] + [p_draw]*count_home[1] + [p_loss]*count_home[2])
print('The entropy for the Home matches is: %f'%(entropy_home))

# 2.1.2. Conditional entropy for Away matches
p_win = count_away[0]/(half_games)
p_draw = count_away[1]/(half_games)
p_loss = count_away[2]/(half_games)

entropy_away = entropy([p_win]*count_away[0] + [p_draw]*count_away[1] + [p_loss]*count_away[2])
print('The entropy for the Away matches is: %f'%(entropy_away))

# 2.2. Conditional entropy for the teams
#count_teams = [[0 for i in range(len(opponents))] for j in range(3)]
#is_team = [0 for i in range(len(opponents))]
#games_team = [0 for i in range(len(opponents))]
#entropy_teams = [0 for i in range(len(opponents))]
#
#
#for i in range(len(opponents)):
#    is_team[i] = games['opponent'] == opponents[i]
#    games_team[i] = games[is_team[i]]
#    i += 1
#
#i = 0
#while i < len(opponents):
#    for a in games_team['result']:
#        if a == 'Win':
#            count_teams[i][0] += 1
#        if a == 'Loss':
#            count_teams[i][1] += 1
#        if a == 'Draw':
#            count_teams[i][2] += 1
#    i += 1


# 3. Information gained for the two variables
# 3.1. Information gained based on the stadium
IG = entropyglobal - (entropy_away + entropy_home)
print(IG)

