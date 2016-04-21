import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from visfunction import *
#import plotmethods as pj
import util

#############################Vj# params
title = 'Super/CUREVO10easy0'
title = 'TNov/FITDIV10easy0'
#title = 'hard/NOV10hard1'
title = 'ssmaze/NOV10medium0'
title = 'hard/RARCURSEVO10hard0'
title = 'smazes/RARCUREVO10easy1'
title = 'smazes/RARCUR10medium0'
title = 'T/RARCURSEVO10easy0'
title = 'T/FITDIV10easy0'
title = 'hard/CUR10hard10'
title = 'T/RAREVO10medium1'

#title = 'NOV2'
#mazefile = '../weavy-maze.txt'
#mazefile = '../medium_maze.txt'
mazefile = '../decep_maze.txt'
mazefile = '../ss_maze.txt'
mazefile = '../s_maze2.txt'
mazefile = '../hard_maze2.txt'
mazefile = '../medium_maze.txt'


##############3#load data
""" data lies in a format 
1. dim: x,y, fitness
2. dim: population size
3. dim: generations
"""
#data = np.load(title + '.npy')
data = util.load_chronic(title)
solved = 0
with open(title + 'Solver.pkl','rb') as f:
	solved =pickle.load(f)
#print 'solved:, ', solved
_,NPop, Ngen = data.shape 
Nplot = np.min([Ngen,1500])
print 'data loaded with ', Nplot, ' generations'
nplots = 7

FIT=0+2    #plain fitness
CUR=1+2    # path entropy
NOV=2+2    # Novelty as mean distance to k nearest neighbours
EVO=3+2    #  #evolvability as entropy over personal stepping stones
PEVO=4+2   #evo as an individuals contribution to the diversity of the stepping stones accumulated by the population
RAR=5+2    # Rarity wrt to accumulated history of behavior
SEVO=6+2   #one-step lookahead contribution to the diversity of the next generatio
DIV=7+2    # Diversity as mean distance to all the population

datax =['x', 'y','FIT', 'CUR', 'NOV', 'EVO','PEVO','RAR', 'SEVO', 'DIV']

#datax = ['x', 'y', 'FIT', 'CUR', 'NOJ', 'EVO','EVOS', 'NOV', 'RAR', 'SEVO','DIV']

for i in range(len(datax)):
	if datax[i] in title:
		datax[i] = datax[i] + '(selected4)'
selected4 = [] # will be filled with what was selected for
xsolv = 5
fs= -1	# iteration when maze was first solved
if solved != {}:
	fs = solved.keys()[0]
cs = -1 	# iteration when it is considered solved, after xsovl solutions
#### PLOT PARAMS #######
elite = 10 # in percentage
eliteN = int(NPop*(elite/100.0))
#print 'Elite: ', elite
dsortedCUR = sort_by_objective(data, CUR, eliteN, until= Nplot)
dsortedNOV = sort_by_objective(data, NOV, eliteN,  until= Nplot)
dsortedRAR = sort_by_objective(data, RAR, eliteN,  until= Nplot)
dsortedEVO = sort_by_objective(data, EVO, eliteN,  until= Nplot)
dsortedPEVO = sort_by_objective(data, SEVO, eliteN,  until= Nplot)
dsortedDIV = sort_by_objective(data, DIV, eliteN,  until= Nplot)


####### PLOT INDIVIDUALS IN MAZE ##############

f = plt.figure(0)
ax = plt.subplot(121)
labelplot(title,fs,ax)
drawMazeOnAxes(ax, mazefile)
scatter_individuals(ax,data, best=False,  alle= True,until= Nplot)#,X = dsorted)
#scatter_individuals(ax,best=False,  alle= True,X = dsortedDIV[:,:20,:],until=2)

ax1 = plt.subplot(122)
ax1.set_title('Best individuals')
drawMazeOnAxes(ax1, mazefile)
scatter_individuals(ax1,data, obj_idxs = [PEVO,EVO],best=True, until= Nplot)
#scatter_individuals(ax1,best=False,  alle= True, X = dsorted)

############### plot density animation ###################
#f2 = plt.figure(2)
#ax11 = plt.subplot(121)
#ax22 = plt.subplot(122)
#animated_density(f2, ax11,data,grid_sz= 2, until = Nplot, dim = 2)
#animated_density(f2, ax22,data,grid_sz= 20, until = Nplot, dim=1)

###############plot PCA #################


############## PLOT OBJECTIVES ##############

f1=plt.figure(1)
ii = 0
want2plot = [ FIT,CUR,RAR,EVO,PEVO]
want2scale = []
if len(want2scale)>0:
	w2s=1
else:
	w2s=0
derivs = []#[NOV, NOJ ]
nplots = len(want2plot) +w2s+ 1

#### plot objectives alone ##############
for obj in want2plot:
	ax = plt.subplot2grid((nplots,4), (ii,0), colspan=3)
	plot_objective(data,obj, ax,plot_max= obj!=NOV,until= Nplot)
	plot_objective(dsortedPEVO,obj, ax, plot_max= False,  lab='eliteSEVO',until= Nplot)
	plot_objective(dsortedEVO,obj, ax, plot_max= False,  lab='eliteSEVO',until= Nplot)
	#plot_objective(dsortedRAR,obj, ax, plot_max= False  , lab='eliteRAR',until= Nplot)
	ii += 1
	plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
	#add_solved_indication(ax, solved)


##### plot objectives together
ax = plt.subplot2grid((nplots,4), (ii,0), colspan=3)
ax.set_title('objectives normalized')
clrs = ['r','c']
cc=0
for obj in want2scale:
	y = np.mean(-data, axis=1)[obj,:Nplot]
	#yelite = np.mean(-dsortedNOJ, axis=1)[obj,:Nplot]
	#scalin
	#print "sfd",np.max(y)
	y /= np.max(y)
	#yelite /= np.max(yelite)
	plt.plot(np.arange(len(y)), y, label='mean '+ datax[obj],color=clrs[cc] )
	#plt.plot(np.arange(len(yelite)), yelite,'--' ,label= datax[obj]+' eliteNOJ',color= clrs[cc])
	cc += 1
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
ii+=1
'''
#### their derivatives ##########
ax = plt.subplot2grid((nplots,4), (ii,0), colspan=3)
ax.set_title('Derivatives')
for obj in derivs:
	y = np.mean(-data, axis=1)[obj,:Nplot]
	yelite = np.mean(-dsortedNOJ, axis=1)[obj,:Nplot]
	#scaling
	y /= np.max(y)
	yelite /= np.max(yelite)
	#plot_derivative(np.arange(len(y)), y, ax, lab = datax[obj], until=30 )
	plot_derivative(np.arange(len(yelite)), yelite, ax, lab = datax[obj]+' eliteNOJ', until=30 )
	#plot_derivative(np.arange(len(y)), y, ax, lab = datax[obj], until = 30,dv = 2)
#plt.legend()
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
'''
plt.show()
