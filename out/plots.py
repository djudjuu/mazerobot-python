import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from visfunction import *
#import plotmethods as pj
from fixedparams import *
from util import util

wallcondition = 'soft'
objectives = 'RARlineageCUR'
expName = 'evoCorr'
mazelevel = 'hard'
mazelevel2 = 'hard'
mazelevel = 'medium'
grid_sz = 15
trialNr= 0
sample_sz =1
title = wallcondition + '/' + expName +'/'+mazelevel+'/' + objectives+str(grid_sz)+ 'samp' + str(sample_sz) +'-'+ str(trialNr)
title = wallcondition + '/' + expName +'/'+mazelevel+'/' + objectives+str(grid_sz)+ '-'+ str(trialNr)
title2 = wallcondition + '/' + expName +'/'+mazelevel2+'/' + objectives+str(grid_sz)+ 'samp' + str(sample_sz) +'-'+ str(trialNr)


mazefile = '../s_maze2.txt'
mazefile = '../ss_maze.txt'
mazefile2 = '../hard_maze.txt'
mazefile = '../medium_maze.txt'


##############3#load data
""" data lies in a format 
1. dim: x,y, fitness
2. dim: population size
3. dim: generations
"""
#data = np.load(title + '.npy')
data = util.load_chronic(title)
data2 = util.load_chronic(title2)

Q = data[:,100:,:]
P = data[:,:100,:]

solved = 0
with open(title + 'Solver.pkl','rb') as f:
	solved =pickle.load(f)
#print 'solved:, ', solved
_,NPop, Ngen = data.shape 
Nplot = np.min([Ngen,1500])
print 'data loaded with ', Nplot, ' generations'
nplots = 7

fs= -1	# iteration when maze was first solved
if solved != {}:
	fs = solved.keys()[0]
eliteN=10
dsorted = [sort_by_objective(data, get_obj_ID(o), eliteN, until=Nplot) for o in obj_names]
dsorted2 = [sort_by_objective(data2, get_obj_ID(o), eliteN, until=Nplot) for o in obj_names]

#### PLOT PARAMS #######
elite = 10 # in percentage
eliteN = int(NPop*(elite/100.0))
#print 'Elite: ', elite


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
want2plot = [ FIT,CUR,lineageCUR]
want2scale = []
if len(want2scale)>0:
	w2s=1
else:
	w2s=0
nplots = len(want2plot) +w2s+ 1

#### plot objectives alone ##############
for obj in want2plot:
	ax = plt.subplot2grid((nplots,4), (ii,0), colspan=3)
        #plot_objective(data,obj, ax,plot_max= obj==FIT,plot_sd=True,until= Nplot)
	plot_objective(Q,obj, ax, plot_max= False, plot_sd=True, lab='Q',until= Nplot)
	plot_objective(P,obj, ax, plot_max= False,  plot_sd=True,lab='P',until= Nplot)
	#plot_objective(dsortedEVO,obj, ax, plot_max= False,  lab='eliteSEVO',until= Nplot)
	#plot_objective(dsortedRAR,obj, ax, plot_max= False  , lab='eliteRAR',until= Nplot)
	ii += 1
	plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
	#add_solved_indication(ax, solved)

####### PLOT INDIVIDUALS IN MAZE ##############

f = plt.figure(0)
#ax = plt.subplot(121)
#labelplot(title,fs,ax)
#ax.set_title('Population until generation 200')
#drawMazeOnAxes(ax, mazefile)
#scatter_individuals(ax,data, best=False,  alle= True,until= 100)
#scatter_individuals(ax,best=False,  alle= True,X = dsortedEVO[:,:20,:],until=Nplot)

obj4best= lineageCUR
ax0 = plt.subplot(121)
ax0.set_title('Top 10 individuals for: '+str(obj4best))
drawMazeOnAxes(ax0, mazefile)
#scatter_individuals(ax1,data, best=False,  alle= True,until= Nplot,midway=False)#,X = dsorted)
#draw_path(ax1,data,nrobs=1,ngens=[20,40,60])
#scatter_individuals(ax1,data, obj_idxs = [RAR],best=True, until= Nplot)
scatter_individuals(ax0,best=False,  alle= True, X = dsorted[obj4best][:,:10,:], until=Nplot)

ax1 = plt.subplot(122)
ax1.set_title('Top 10 individuals for: '+str(obj4best))
drawMazeOnAxes(ax1, mazefile2)
scatter_individuals(ax1,data, best=False,  alle= True,until= Nplot,midway=False)#,X = dsorted)
#draw_path(ax1,data,nrobs=1,ngens=[20,40,60])
#scatter_individuals(ax1,data, obj_idxs = [RAR],best=True, until= Nplot)
#scatter_individuals(ax1,best=False,  alle= True, X = dsorted2[get_obj_ID(objectives)][:,:10,:], until=Nplot)

#ax2 = plt.subplot(133)
#ax2.set_title('Path through maze')
#drawMazeOnAxes(ax2, mazefile)
#scatter_individuals(ax1,data, best=False,  alle= True,until= Nplot,midway=True)#,X = dsorted)
#draw_path(ax2,data,nrobs=1,ngens=[20,40,60])
#scatter_individuals(ax1,data, obj_idxs = [VIAB],best=True, until= Nplot)
#scatter_individuals(ax1,best=False,  alle= True, X = dsorted)

##### plot normalized objectives together in one plot 
'''
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
