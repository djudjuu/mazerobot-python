import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import util
from fixedparams import *

def visCorrAtGen(triallist,x_objs, y_objs,color_obj,gen=-1):
        '''visualizes the interactions of different objectives against 
        each other and colors by a third one
        expect exp to be a  list of trials of the same experiment
        so far only works for the last generation
        '''
        #centering data and dividing by variance
        ds = triallist
        R = util.get_correlation_table(ds, gens=[-1])
        print R[0]
        
        Xraw = util.concatenate_trials(ds,gens=[-1])[0]
        Xcentered = Xraw - np.mean(Xraw,axis=1)[:,np.newaxis]
        X = Xcentered / np.std(Xcentered,axis=1)[:,np.newaxis]
        X = -X #nsga2 asumes  minimization of objectives
        X = X[2:,:]     # discard x,y posiitons
       
        plt.figure(33)
        nrows = len(x_objs)
        ncols = len(y_objs)
        for xobj,xi in zip(x_objs,range(nrows)):
            sc=0
            for yobj, yi in zip(y_objs,range(ncols)):
                ax = plt.subplot(nrows,ncols,ncols*xi+yi)
                ax.set_title(obj_names[xobj] + 'vs' +obj_names[yobj] +'\n r = ' + "%.2f"%R[0][xobj,yobj])
                ax.set_xlabel(obj_names[xobj])
                ax.set_ylabel(obj_names[yobj])
                sc=ax.scatter(X[xobj,:],X[yobj,:],c=X[color_obj][:,np.newaxis],cmap='jet')
            plt.colorbar(sc)




def plot_histogramm(stats_grid, name='exp'):
	f=plt.figure()	
	im= plt.imshow(stats_grid, cmap = plt.get_cmap('jet'))
	plt.show()
	f.savefig(name + '.png')
	
	

#makes an animated plot of the populations density through the generations
#expects a figure
# a list of arrays containing the grids of subsequent runs
def animated_density(fig,X, until=30 , contrastfactor = 1):
	im= plt.imshow(X[0], cmap = plt.get_cmap('jet'))
	#iii=0
	def updatefig(*args):
		global iii	
		iii += 1
		iii = iii% until
		im.set_array(X[iii]*contrastfactor)
		return im,
	
	ani = animation.FuncAnimation(fig, updatefig, interval=90, blit=True, repeat_delay=5000)
	return ani

def add_solved_indication(ax,solved,  solv_thres=5, clr= 'black'):
	if solved != {}:
		fs = solved.keys()[0]
		i=1
		while np.sum(solved.values()[:i]) < solv_thres:
			i+=1
		cs = solved.keys()[i-1]
		ax.axvline(fs,linestyle =  '--')#, color = clr) #ax.axvline(cs)
		#plt.xticks(list(plt.xticks()[0]) + [first_solved, considered_solved])

def plot_derivative(x, y, ax, lab = '', until = 30, dv = 1):
	xx = x[:until]
	yy = y[:until]
	for i in range(dv):
		yy =  np.diff(yy)/np.diff(xx)
		xx = xx[:len(yy)]
	ax.plot(xx,yy, label = lab+dv*"'")

def drawMazeOnAxes(axes, mazefile):
	'''
	draws a given maze into a 2d plot
	'''
	mazedata = []
	with  open(mazefile,'r') as f: 
		mazedata = [map(int, line.split()) for line in f]
	md= np.asarray(mazedata[7:])
	xmin = np.min(md[:,0::2])
	ymin = np.min(md[:,1::2])
	md[:,0::2] -= xmin
	md[:,1::2] -= ymin
	#print md
	xmax = np.max(md[:,0::2])
	ymax = np.max(md[:,1::2])
	norm = np.asarray([xmax, ymax, xmax, ymax]) * 1.0
	start = (np.asarray(mazedata[3])-np.asarray([xmin, ymin]))/norm[:2]
	goal  = (np.asarray(mazedata[5])-np.asarray([xmin, ymin]))/norm[:2]
	lines = md/norm
	#print lines
	axes.scatter(start[0], start[1] )
	axes.scatter(goal[0], goal[1],marker = 'x')
	for row in lines:
		axes.plot(row[::2], row[1::2], lw = 2.1, color = 'black')

mazelist = ['u-maze.txt', 's-maze2.txt','s-maze.txt','weave-maze.txt','weavy-maze.txt','weave-maze3.txt','weave-maze4.txt','weave-maze5.txt','weave-maze6.txt','weave-maze7.txt','hard_maze.txt', 'decep_maze.txt','hard_maze.txt']
mazelist = [ '2s_maze.txt','hard_maze.txt', 'medium_maze.txt']
def draw_mazes(mazes=mazelist):
	plt.figure()
	axes = []
	Nm=len(mazes)
	for i in range(Nm):
		if Nm <4:
			axes.append(plt.subplot(1,Nm,i))
		else:
			axes.append(plt.subplot((Nm/3)+1,3,i))
	for maze, axis in zip(mazes, axes):
		axis.set_title(maze)
		plt.axis('off')
		print maze
		drawMazeOnAxes(axis,'../'+maze)
	

def labelplot(title,fs,ax, params = None):
	obj_string = ['NOV','PEVO', 'EVO', 'CUR', 'FIT', 'RAR', 'SEVO', 'DIV']
	objs = [ s for s in obj_string if s in title]
	xpnr = title[-1]
	s = 'objective: ' + str(objs)+ ', Exp-Nr: '+str( xpnr) + '\n solved: ' + str(fs)
	if params != None:
		s += str(params)
	ax.set_title(s)

def sort_by_objective(data, obj, top, until = 30):
	'''
	this quagmire sorts the individual column of the 3dim data matrix
	 according to one objectve
	until the untilst generation
	returns the top individuals
	'''
	idx = np.argsort(data, axis= 1)
	dsorted = np.zeros((data.shape[0],top,0))
	for i in range(until):
		tmp = data[:,idx[obj,:top,i],i][:,:,np.newaxis]
		dsorted=np.concatenate((dsorted,tmp ), axis=2)
	return dsorted


def plot_objective(X, obj_idx, ax,  until = 30, plot_max=True,  lab = None):
	if lab == None:
		lab = datax[obj_idx]
	ax.set_title(datax[obj_idx])
	if obj_idx == FIT:
		ax.plot(np.arange(until),1-np.mean(X, axis=1)[obj_idx,:until], label = 'mean '+lab)
		if plot_max:
			ax.plot(np.arange(until),1- np.min(X, axis=1)[obj_idx,:until], label = 'max ' +lab)
	else:
		ax.plot(np.arange(until),np.mean(-X, axis=1)[obj_idx,:until], label = 'mean '+lab)
		if plot_max:
			ax.plot(np.arange(until), np.max(-X, axis=1)[obj_idx,:until], label = 'max ' +lab)

def clear_axis(ax):
	plt.sca(ax)
	plt.cla()

cmaps = ['RdYlBu','autumn', 'spring', 'summer', 'winter', 'Greys']
markers = ['o','v','s', 'p', '8','D'] 
def scatter_individuals(ax,X,  obj_idxs=None, until = 30, best=True, alle=False, cm = cmaps, markr = markers    ):
	ii = 0
	if best:
		sc = 0
		for obj_idx in obj_idxs:
			#sort by objective
			idx = np.argmin(X[obj_idx,:,:until], axis= 0)
			if obj_idx == FIT:
				idx = np.argmax(X[obj_idx,:,:until], axis= 0)
			xs = []
			ys =[]
			for i in range(until):
				xs.append(X[0,idx[i],i])
				ys.append(X[1,idx[i],i])
			sc = ax.scatter(xs, ys, c = np.arange(until),marker = markr[ii], cmap=cm[0], s=25.0, label =datax[obj_idx]) 
			ii += 1
		plt.colorbar(sc)
		plt.legend(prop={'size':10})
		plt.tick_params( axis='x', which='both',
			    bottom='off', top='off')
	if alle:
		for i in range(X.shape[1]):
			ax.scatter(X[0,i,:until], X[1,i,:until], c = np.arange(-until,0), cmap='hot')




