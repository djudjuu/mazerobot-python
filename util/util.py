import numpy as np
import pickle
import scipy.linalg
from fixedparams import *
import entropyObjectives as eob

print "util imported from thesis/util"

def correlationMatrix(data, objs, gens=[-1]):
	'''
	gets a correlation matrix of the evolved objectives
	default is evo fit
	for the last generation
	'''
	corrs = []
	for gi in gens:
		X= data[objs,:,gi]
		corrs.append(np.corrcoef(X))
	return corrs

def doPCA(chronic,objs=[2,5,8,9],gens=[-1]):
	'''
	does a pca of the evolved objectives
	default is cur, rar, evo , sevo
	for the last generation
	'''
	us=[]
	vs = np.zeros((0,4))
	for gi in gens:
		X= chronic[objs,:,gi]
		R = X.dot(X.T)
		u,v,_ = scipy.linalg.svd(R)
		us.append(u)
		vs= np.concatenate((vs,v[np.newaxis,:]), axis=0)
	return us,vs

def map_into_array(x,y, grid_sz):
	binsize = 1.0/grid_sz
	tmp = np.clip((x/binsize, y/binsize),0, grid_sz-1)	
	return (tmp[0], tmp[1])

def chronic_to_pop_array(X, grid_sz):
	ret  = []
	for gg in range(X.shape[2]):
		A = np.zeros((grid_sz, grid_sz))
		for i in range(X.shape[1]):
			A[map_into_array(X[0,i,gg],X[1,i,gg], grid_sz)]+= 1
		ret.append(A)
	#assert the arrays are in fact different
	assert any( [ret[0].flatten()[i] != ret[1].flatten()[i] for i in range(np.size(ret[0]))])
	return ret

def make_title_from_objective(objs,names):
	obj = [o-2 for o in objs]
	string = names[obj[0]]
	for i in range(1,len(obj)):
		string += '-'+names[obj[i]]
	return string

def load_chronic(title):
	data = np.load(title+'-'+str(0)+'.npy')
	#for i in range(1,Nslices):
	#	tmp = np.load(title+str(i)+'.npy')
	#	data = np.concatenate((data, tmp), axis=2)
	i=1
	flag = True
	while flag:
		try:
		  tmp = np.load(title+'-'+str(i)+'.npy')
                  print tmp.shape, data.shape
	 	  data = np.concatenate((data, tmp), axis=2)
		  i+=1
		  #print 'found slice', str(i)
		except IOError:
		  flag = False
		  print 'found'+str(i)+'slices, totalling ', data.shape[2], ' generations'
	return data

def load_exp_series(name, Nexp=1000000, solvers = False):
	'''
	returns all chronics of a series of experiments with the given name
	'''
	exp_names = []
	flag = True
	i=0
	while flag and Nexp>i:
	  try:
	    tmp = np.load(name+'-'+str(i)+'Archive.npy')
	    exp_names.append(name+'-'+str(i))
	    i += 1
	  except IOError:
	    flag = False
	if solvers:
		expSolvers= []
		for exp in exp_names:
			with open(exp + 'Solver.pkl','rb') as f:
				expSolvers.append(pickle.load(f))
		return expSolvers
	else:
		datas = [(load_chronic(exp)) for exp in exp_names]
		return datas
		
def concatenate_trials(chronics, gens=[-1]):
        '''receives a list of chronics and concatenates them for a given generation (default last)
        returns a list of concatend trials
        '''
        ret = []
        for g in gens:
                re = np.zeros((chronics[0].shape[0],0))
                for chronic in chronics:
                        re = np.concatenate((re,chronic[:,:,g]),axis=1)
                ret.append(re)
        return ret

def get_correlation_table(chronics, gens=[-1]):
        '''receives a list of chronics and returns the correlation 
        matrix (pearson) for each given generation for all the objective, some will be nan though
        '''
        Xs = concatenate_trials(chronics,gens)
        return [np.corrcoef(X) for X in Xs]

def reduce_grid_sz(grid, factor):
    old_sz = grid.shape[0]
    new_sz = int(old_sz * factor)
    new_grid = np.zeros((new_sz,new_sz))
    for i in range(old_sz):
        for j in range(old_sz):
            new_grid[int((i/float(old_sz))*new_sz),int((j/float(old_sz))*new_sz)] += grid[i,j]
    return new_grid

#test reduce_grid_sz
#r = np.arange(16).reshape((4,4));
#print reduce_grid_sz(r,.3)
#print reduce_grid_sz(r,.9)




