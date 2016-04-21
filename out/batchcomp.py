from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import  visfunction 
import itertools
import util
import scipy.stats
from fixedparams import *

folder = '2s'
folder = 'superhard'
folder = 'medium'

expObjs = ['CUR/RAR/EVO','RAR/EVO','FIT/EVO','CUR/EVO','RAR/PEVO','FIT','CUR','SEVO','CUR/SEVO','RAR/CUR', 'RAR/SEVO','FIT/DIV', 'NOV','RAR/CUR/EVO/SEVO','RAR/CUR/SEVO','RAR/CUR/EVO','RAR', 'FFA']#,'RAR/CUR/PEVO','CUR/PEVO', 'PEVO/EVO',  'NOV/EVO','NOV/PEVO','FIT/PEVO']
#expObjs=['RAR/PEVO']
mazelevel = 'easy'
mazelevel = 'superhard'
mazelevel = 'medium'

mazelevel = folder
pp = PdfPages(mazelevel+'-multiplot.pdf')

grid_sz= 10
cn = '' #comparison number that can be used to differ between different analyses 
exps = [ folder+str('/')+s.replace('/','')+str(grid_sz)+mazelevel for s in expObjs]
print 'lenexps', len(exps)

mazefile = '../s-maze2.txt'
mazefile = '../ss_maze.txt'
mazefile = '../hard_maze2.txt'
mazefile = '../medium_maze.txt'

#### load objectives or solvers


# load only trials have been solved 
solvers  = [util.load_exp_series(exp, solvers = True) for exp in exps]

firstSolved = [[ solved.keys()[0]  for solved in exp if solved != {}] for exp in solvers]
meanfirst =[np.mean(exp) for exp in firstSolved] 
stdfirst =[np.std(exp) for exp in firstSolved] 


# rearraning from best to worst while taking out all experiments that never solved it

mfs = np.asarray(meanfirst)
order = np.argsort(mfs)
order = [o for o in order if not math.isnan(mfs[o])]
meanfirst= [meanfirst[i] for i in order]
stdfirst= [stdfirst[i] for i in order]
firstSolved = [firstSolved[i] for i in order]
expObjs = [ expObjs[i] for i in order]
solvers = [solvers[i] for i in order]
exps = [exps[i] for i in order]

print expObjs
print len(meanfirst), len(firstSolved)
# load chronic with all objectives and positions
#Ds =[util.load_exp_series(exp) for exp in exps] #Ds is a list of expseries list of(list of chronics)


################# plot objectives against each other ################
'''
ds = Ds[0] # link here to the dataset
x_obj = [RAR]
y_objs = [PEVO,CUR,FIT]
color_obj= FIT

visfunction.visCorrAtGen(ds,x_obj,y_objs,color_obj)
plt.show()
'''
####### make a summary table for the experiment ############

Ns = [len(exp) for exp in solvers]
convs=  [ len([solver for solver in exp if solver != {}])/float(len(exp)) for exp in solvers]
print convs, Ns

filename = './'+str(mazelevel)+str(grid_sz) + '-Summary'+str(cn)+ '.csv'
with open(filename,'w') as f:
	f.write('Objectives' + ',' + 'Solved'+ ','+ 'STD'+ ','+ 'Convergence Rate' + ',' +'N' +'\n') 
	for exp,mf,std,cr,n in zip(expObjs,meanfirst,stdfirst,convs,Ns):
		f.write(exp.replace(',','/') + ',' + "%.1f"%mf + ',' + "%.1f"%std + ',' +"%.1f"% cr + ',' +str(n) +'\n') 


####### make statistical significance table #####
ps = [ [  scipy.stats.mannwhitneyu(i,j)[1]*2 for i in firstSolved if i!= j ] for j in firstSolved]

[ps[i].insert(i, -1) for i in range(len(ps))]
print 'ps:', len(ps)

with open(filename,'a') as f:
        f.write('Differences between experiments significant?\n')
	f.write( '\n ,'+str(expObjs)+ '\n')
	for ie in range(len(ps)):
		row =str( expObjs[ie])+','
		for je in range(len(ps)):
			if ie == je or  meanfirst[ie]<meanfirst[je]:
				row += '- ,'
			else :
				row += 'p=' + "%.3f" % ps[ie][je]
				if ps[ie][je]<0.05:
					row += '*,' 
				else: row += ','
		f.write(row + '\n')
				
###################### make a correlation table 
expObjs2correlate = ['RAR/PEVO']
exps2correlate = [ folder+str('/')+s.replace('/','')+str(grid_sz)+mazelevel for s in expObjs2correlate]
Ds2corr =[util.load_exp_series(exp) for exp in exps2correlate]
Rs= [util.get_correlation_table(ds,gens=[-1]) for ds in Ds2corr]


with open(filename,'a') as f:
    f.write("Correlation Tables (Pearson)")
    for expname, R in zip(expObjs2correlate, Rs):
        exp = expname.split('/')
        exp += ['FIT']
        f.write( '\n ,'+str(exp)+ '\n')
        for obj in exp:
            row = obj+ ','
            for obj2 in exp:
                if obj == obj2:
                    row += '-,'
                else:
                    row +="%.2f" %R[0][get_obj_ID(obj),get_obj_ID(obj2)]  + ','
            f.write(row + '\n')


'''

############# plot average convergence rate ###########
plt.figure(0)
ax=plt.subplot(111)
#lenexps = [np.asarray([di.shape[2] for di in exp]) for exp in Ds]
firstSolvedAsArray = [np.asarray(solved) for solved  in firstSolved]

maxgen =min(1500,np.max([np.max(x) for x in firstSolvedAsArray]) )
convRates = [[len(solved[solved<=i])/float(ntrials) for i in range(maxgen)] for solved,ntrials in zip(firstSolvedAsArray, Ns)]
#convRates = [[ len(lenexp[lenexp<=i])/float(len()) for i in range(maxgen)] for lenexp in lenexps]
ax.set_xlabel('generations')
ax.set_ylabel('convergence rate')
[ax.plot( range(maxgen), convRate,label =exp)  for convRate,exp in zip(convRates,expObjs)]
plt.legend(loc=4)
plt.savefig('./'+mazelevel+'/'+mazelevel+str(grid_sz)+str('-ConvergenceRate.png'))
pp.savefig()
plt.show()
'''

'''

############## plot average maximum fitness over generations #################
# manipulate the data:
#1) sort by objective
DsortedByObjective = [[ np.sort(di,axis=1) for di in exp] for exp in Ds]

#2)make all the same length and save that adjusted list in DDs
DDs = []
for exp in  Ds:
	maxgen =max( [ di.shape[2] for di in exp])
	nobjs, npop,_ = exp[0].shape
	eexp = []
	for di in exp:
		dii=np.concatenate((di, np.zeros((nobjs,npop,maxgen - di.shape[2]))), axis=2)
		eexp.append(dii)
	DDs.append(eexp)

#3) average over max 
Dmaxfit = [np.mean([ np.sort(di,axis=1)[FIT,0,:] for di in exp],axis=0) for exp in DDs]

#4) actual plotting
plt.figure(1)
ax = plt.subplot(111)

ax.set_ylabel('average max fitness')
ax.set_xlabel('generation')

a = [ax.plot(range(len(Dmaxfit[i])),1- Dmaxfit[i], label=expObjs[i]) for i in range(len(Dmaxfit))]
plt.legend(loc=4)
plt.savefig('./'+mazelevel+'/'+mazelevel+str(grid_sz)+str('-AverageMaxFitness.png'))
pp.savefig()
plt.show()
######### boxplot ##############

plt.figure(2)
ax=plt.subplot(111)
ax.boxplot(  firstSolved,1, '')
plt.xticks(range(1, 1+len(expObjs)),expObjs)
#ax.set_xticks(expObjs)
ax.set_ylabel('generation')
plt.savefig('./'+mazelevel+'/'+mazelevel+str(grid_sz)+str('-Boxplot.png'))
pp.savefig()
plt.show()

'''
'''
########## scatter plots #############
# all series in one plot
plt.figure()
Nscatter = 290
nexp=2
for i in range(len(exps)):
	ax = plt.subplot(1, len(exps),i)
	drawMazeOnAxes(ax, mazefile)
	for ii in range(nexp):	#d in Ds[i]:
		d = Ds[i][ii]
		scatter_individuals(ax,d,
				 until= d.shape[2], 
				best=False, alle=True)		
	ax.set_title(exps[i] + ', '+ str(nexp)+ 'trials')

plt.show()
'''
pp.close()
