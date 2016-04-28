from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import  visfunction 
import itertools
from util import util
import scipy.stats
from fixedparams import *

wallcondition = 'brittle'

mazeName = 'superhard'
mazeName = 'medium'
mazeName = 'T'
mazeName = 'supereasy'
mazeName = 'easy'

mazefile = '../medium_maze.txt'
mazefile = '../s_maze2.txt'

expObjs = ['CUR/RAR/EVO','RAR/EVO','FIT/EVO','CUR/EVO','RAR/PEVO','FIT','CUR','SEVO','CUR/SEVO','RAR/CUR', 'RAR/SEVO','FIT/DIV', 'NOV','RAR/CUR/EVO/SEVO','RAR/CUR/SEVO','RAR/CUR/EVO','RAR', 'FFA']#,'RAR/CUR/PEVO','CUR/PEVO', 'PEVO/EVO',  'NOV/EVO','NOV/PEVO','FIT/PEVO']
#expObjs=['RAR/PEVO']
expObjs = ['RAR','RAR/SOL', 'CUR', 'CUR/SOL', 'LRAR','RAR/IRAR', 'FIT','NOV','FFA']

pp = PdfPages(mazeName+'-multiplot.pdf')

grid_sz= 10
cn = '' #comparison number that can be used to differ between different analyses 
exps = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs]
print 'lenexps', len(exps)
print 'exps',exps

#### load objectives or solvers


# load only trials have been solved 
solvers  = [util.load_exp_series(exp, solvers = True) for exp in exps]
#print 'solverss:', len(solvers)

firstSolved = [[ solved.keys()[0]  for solved in exp if solved != {}] for exp in solvers]
#print 'fs:', firstSolved
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

print 'expObjs',expObjs
print 'len meanfirst, firstsolved:',len(meanfirst), len(firstSolved)
# load chronic with all objectives and positions
#Ds =[util.load_exp_series(exp) for exp in exps] #Ds is a list of expseries list of(list of chronics)


####### make a summary table for the experiment ############

Ns = [len(exp) for exp in solvers]
convs=  [ len([solver for solver in exp if solver != {}])/float(len(exp)) for exp in solvers]
#print convs, Ns

filename = './'+wallcondition +'/'+str(mazeName)+str(grid_sz) + '-Summary'+str(cn)+ '.csv'
with open(filename,'w') as f:
	f.write('Objectives' + ',' + 'Solved'+ ','+ 'STD'+ ','+ 'Convergence Rate' + ',' +'N' +'\n') 
	for exp,mf,std,cr,n in zip(expObjs,meanfirst,stdfirst,convs,Ns):
		f.write(exp.replace(',','/') + ',' + "%.1f"%mf + ',' + "%.1f"%std + ',' +"%.1f"% cr + ',' +str(n) +'\n') 


####### make statistical significance table #####
ps = [ [  scipy.stats.mannwhitneyu(i,j)[1]*2 for i in firstSolved if i!= j ] for j in firstSolved]

[ps[i].insert(i, -1) for i in range(len(ps))]

with open(filename,'a') as f:
        f.write('\nDifferences between experiments significant?\n')
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
expObjs2correlate = ['RAR/SOL']
exps2correlate = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2correlate]
Ds2corr =[util.load_exp_series(exp) for exp in exps2correlate]
Rs= [util.get_correlation_table(ds,gens=[-1]) for ds in Ds2corr]


with open(filename,'a') as f:
    f.write("\nCorrelation Tables (Pearson)")
    for expname, R in zip(expObjs2correlate, Rs):
        exp = expname.split('/')
        exp += ['FIT','EVO','REVO']
        f.write( '\n ,'+str(exp)+ '\n')
        for obj in exp:
            row = obj+ ','
            for obj2 in exp:
                if obj == obj2:
                    row += '-,'
                else:
                    row +="%.2f" %R[0][get_obj_ID(obj),get_obj_ID(obj2)]  + ','
            f.write(row + '\n')
################# plot objectives against each other ################
'''
expObjs2VSPlot = ['RAR/SOL']
Gen2VisCorr= 30
exps2VSPlot = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2VSPlot]
Ds2VSPlot =[util.load_exp_series(exp) for exp in exps2VSPlot]

for ds in Ds2VSPlot:
    x_obj = [RAR]
    y_objs = [EVO,REVO, SOL]
    color_obj= FIT
    visfunction.visCorrAtGen(ds,x_obj,y_objs,color_obj,gen=Gen2VisCorr)
plt.show()
'''

############# plot average convergence rate ###########
plt.figure(0)
ax = plt.subplot2grid((1,4), (0,0), colspan=3)
#lenexps = [np.asarray([di.shape[2] for di in exp]) for exp in Ds]
firstSolvedAsArray = [np.asarray(solved) for solved  in firstSolved]

maxgen =min(1500,np.max([np.max(x) for x in firstSolvedAsArray]) )
convRates = [[len(solved[solved<=i])/float(ntrials) for i in range(maxgen)] for solved,ntrials in zip(firstSolvedAsArray, Ns)]
#convRates = [[ len(lenexp[lenexp<=i])/float(len()) for i in range(maxgen)] for lenexp in lenexps]
ax.set_xlabel('generations')
ax.set_ylabel('convergence rate')
[ax.plot( range(maxgen), convRate,label =exp)  for convRate,exp in zip(convRates,expObjs)]
#plt.legend(loc=4)
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-ConvergenceRate.png'))
pp.savefig()
#plt.show()

############ PLOT EVOLVABILITY COMPARISON ######################
expObjs2EvoComp = ['RAR','RAR/SOL', 'CUR', 'CUR/SOL', 'LRAR','RAR/IRAR', 'FIT','NOV','FFA']
expObjs2EvoComp = ['RAR','RAR/SOL']#, 'CUR', 'CUR/SOL', 'FFA']
exps2EvoComp = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2EvoComp]
Ds2EvoComp =[util.load_exp_series(exp) for exp in exps2EvoComp]
#find which generations evolvability was measured
plt.figure(22)
#find out which gens evo was measured
X = Ds2EvoComp[0][0]
nGens = X.shape[2]
gensEvoMeasured = []
AllEvos =[]
for exp in Ds2EvoComp:
    X = exp[0]
    gensEvo = [i for i in range(X.shape[2]) if np.sum(X[EVO,:,i])< 0]
    print 'gens Evo was measured', gensEvo
    gensEvoMeasured.append(gensEvo)
    meanEvos = [np.mean(d[EVO,:,gensEvo],axis=1) for d in exp]
    meanEvosExp = np.mean(meanEvos, axis=0)
    AllEvos.append(meanEvosExp)

ax = plt.subplot2grid((2,4), (0,0), colspan=3)
[ax.plot(gensEvo,-evos ,'-o', label= exp) for  evos, exp, gensEvo in zip(AllEvos, expObjs2EvoComp, gensEvoMeasured)]
ax.set_xlim([0,nGens])
ax.set_xlabel('generations')
ax.set_ylabel('mean evolvability')
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)

#color background according to first experiment
genEvoIntervall = gensEvoMeasured[0][1] - gensEvoMeasured[0][0]
[ax.axvspan(i, i+genEvoIntervall, facecolor='0.2', alpha=0.3) for i in gensEvoMeasured[0][::2]]

ax = plt.subplot2grid((2,4), (1,0), colspan=3)
weirdos = [(nGens-np.mean([np.sum(d[WEIRDO,:,:],axis=0) for d in  exp], axis=0))/float(nGens) for exp in Ds2EvoComp]
[ax.plot(range(nGens), weirdo , label=exp ) for weirdo, exp in zip(weirdos, expObjs2EvoComp)]
ax.set_xlabel('generations')
ax.set_ylabel('mean % viable children')
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
plt.show()



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
ax = plt.subplot2grid((1,4), (0,0), colspan=3)

ax.set_ylabel('average max fitness')
ax.set_xlabel('generation')

a = [ax.plot(range(len(Dmaxfit[i])),1- Dmaxfit[i], label=expObjs[i]) for i in range(len(Dmaxfit))]
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-AverageMaxFitness.png'))
pp.savefig()
plt.show()
######### boxplot ##############

plt.figure(2)
ax=plt.subplot(111)
ax = plt.subplot2grid((2,4), (1,0), colspan=3)
ax.boxplot(  firstSolved,1, '')
plt.xticks(range(1, 1+len(expObjs)),expObjs)
#ax.set_xticks(expObjs)
ax.set_ylabel('generation')
plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-Boxplot.png'))
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
