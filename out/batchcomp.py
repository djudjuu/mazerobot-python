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

wallcondition = 'soft'#soft''

mazeName = 'superhard'
mazeName = 'supereasy'
mazeName = 'easy'
mazeName = 'gridComp'
mazeName = 'T'
mazeName = 'mediumNegSOL'
mazeName = 'medium'

mazefile = '../medium_maze.txt'
mazefile = '../s_maze2.txt'

expObjs = ['CUR/RAR/EVO','RAR/EVO','FIT/EVO','CUR/EVO','RAR/PEVO','FIT','CUR','SEVO','CUR/SEVO','RAR/CUR', 'RAR/SEVO','FIT/DIV', 'NOV','RAR/CUR/EVO/SEVO','RAR/CUR/SEVO','RAR/CUR/EVO','RAR', 'FFA']#,'RAR/CUR/PEVO','CUR/PEVO', 'PEVO/EVO',  'NOV/EVO','NOV/PEVO','FIT/PEVO']
#expObjs=['RAR/PEVO']
expObjs = ['RAR']
expObjs = ['RAR/SOLnd','RAR/SOLr','RAR/VIAB','RAR/shSOLr','RAR/shSOLnd']#,'RAR/shSOLnd'] #normal
expObjs = ['RAR/SOLnd','RAR/SOLr','RAR/SOLnd','RAR/SOLrnd','RAR/shSOLr','RAR/shSOLrnd']#,'RAR/shSOLnd']# negSOL
expObjs = ['RAR/SOLnd','RAR/SOLr','RAR/SOLnd','RAR/shSOLr',]#,'RAR/shSOLnd']# medium
expObjs = ['RAR/SOLr','RAR/VIAB','NOV','NOV/VIAB','FFA','FIT'] #normal

pp = PdfPages(mazeName+'-multiplot.pdf')

grid_szs= [13,18,20,25,300]
grid_szs= [8,10,13,15,18,20,23,25,30]
grid_szs= [15]
cn = '' #comparison number that can be used to differ between different analyses 
exps = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s,grid_sz in list(itertools.product(expObjs,grid_szs))]
print 'lenexps', len(exps)
print 'exps',exps

expObjs = [e+str(grid_sz) for e,grid_sz in list(itertools.product(expObjs,grid_szs))]
print expObjs

########### load objectives or solvers (and sorting them)

# load only trials have been solved 
solvers  = [util.load_exp_series(exp, solvers = True) for exp in exps]
#print 'solverss:', len(solvers)

firstSolved = [[ solved.keys()[0]  for solved in exp if solved != {}] for exp in solvers]
#print 'fs:', firstSolved
meanfirst =[np.mean(exp) for exp in firstSolved] 
stdfirst =[np.std(exp) for exp in firstSolved] 
convs=  [ len([solver for solver in exp if solver != {}])/float(len(exp)) for exp in solvers]

########### rearraning from best to worst while taking out all experiments that never solved it
#sort after ConvRate, then after speed
criteria = [exps,expObjs,solvers,meanfirst,firstSolved,stdfirst,convs]


sumExp = zip(exps,expObjs,solvers,meanfirst,firstSolved,stdfirst,convs)
sumExp = zip(*criteria)

sumExp=sorted(sumExp, key=lambda x: (-x[6],x[3]))

exps = [e[0] for e in sumExp]
expObjs = [e[1] for e in sumExp]
solvers = [e[2] for e in sumExp]
meanfirst = [e[3] for e in sumExp]
firstSolved = [e[4] for e in sumExp]
stdfirst = [e[5] for e in sumExp]
convs = [e[6] for e in sumExp]
# [crit = [e[i] for e in sumExp] for i,crit in enumerate(criteria)]

print 'expObjs',expObjs
#print 'len meanfirst, firstsolved:',len(meanfirst), len(firstSolved)

# load chronic with all objectives and positions
#Ds =[util.load_exp_series(exp) for exp in exps] #Ds is a list of expseries list of(list of chronics)

########################## SUMMARY  ############

Ns = [len(exp) for exp in solvers]
#convs=  [ len([solver for solver in exp if solver != {}])/float(len(exp)) for exp in solvers]
#print convs, Ns
filename = './'+wallcondition +'/'+str(mazeName)+str(grid_sz) + '-Summary'+str(cn)+ '.csv'
with open(filename,'w') as f:
	f.write('Objectives' + ',' + 'Convergence Rate' + ',' +'Solved at'+ ','+ 'STD'+ ','+ 'N' +'\n') 
	for exp,mf,std,cr,n in zip(expObjs,meanfirst,stdfirst,convs,Ns):
		f.write(exp.replace(',','/') + ',' + "%.2f"% cr+',' +"%.1f"%mf + ',' + "%.1f"%std + ',' +str(n) +'\n') 
print 'summary table made...\n'


######################### STATISTICAL SIGNIFICANCE #####
'''
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
print 'significance table made...'
'''

########################### CORRELATION ###################
print 'making correlation table...'
expObjs2correlate = ['RAR/SOLnd','RAR/SOLr','RAR/SOLnd','RAR/SOLrnd','RAR/shSOLr','RAR/shSOLrnd']#,'RAR/shSOLnd']# negSOL
expObjs2correlate = ['RAR/SOLr','RAR/VIAB'] #normal
exps2correlate = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2correlate]
Ds2corrQ =[util.load_exp_series(exp,part='Q') for exp in exps2correlate]
Ds2corrP =[util.load_exp_series(exp,part='P') for exp in exps2correlate]
Ds2corr =[util.load_exp_series(exp,part='all') for exp in exps2correlate]

evozeros = [np.mean([ np.sum(d[EVO,:,-1]==0)/float(d.shape[1]) for d in Ds]) for Ds in Ds2corrQ]
print 'on average Evo==0 for ', evozeros, '% of robots in the ELITE?!?!'

gensEvo = [[i for i in range(X[0].shape[2]) if np.sum(X[0][EVO,:,i])< 0] for X in Ds2corrQ]
print 'generations here EVO was measured:',gensEvo

RsP= [util.get_correlation_table(ds,gens=[gen[-1]]) for ds,gen in zip(Ds2corrP,gensEvo)]
RsQ= [util.get_correlation_table(ds,gens=[gen[-1]]) for ds,gen in zip(Ds2corrQ,gensEvo)]
RsAll = [util.get_correlation_table(ds,gens=[gen[-1]]) for ds,gen in zip(Ds2corr,gensEvo)]

#with open(filename,'a') as f:
    #f.write("\nCorrelation Tables (Pearson)\n (parents)")
    #for expname, R in zip(expObjs2correlate, RsP):
        #exp = expname.split('/')
        #f.write('\n'+expname + '\n:')
        #exp += ['FIT','EVO','REVO','RAR','shSOL','VIAB','shSOLr','shSOLnd','shSOLrnd','SOLr','SOLnd','SOLrnd',]
        #f.write( '\n ,'+str(exp)+ '\n')
        #for obj in exp:
            #row = obj+ ','
            #for obj2 in exp:
                #if obj == obj2:
                    #row += '-,'
                #else:
                    #row +="%.2f" %R[0][get_obj_ID(obj),get_obj_ID(obj2)]  + ','
            #f.write(row + '\n')

#with open(filename,'a') as f:
#    f.write("\nCorrelation Tables (Pearson) (data of children)")
#    for expname, R in zip(expObjs2correlate, RsQ):
#        exp = expname.split('/')
#        f.write('\n'+expname + '\n:')
#        exp += ['FIT','EVO','REVO','RAR','shSOL','VIAB']
#        f.write( '\n ,'+str(exp)+ '\n')
#        for obj in exp:
#            row = obj+ ','
#            for obj2 in exp:
#                if obj == obj2:
#                    row += '-,'
#                else:
#                    row +="%.2f" %R[0][get_obj_ID(obj),get_obj_ID(obj2)]  + ','
##            f.write(row + '\n')
with open(filename,'a') as f:
    f.write("\nCorrelation Tables (Pearson)\n (all)")
    for expname, R in zip(expObjs2correlate, RsAll):
        exp = expname.split('/')
        f.write('\n'+expname + '\n:')
        exp += ['FIT','EVO','REVO','RAR','shSOL','VIAB','shSOLr','shSOLnd','shSOLrnd','SOLr','SOLnd','SOLrnd',]
        f.write( '\n ,'+str(exp)+ '\n')
        for obj in exp:
            row = obj+ ','
            for obj2 in exp:
                if obj == obj2:
                    row += '-,'
                else:
                    row +="%.2f" %R[0][get_obj_ID(obj),get_obj_ID(obj2)]  + ','
            f.write(row + '\n')
print "Correlations table made...\n"
################# plot objectives against each other ################
'''
print 'preparing to plot objectives against each other...'
expObjs2VSPlot = ['RAR/VIAB']
Gen2VisCorr= -1
exps2VSPlot = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2VSPlot]
Ds2VSPlotQ =[util.load_exp_series(exp, part='Q') for exp in exps2VSPlot]
Ds2VSPlotP =[util.load_exp_series(exp, part='P') for exp in exps2VSPlot]

for ds in Ds2VSPlotP:
    print 'dsshape:' ,ds[0].shape
    x_obj = [SOLnd,shSOLr,VIAB]
    y_objs = [EVO]
    color_obj= FIT
    visfunction.visCorrAtGen(ds,x_obj,y_objs,color_obj,gen=Gen2VisCorr)
pp.savefig()
plt.show()
'''
############ average convergence rate ###########
'''
plt.figure('average convergence rate')
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
plt.show()
'''

# ############# EVOLVABILITY COMPARISONS #########################
'''
print 'starting to look at evolvability...'
expObjs2EvoComp = ['RAR/SOLnd','RAR/shSOLrnd' ]
expObjs2EvoComp = ['RAR/SOLnd','RAR/SOLr','RAR/SOLnd','RAR/SOLrnd','RAR/shSOLr','RAR/shSOLrnd']#,'RAR/shSOLnd']# negSOL
expObjs2EvoComp = ['RAR/SOLnd','RAR/SOLr','RAR/SOLnd','RAR/shSOLr',]#,'RAR/shSOLnd']# medium
exps2EvoComp = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2EvoComp]
Ds2EvoCompQ =[util.load_exp_series(exp,part='Q') for exp in exps2EvoComp]
Ds2EvoCompP =[util.load_exp_series(exp,part='P') for exp in exps2EvoComp]
Ds2EvoCompAll =[util.load_exp_series(exp,part='all') for exp in exps2EvoComp]


plt.figure('How does the evolvability develop over time?')
#find out which gens evo was measured
X = Ds2EvoCompQ[0][0]
nGens = X.shape[2]
gensEvoMeasured = []
AllEvos =[]
AllEvosP =[]
for exp, expAll in zip(Ds2EvoCompQ,Ds2EvoCompAll):
    X = exp[0]
    print 'Xshape :' , X.shape
    gensEvo = [i for i in range(X.shape[2]) if np.sum(X[EVO,:,i])< 0]
    print 'gens Evo was measured', gensEvo
    gensEvoMeasured.append(gensEvo)
    #Q
    meanEvos = [np.mean(d[EVO,:,gensEvo],axis=1) for d in exp]
    meanEvosExp = np.mean(meanEvos, axis=0)
    AllEvos.append(meanEvosExp)
    #All
    meanEvosP = [np.mean(d[EVO,:,gensEvo],axis=1) for d in expAll]
    meanEvosExpP = np.mean(meanEvosP, axis=0)
    AllEvosP.append(meanEvosExpP)

ax = plt.subplot2grid((1,4), (0,0), colspan=3)
colormap = plt.cm.nipy_spectral
ax.set_color_cycle([colormap(i) for i in np.linspace(0, 1,len(expObjs2EvoComp))])
#[ax.plot(gensEvo,-evos ,'-o', label= exp+'Q') for  evos, exp, gensEvo in zip(AllEvos, expObjs2EvoComp, gensEvoMeasured)]
[ax.plot(gensEvo,-evos ,'-o', label= exp+'All') for  evos, exp, gensEvo in zip(AllEvosP, expObjs2EvoComp, gensEvoMeasured)]
ax.set_xlim([0,nGens])
ax.set_xlabel('generations')
ax.set_ylabel('mean evolvability')
plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
#color background according to first experiment
genEvoIntervall = gensEvoMeasured[0][1] - gensEvoMeasured[0][0]
[ax.axvspan(i, i+genEvoIntervall, facecolor='0.2', alpha=0.3) for i in gensEvoMeasured[0][::2]]

#uncomment this to see how the rate of viable children changes
#ax = plt.subplot2grid((2,4), (1,0), colspan=3)
#weirdos = [(nGens-np.mean([np.sum(d[WEIRDO,:,:],axis=0) for d in  exp], axis=0))/float(nGens) for exp in Ds2EvoComp]
#[ax.plot(range(nGens), weirdo , label=exp ) for weirdo, exp in zip(weirdos, expObjs2EvoComp)]
#ax.set_xlabel('generations')
#ax.set_ylabel('mean % viable children')
#plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)
plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-EvolvabilityOverTime.png'))
pp.savefig()
plt.show()


##### PLOT evolvability correlation over time ###########
f = plt.figure('How does the correlation with evolvability change over time?')
RsP= [util.get_correlation_table(ds,gens=gensEvo) for ds,gensEvo in zip(Ds2EvoCompP,gensEvoMeasured)]
RsQ= [util.get_correlation_table(ds,gens=gensEvo) for ds,gensEvo in zip(Ds2EvoCompQ,gensEvoMeasured)]
#print 'lenRs', len(RsQ)

for i,exp in enumerate( expObjs2EvoComp):
    #plot correlation of objectives with evo
    ax = plt.subplot2grid((len(expObjs2EvoComp),4), (i,0), colspan=3)
    for obj in exp.split('/'):
        rsQ = [ RsQ[i][gi][EVO,get_obj_ID(obj)] for gi in range(len(RsQ[i]))]
        rsP = [ RsP[i][gi][EVO,get_obj_ID(obj)] for gi in range(len(RsP[i]))]
        #print 'corrs', rs
        ax.plot(gensEvo,rsQ, '-o',label='R(EVO,'+str(obj)+'-Q)')
        ax.plot(gensEvo,rsP, '-o',label='R(EVO,'+str(obj)+'-P)')
        
    ax.set_xlim([0,nGens])
    ax.set_ylim(top=1)
    ax.set_xlabel('generations')
    ax.set_title('selected for: '+exp)
    ax.set_ylabel('R')
    plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)

    genEvoIntervall = gensEvoMeasured[0][1] - gensEvoMeasured[0][0]
    [ax.axvspan(g, g+genEvoIntervall, facecolor='0.2', alpha=0.3) for g in gensEvoMeasured[0][::2]]

    #plot mean objectives over time under the 
    #ax = plt.subplot2grid((len(expObjs2EvoComp)*2,4), (i*2+1,0), colspan=3)
    #meansobjs = [np.mean(-d,axis=1) for d in Ds2EvoComp[i]]
    #meanObjExp = np.mean(meansobjs, axis=0)
    #print 'meanObjExp', meanObjExp.shape
    #[ax.plot(range(meanObjExp.shape[1]), meanObjExp[get_obj_ID(obj)], label= 'mean ' + obj) for obj in exp.split('/')]
    #ax.set_xlabel('generations')
    #plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)

plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-EvolvabilityCorreltaionOverTime.png'))
'''

############ plot average objectives over generations ####################
'''
expObjs2AvgComp = ['RAR','RAR/VIAB','NOV/VIAB','NOV' ]

exps2AvgComp = [ wallcondition+'/'+mazeName + '/' + s.replace('/','')+str(grid_sz) for s in expObjs2AvgComp]
Ds2AvgCompQ Ds2AvgCompP =[util.load_exp_series(exp,part='P') for exp in exps2AvgComp]
Ds2AvgCompAll =[util.load_exp_series(exp,part='all') for exp in exps2AvgComp]

part = 'P'
plt.figure('what are the mean values of the objectives ('+part+') in the compared experiments?')
obj2plot = [RAR,VIAB,PROGRESS ]

Ds2AvgComp = 0
if part == 'Q':
    Ds2AvgComp = [util.load_exp_series(exp,part='Q') for exp in exps2AvgComp]


for i,obj in enumerate(obj2plot):
    ax = plt.subplot2grid((len(obj2plot),4), (i,0), colspan=3)
    for name,exp in zip(expObjs2AvgComp,Ds2AvgComp):
        meansobjs = [np.mean(-d,axis=1) for d in exp]
        stdsobjs = [np.std(-d,axis=1) for d in exp]
        meanObjExp = np.mean(meansobjs, axis=0)
        meanStdExp = np.mean(stdsobjs,axis=0)

        print 'meanObjExp', meanObjExp.shape
        ax.plot(range(meanObjExp.shape[1]), meanObjExp[obj], label= name )
        ax.set_title(obj_names[obj])
        ax.set_ylabel('average mean'+ obj_names[obj])
        plt.legend(bbox_to_anchor=(1.05,1. ), loc=2, borderaxespad=0.)

plt.savefig('./'+wallcondition+'/'+mazeName+str(grid_sz)+str('-AverageObjectivesOvertime.png'))
pp.savefig()
plt.show()

'''
############## plot average maximum fitness over generations #################
# manipulate the data:
'''
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
'''

######### boxplot ##############

'''
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
