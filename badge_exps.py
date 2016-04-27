"""created on 15/01/2016/
@author: julius
"""

import random, math
from nsga2JF import Solution
from nsga2JF import NSGAII

import mazepy
import numpy as np 
import entropyObjectives as eob
import pickle
import itertools
from util import util

from fixedparams import *

import visuals

class MazeSolution(Solution):
    '''
    instances serve as navigator for the mazes
    '''
    def __init__(self,obj,robot=False):
        '''
        Constructor.
        '''
        Solution.__init__(self, len(obj))
        #another list for objectives /
        #next to the one provided by the Superclass Solution?
        self.selected4 = obj
        self.grid_sz = -1
        self.grid = 0 #4 personal history
        self.objs=[0.0]*len(obj_names) #here i save everything for analysis
        self.solver = False
        self.IRARflag = False  #true if IRAR has been evaluated
        if(not robot):
         self.robot=mazepy.mazenav()
         self.robot.init_rand()
         self.robot.mutate()
        else:
         self.robot=robot
         
    def evaluate_solution(self):
        '''
        Implementation of method evaluate_solution() 
        is just called once per individual's lifetime
        minimization of objectives is assumed
        fitness (between 0 and -1 (=solved)
        curiosity
        sets behavior = end location
        also increments the personal history grid
        '''
        self.robot.map()
        x = mazepy.feature_detector.endx(self.robot)
        y = mazepy.feature_detector.endy(self.robot)
        self.behavior=np.array([x,y])
        self.objs[XEND] = x
        self.objs[YEND] = y

        #record fitness and curiosity and evolvabilities
        dist2goal = mazepy.feature_detector.end_goal(self.robot)
        self.objs[FIT] = - (1-dist2goal)
        self.objs[CUR] = - mazepy.feature_detector.state_entropy(self.robot)
        if(self.robot.solution()):
         self.solver = True
         print "solution" 
        else: self.solver = False
    
    def evaluate2(self,pop, archivegrid,
                  NovArchive=None,FFAArchive=None,
                  probe_Evo=False, EvoMuts=200,
                  gammaLRAR=0.5,gammaGrid=0.5,
                  recordObj=[],
                 probe_RARs=False):
        '''
        calculate all coevolutionary objectives that require reference to the history or the other individuals in the populaiton:
        novelty, diversity, ffa, pevo, rar,sol,lrar,irar
        increment personal history grid
        and write everything into the list that is used for nsga selection
        params:
                archivegrid used for rarity computations
                nov_archive: if given used for novelty, otherwise wrt to pop only
                ffa_archive: list of fitness frequencies  throughout history
                prove_evo: if true, evolvability is evaluated with Nmuts offsprings for each individual
                gammaLRAR: factor by which herited rarity is discounted
                gammaGrid: factor by which grid reduced to measure spread of rarity for IRAR and SOL
                recordObj:list of objectives to be recorded but not selected 4
        '''
        # if individual is not viable, set all objectives to 0 so it wont be selected
        if all(self.behavior <0):
            self.objs[:] = [0.01]* len(obj_names)
            self.objs[WEIRDO] = 1
            for k in range(len(self.selected4)):
                self.objectives[k] = self.objs[self.selected4[k]]
        else:
            self.objs[WEIRDO] = 0
            if (NOV in self.selected4+recordObj) or (DIV in self.selected4):
                refPop = [x for x in pop if np.all(x.behavior>=0)] 
                self.dists=[np.sum((self.behavior-x.behavior)**2) for x in refPop]
                self.objs[DIV] = - np.mean(self.dists)
                if NovArchive != None:
                        arch_dists = [np.sum((self.behavior - a)**2) for a in NovArchive]
                        self.dists += (arch_dists)
                self.dists.sort()
                self.objs[NOV] = - np.sum(self.dists[:NNov])
            #RARITY
            if (RAR or LRAR) in self.selected4+recordObj:
                self.objs[RAR] = - eob.calc_individual_entropy(archivegrid,self,self.grid_sz)
                rar = self.objs[RAR]
                if rar ==0: 
                    print 'rar0 but viable',self.behavior
                    print archivegrid
                    a= eob.map_into_grid(self,self.grid_sz)
                    print archivegrid[a]
            #FFA
            if FFA in self.selected4+recordObj:
                self.objs[FFA] = - eob.calc_FFA(FFAArchive,self)
                ffa = self.objs[FFA]
            if PEVO in self.selected4+recordObj:
                self.objs[PEVO] = - eob.grid_contribution_to_population(self.grid, archivegrid)
            #EVOLVABILITY measure
            if probe_Evo:
                mutantgrid = eob.map_mutants_to_grid(self, EvoMuts, self.grid_sz)
                if np.sum(mutantgrid)==0:
                        self.objs[EVO] = 0
                        self.objs[REVO] = 0
                else:
                        print "writing something"
                        self.objs[EVO] = - eob.grid_entropy(mutantgrid)
                        self.objs[REVO] = - eob.grid_contribution_to_population(mutantgrid, mutantgrid+archivegrid)

            if LRAR in self.selected4 or LRAR in recordObj:
                    self.objs[LRAR] =  self.objs[RAR]+gammaLRAR *self.objs[LRAR]
            #STEPPING STONES DIVERSITY SOL
            if probe_RARs and SOL in self.selected4 + recordObj:
                lineagegrid = util.reduce_grid_sz(self.grid,gammaGrid)
                sol = - eob.grid_entropy(lineagegrid)
                self.objs[SOL] = sol
                if sol == - 100:
                    print 'SOL is off...'
            if probe_RARs and IRAR in self.selected4 + recordObj:
                    if not self.IRARflag:
                            metaGrid = util.reduce_grid_sz(self.grid,gammaGrid)
                            try:
                                    self.objs[IRAR] = -eob.calc_individual_entropy(metaGrid, self, metaGrid.shape[0])
                            except ValueError:
                                    #print "IRAR tickt nicht richtig"
                                    self.objs[IRAR]=0
                    else:
                            self.objs[IRAR] = 0
                    self.IRARflag = not self.IRARflag

            self.grid[eob.map_into_grid(self, self.grid_sz)] += 1
            for k in range(len(self.selected4)):
                self.objectives[k] = self.objs[self.selected4[k]]
        

        #write objectives into list that is used for nsga2 selection 

    def set_grid_sz(self,gs, historygrid=None):
       self.grid_sz = gs
       if historygrid==None:
           self.grid = np.zeros((gs,gs))
       else:
            self.grid = np.copy(historygrid)

    def crossover(self, other):
        '''
        Crossover of T1 solutions.
        '''
        child_solution = MazeSolution(self.selected4, self.robot.copy())
        child_solution.set_grid_sz(self.grid_sz, self.grid)
        child_solution.objs[LRAR] = int(np.copy(self.objs[LRAR]))
        child_solution.IRARflag = not self.IRARflag 
        return child_solution
    
    def mutate(self):
        '''
        Mutation of T1 solution.
        '''
        self.robot.mutate()

def write_params(paramlist, expname, trialnr):
   paramfile = './out/'+expname+str(trialnr)+'params.txt'
   with open(paramfile, 'w') as f:
      [ f.write(str(param)+"\n") for param in paramlist]

#mazepy.mazenav.initmaze("easy_maze_list.txt", "neat.ne")
#mazepy.mazenav.initmaze("medium_maze_list.txt", "neat.ne")
#mazepy.mazenav.initmaze("hard_maze_list.txt", "neat.ne")

##### PARAMS #########
NNov = 15  # neigbours looked at when computing novelty
datapath = './out/'
wallpunish = True
breakflag = True # stop trial after first success   
disp=True
NovTresh = 0.08
   
urname = "hard" # there must be a directory with this name in /out
urname = "easy"
urname = "medium" # there must be a directory with this name in /out
urname = "T"

mazelevels= [ 'superhard']
mazelevels= [ 'hard']
mazelevels= [ 'easy']
mazelevels= [ 'medium']

objsNoGrid =[]
objsGr = []
objsGr = [[RAR,SOL],[RAR,CUR],[CUR],[CUR,SOL] ]
objs2BRecorded = []#,LRAR,IRAR]
grid_szs = [10]
NPop = 100 # Population size
NGens = [1500] #according to maze level
NovGamma = int(NPop*.03)
gridGamma = .4 #how much reduce the grid to measure SOL
evoAllX = 9999
evoMutants = 20
rarsAfterX =30 
trial_start=0
Ntrials =15
No_grid_szs = [30]*len(objsNoGrid)

params = {'Npop':NPop,'Ngens': NGens[0], 'grid_sz': grid_szs[0],
           'NMutation': evoMutants,
           'kNov':NNov, 'breakAfterSolved':breakflag,
           'wallpunish':wallpunish}
#Execution starts here

import sys

if __name__ == '__main__':
#maybe add random seeding, but make sure to save it later for reproducability
    for mazelevel,ngen in zip(mazelevels,NGens):
       statfile = datapath+urname+'/'+mazelevel+'-stats.csv'
       mazepy.mazenav.initmaze(mazelevel + '_maze_list.txt', "neat.ne")
       mazepy.mazenav.random_seed()
       #the next line combines objectives with various grid sizes
       exp_list = list(itertools.product(objsGr, grid_szs))
       #and extends that list with objectives that need no variation in gridsize
       exp_list.extend( zip( objsNoGrid,No_grid_szs))
       print exp_list
       for obj, gridsz in exp_list:
          exp_name = ''+urname+ '/'
          for o in obj:
            exp_name += str(obj_names[o])
          exp_name += str(gridsz) +   mazelevel
          print exp_name
          print obj, ' with grid_sz: ', str(gridsz)
          with open(statfile,'a') as f:
             f.write(''+ exp_name + ',' + str(params) + '\n')
          for ti in range(trial_start,Ntrials):
             print 'trial: ', ti
             nsga2 = NSGAII(len(obj), mutation_rate=.9,
                            crossover_rate=1.0,
                            grid_sz = gridsz,
                            thresNov=NovTresh,
                            NovGamma=NovGamma,
                            gridGamma= gridGamma
                           )
             P = []
             for i in range(NPop):
                 P.append(MazeSolution(obj))
                 P[-1].set_grid_sz(gridsz)
             print "run nsga2"
             s = nsga2.run(P, NPop, ngen, visualization = disp, title = exp_name + str(ti),
                                   NovArchive= (NOV in obj),
                                   select4SEVO=(SEVO in obj),
                                   select4PEVO= (PEVO in obj),
                                    FFAArchive = FFA in obj,
                                   breakAfterSolved = breakflag,
                                   recordObj = objs2BRecorded,
                                  probeEvoIntervall=evoAllX,
                                  probeRARSafterX = rarsAfterX)
             with open(statfile,'a') as f:
               f.write(''+ str(s)+ '\n')
