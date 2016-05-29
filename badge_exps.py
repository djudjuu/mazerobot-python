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
from itertools import count
from util import util

from fixedparams import *

import visuals

class MazeSolution(Solution):
    '''
    instances serve as navigator for the mazes
    '''
    _ids = count(1)

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
        self.history = []
        self.objs=[0.0]*len(obj_names) #here i save everything for analysis
        self.solver = False
        self.IRARflag = False  #true if IRAR has been evaluated
        self.parentRar = 0
        self.newInArchive = True
        self.childrenInQ = 0
        self.id = self._ids.next()
        self.parentIDs = []
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
         print 'solution, (needed ',len(self.history),' mutations.'
        else: self.solver = False
    
    def evaluate2(self,pop, archivegrid,
                  NovArchive=None,FFAArchive=None,
                  probe_Evo=False, EvoMuts=200,
                  gammaLRAR=0.2,gammaGrid=0.5,
                  shSOLSpan = 20,
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
            self.objs[:] = [0.00]* len(obj_names)
            self.objs[WEIRDO] = 1
            self.objs[EVO]=0
            self.objs[REVO]=0
            for k in range(len(self.selected4)):
                self.objectives[k] = self.objs[self.selected4[k]]
        else:
            self.objs[WEIRDO] = 0

            #NOVELTY and behav DIVERSITY
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
            if np.any([ r in self.selected4+recordObj for r in[ RAR , LRAR]]):
                self.objs[RAR] = - eob.calc_individual_entropy(archivegrid,self,self.grid_sz)
            #FFA
            if FFA in self.selected4+recordObj:
                self.objs[FFA] = - eob.calc_FFA(FFAArchive,self)
                ffa = self.objs[FFA]
            #EVOLVABILITY measure
            if probe_Evo:
                mutantgrid = eob.map_mutants_to_grid(self, EvoMuts, self.grid_sz)
                self.objs[EVO] = - eob.grid_entropy(mutantgrid)
                self.objs[REVO] = - eob.grid_contribution_to_population(mutantgrid, mutantgrid+archivegrid)
                del mutantgrid

            #Lineage rarity: passing on heredity from
            if LRAR in self.selected4 or LRAR in recordObj:
                    self.objs[LRAR] =  self.objs[RAR]+gammaLRAR *self.parentRar
            
            #STEPPING STONES DIVERSITY SOL
            #currently without current position, if it should be included uncomment this line and comment it in 152
            #self.grid[eob.map_into_grid(self, self.grid_sz)] += 1
            if probe_RARs and (SOLr in self.selected4 + recordObj or
                                SOLrnd in self.selected4 + recordObj or
                                SOLnd in self.selected4 + recordObj):
                lineagegrid = util.reduce_grid_sz(self.grid,gammaGrid)
                sol = - eob.grid_entropy(lineagegrid)
                self.objs[SOLr] = sol
                self.objs[SOLrnd] = - np.sum(lineagegrid>0)/float(np.sum(lineagegrid>0))
                self.objs[SOLnd] = - np.sum(self.grid>0)/float(np.sum(self.grid>0))
                if sol == - 100:
                    print 'SOL is off...'

            #short term SOL
            if probe_RARs and (shSOL in self.selected4 + recordObj or
                                shSOLnd in self.selected4 + recordObj or
                                shSOLr in self.selected4 + recordObj or
                               shSOLrnd in self.selected4 + recordObj):
                #shSOL auf normalen grid
                lineagegrid = eob.map_behaviors_to_grid(self.history[-shSOLSpan:], self.grid_sz)
                self.objs[shSOL] = - eob.grid_entropy(lineagegrid)
                self.objs[shSOLnd] = - np.sum(lineagegrid>0)/float(np.sum(lineagegrid>0))
                #shSOL auf reduced  grid
                lineagegridreduced = util.reduce_grid_sz(lineagegrid,gammaGrid)
                self.objs[shSOLr]= - eob.grid_entropy(lineagegridreduced)
                self.objs[shSOLrnd]= - np.sum(lineagegrid>0)/float(np.sum(lineagegrid>0))





            #IRAR
            if probe_RARs and IRAR in self.selected4 + recordObj:
                    if not self.IRARflag:
                            metaGrid = util.reduce_grid_sz(self.grid,gammaGrid)
                            metaGrid[eob.map_into_grid(self,metaGrid.shape[0])] += 1
                            self.objs[IRAR] = -eob.calc_individual_entropy(metaGrid, self, metaGrid.shape[0])
                    else:
                            self.objs[IRAR] = 0
                    self.IRARflag = not self.IRARflag

            self.grid[eob.map_into_grid(self, self.grid_sz)] += 1
            self.history.append(eob.map_into_grid(self, self.grid_sz))

            for k in range(len(self.selected4)):
                self.objectives[k] = self.objs[self.selected4[k]]
        

    def set_grid_sz(self,gs, historygrid=None):
       self.grid_sz = gs
       if historygrid==None:
           self.grid = np.zeros((gs,gs))
       else:
            self.grid = np.copy(historygrid)

    def crossover(self, other):
        ''' Crossover of T1 solutions.
        '''
        child_solution = MazeSolution(self.selected4, self.robot.copy())
        child_solution.set_grid_sz(self.grid_sz, self.grid)
        child_solution.parentRar= int(np.copy(self.objs[LRAR]))
        child_solution.IRARflag = not self.IRARflag 
        child_solution.parentIDs.append(self.id)
        child_solution.history = list(self.history)
        self.childrenInQ += 1
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
wallcondition = 'soft' #'soft'
datapath = './out/'+wallcondition+'/'
wallpunish = False
breakflag = False #True  stop trial after first success   
disp=True
NovTresh = 0.08
   
mazeName = "hard" # there must be a directory with this name in /out
mazeName = "T"
mazeName = "medium" # there must be a directory with this name in /out

mazelevels= [ 'superhard']
mazelevels= [ 'easy']
mazelevels= [ 'hard']
mazelevels= [ 'medium']

objsNoGrid =[[NOV,VIAB]]
objsNoGrid =[]
objsGr = [[RAR,SOLnd],[RAR,shSOLnd]]
objsGr = [[RAR]]
objs2BRecorded = [RAR,shSOL,SOLr]
grid_szs = [5,8,13,18,20,25,30,40]
No_grid_szs = [10]*len(objsNoGrid)
NPop = 100 # Population size
NGens = [600] #according to maze level
NovGamma = int(NPop*.03)
gammaLRAR = .2
gridGamma = .4 #how much reduce the grid to measure SOL
EvoBoosterIntervall= 1000
evoMutants = 150
trial_start=0
Ntrials = 15
shSOLSpan = 20

params = {'Npop':NPop,'Ngens': NGens[0], 'grid_sz': grid_szs[0],
           'NMutation': evoMutants,
           'kNov':NNov, 'breakAfterSolved':breakflag,
           'wallpunish':wallpunish}
#Execution starts here

import sys

if __name__ == '__main__':
#maybe add random seeding, but make sure to save it later for reproducability
    for mazelevel,ngen in zip(mazelevels,NGens):
       statfile = datapath+mazeName+'-stats.csv'
       mazepy.mazenav.initmaze(mazelevel + '_maze_list.txt', "neat.ne")
       mazepy.mazenav.random_seed()
       #the next line combines objectives with various grid sizes
       exp_list = list(itertools.product(objsGr, grid_szs))
       #and extends that list with objectives that need no variation in gridsize
       exp_list.extend( zip( objsNoGrid,No_grid_szs))
       print exp_list
       for obj, gridsz in exp_list:
          exp_name = wallcondition + '/'+mazeName+ '/'
          for o in obj:
            exp_name += str(obj_names[o])
          exp_name += str(gridsz) + '-'
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
                            gridGamma= gridGamma,
                            gammaLRAR= gammaLRAR,
                            shSOLSpan = shSOLSpan
                           )
             P = []
             for i in range(NPop):
                 P.append(MazeSolution(obj))
                 P[-1].set_grid_sz(gridsz)
             print "run nsga2"
             s = nsga2.run(P, NPop, ngen, visualization = disp, title = exp_name + str(ti),
                                   NovArchive= (NOV in obj),
                                   select4SEVO=(SEVO in obj),
                                    FFAArchive = FFA in obj,
                                   breakAfterSolved = breakflag,
                                   recordObj = objs2BRecorded,
                                  EvoBoosterIntervall=EvoBoosterIntervall,
                          probeEvoNmutants = evoMutants
                          )
             with open(statfile,'a') as f:
               f.write(''+ str(s)+ '\n')
