"""created on 15/01/2016/
@author: julius
"""

import random, math
from nsga2JF import Solution
from nsga2JF import NSGAII

import mazepy
import numpy
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
        fitness
        curiosity
        sets behavior = end location
        also increments the personal history grid
        '''
        self.robot.map()
        x = mazepy.feature_detector.endx(self.robot)
        y = mazepy.feature_detector.endy(self.robot)
        self.behavior=numpy.array([x,y])
        self.grid[eob.map_into_grid(self, self.grid_sz)] += 1

        #record fitness and curiosity and evolvabilities
        self.objs[FIT]  = mazepy.feature_detector.end_goal(self.robot)
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
        novelty, diversity, ffa, pevo, rar,
        and write everything into the list that is used for nsga selection
        if an archive is given for novelty (list of end positions), then it will be used
        otherwise novelty is computed wrt to parent and current population
        params:
                archivegrid used for rarity computations
                nov_archive: if given used for novelty, otherwise wrt to pop only
                ffa_archive: list of fitness frequencies  throughout history
                prove_evo: if true, evolvability is evaluated with Nmuts offsprings for each individual
                gammaLRAR: factor by which herited rarity is discounted
                gammaGrid: factor by which grid reduced to measure spread of rarity for IRAR and SOL
                recordObj:list of objectives to be recorded but not selected 4
        '''
        #high distances to nearest neighbours are good
        #  but for nsga2 low is better
        # thats why - sum...
        if (NOV in self.selected4+recordObj) or (DIV in self.selected4):
            refPop = [x for x in pop if all(x.behavior>=0)] #initially some individuals have end positions outside the maze...those are excluded
            if len(refPop)==0:  print 'refPop 0...makes no sense!'
            self.dists=[sum((self.behavior-x.behavior)**2) for x in refPop]
            self.objs[DIV] = - numpy.mean(self.dists)
            if NovArchive != None:
                    #print "used"
                    arch_dists = [sum((self.behavior - a)**2) for a in NovArchive]
                    self.dists += (arch_dists)
            self.dists.sort()
            self.objs[NOV] = - sum(self.dists[:NNov])
        if RAR in self.selected4+recordObj:
            self.objs[RAR] = - eob.calc_individual_entropy(archivegrid,self,self.grid_sz)
        if FFA in self.selected4+recordObj:
            self.objs[FFA] = - eob.calc_FFA(FFAArchive,self)
        if PEVO in self.selected4+recordObj:
            self.objs[PEVO] = - eob.grid_contribution_to_population(self.grid, archivegrid)
        if probe_Evo:
                self.objs[EVO] = - eob.grid_entropy(eob.map_mutants_to_grid(self, EvoMuts, self.grid_sz))
        if probe_RARs and LRAR in self.selected4+recordObj:
                self.objs[LRAR] *= gammaLRAR + self.objs[RAR]
        if probe_RARs and SOL in self.selected4 + recordObj:
                self.objs[SOL] = - eob.grid_entropy(util.reduce_grid_sz(self.grid,gammaGrid))
        if probe_RARs and IRAR in self.selected4 + recordObj:
                if not self.IRARflag:
                        metaGrid = util.reduce_grid_sz(self.grid,gammaGrid)
                        try:
                                self.objs[IRAR] = -eob.calc_individual_entropy(metaGrid, self, metaGrid.shape[0])
                        except ValueError:
                                print "IRAR tickt nicht richtig"
                                self.objs[IRAR]=0
                else:
                        self.objs[IRAR] = 0
                self.IRARflag = not self.IRARflag



        for k in range(len(self.selected4)):
            self.objectives[k] = self.objs[self.selected4[k]]
        

        #write objectives into list that is used for nsga2 selection 

    def set_grid_sz(self,gs, historygrid=None):
       self.grid_sz = gs
       if historygrid==None:
           self.grid = numpy.zeros((gs,gs))
       else:
            self.grid = historygrid

    def crossover(self, other):
        '''
        Crossover of T1 solutions.
        '''
        child_solution = MazeSolution(self.selected4, self.robot.copy())
        child_solution.set_grid_sz(self.grid_sz, self.grid)
        child_solution.objs[LRAR] = self.objs[LRAR]
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
NPop = 100 # Population size
NMutation = 10 # how many offsprings are produced to calculate evolvability
wallpunish = False
breakflag = True # stop trial after first success   
disp=True
NovTresh = 0.08
NGens = [400] #according to maze level
   
urname = "hard" # there must be a directory with this name in /out
urname = "T"
urname = "medium" # there must be a directory with this name in /out

mazelevels= [ 'superhard']
mazelevels= [ 'hard']
mazelevels= [ 'medium']


#superhard
objsNoGrid =[]
objsGr = [ ]
objsGr = [ [RAR,IRAR],[RAR,LRAR],[RAR],[RAR,SOL,IRAR]]
objs2BRecorded = [SOL,LRAR,IRAR]
grid_szs = [10]
evoAllX = 30
evoMutants = 100
rarsAfterX = 20
trial_start=0
Ntrials = 30
No_grid_szs = [grid_szs[0]]*len(objsNoGrid)

params = {'Npop':NPop,'Ngens': NGens[0], 'grid_sz': grid_szs[0],
           'NMutation': NMutation,
           'kNov':NNov, 'breakAfterSolved':breakflag,
           'wallpunish':wallpunish}
#Execution starts here

import sys

if __name__ == '__main__':
#maybe add random seeding, but make sure to save it later for reproducability
    for mazelevel,ngen in zip(mazelevels,NGens):
       statfile = './out/'+urname+'/'+mazelevel+'-stats.csv'
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
             nsga2 = NSGAII(len(obj), mutation_rate=.9, crossover_rate=1.0, grid_sz = gridsz,thresNov=NovTresh)
             P = []
             for i in range(NPop):
                 P.append(MazeSolution(obj))
                 P[-1].set_grid_sz(gridsz)
             print "run nsga2"
             s = nsga2.run(P, NPop, ngen, visualization = disp, title = exp_name + str(ti),
                                   NovArchive=(NOV in obj),
                                   select4SEVO=(SEVO in obj),
                                   select4PEVO= (PEVO in obj),
                                    FFAArchive = FFA in obj,
                                   breakAfterSolved = breakflag,
                                   recordObj = objs2BRecorded,
                                  probeEvoIntervall=evoAllX,
                                  probeRARSafterX = rarsAfterX)
             with open(statfile,'a') as f:
               f.write(''+ str(s)+ '\n')
