"""created on 15/01/2016/
@author: julius
"""

import random, math
from nsga2J import Solution
from nsga2J import NSGAII

import mazepy
import numpy
#import entropy
import entropyJ2
import entropyJ
import pickle
import itertools

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
        self.objs=[0.0]*9 # here i save everything for analysis
        self.solver = False
        if(not robot):
         self.robot=mazepy.mazenav()
         self.robot.init_rand()
         self.robot.mutate()
        else:
         self.robot=robot
         
    def evaluate_solution(self):
        '''
        Implementation of method evaluate_solution() 
        fitness
        curiosity
        evolvability
        sets behavior = end location
        also increments the personal history grid
        is just called once per individual's lifetime
        '''
        self.robot.map()
        x = mazepy.feature_detector.endx(self.robot)
        y = mazepy.feature_detector.endy(self.robot)
        self.behavior=numpy.array([x,y])
        self.grid[entropyJ2.map_into_grid(self, self.grid_sz)] += 1

        #record fitness and curiosity and evolvabilities
        self.objs[FIT]  = mazepy.feature_detector.end_goal(self.robot)
        self.objs[CUR] = - mazepy.feature_detector.state_entropy(self.robot)
        self.objs[EVO] = - entropyJ2.grid_entropy(self.grid)
        if(self.robot.solution()):
         self.solver = True
         print "solution" 
        else: self.solver = False
    
    def evaluate2(self,pop, archivegrid, NovArchive=None,FFAArchive=None):
        '''
        calculate novelty and diversity
        and write everything into the list that is used for nsga selection
        if an archive is given for novelty (list of end positions), then it will be used
        otherwise novelty is computed wrt to parent and current population
        '''
        #high distances to nearest neighbours are good
        #  but for nsga2 low is better
        # thats why - sum...
        if (NOV in self.selected4) or (DIV in self.selected4):
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
        
        #print "RAR is computed..."  
        self.objs[RAR] = - entropyJ2.calc_individual_entropy(archivegrid,self)
        #compute FFA
        self.objs[FFA] = - entropyJ2.calc_FFA(FFAArchive,self)

        for k in range(len(self.selected4)):
            self.objectives[k] = self.objs[self.selected4[k]]
        #self.objs[EVO] = - entropyJ2.grid_entropy(self.grid)

        # unmcomment PEVO if it should be recorded irrespective of evolved for or not,( good idea but not working yet)
        #self.objs[PEVO] = - entropyJ2.grid_contribution_to_population(self.grid, pop)

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
grid_szs = [10]
NGens = [600] #according to maze level
   
urname = "superhard" # there must be a directory with this name in /out
urname = "medium" # there must be a directory with this name in /out
urname = "T"

mazelevels= [ 'superhard']
mazelevels= [ 'medium']


#next up:
#- more fitdivs im medium level trials done 10-70 
#- run more of the other objectives while recording evolvability -> modify code first for speed
#- run NOVmedium with higher maxgens
#- run hard again with SEVO, RARCURSEVO, RAR30-70
#changed refpop check to >=0 see if it matters



#superhard
objsGr = []
objsGr = [ [RAR, CUR, PEVO],[CUR,PEVO],[RAR,PEVO], [PEVO,EVO] ]
objsGr = [ ]
recordPevo = False
#rarpevo bis 8 gekomme
objsNoGrid = [[FFA]]
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
             print  NOV in obj
             s = nsga2.run(P, NPop, ngen, visualization = disp, title = exp_name + str(ti),
                                           NovArchive=(NOV in obj),
                                           select4SEVO=(SEVO in obj),
                                           select4PEVO=recordPevo or (PEVO in obj),
                                            FFAArchive = FFA in obj,
                                           breakAfterSolved = breakflag) 
             with open(statfile,'a') as f:
               f.write(''+ str(s)+ '\n')
