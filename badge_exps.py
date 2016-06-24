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
        self.BEHAV_DIM = 0
        self.behaviorSamples=np.zeros(1)
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

        #ACESS METHOD featuredetector ()
        x = mazepy.feature_detector.endx(self.robot)
        y = mazepy.feature_detector.endy(self.robot)
        self.behavior=np.array([x,y])
        self.objs[XEND] = x
        self.objs[YEND] = y
        prev_bhv = self.behaviorSamples.copy()
        self.behaviorSamples=np.array([self.robot.get_data_at_x(i) for i in range(self.BEHAV_DIM)])
        #self.objs[locINT] = - eob.entropy_diff_path(self,prev_bhv,
                                                    #self.grid_sz,
                                                   #maze=True)
        #print len(self.behaviorSamples)
        #increment personal grid
        for i in range(0,self.BEHAV_DIM,2):
            self.grid[
                int(self.behaviorSamples[i]*self.grid_sz),
                int(self.behaviorSamples[i+1]/self.grid_sz)
                    ] += 1

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
                 probe_RARs=False,
                 archives=None):
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
            print 'WEIRDO detected in evaluate2'
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
        
            #RARITIES
            if np.any([ r in self.selected4+recordObj for r in[ RAR , LRAR,naiveRAR,tRAR]]):
                self.objs[RAR] = - eob.HD_entropy(archives[3],self, self.grid_sz)
                #dicRAR = - eob.HD_entropy(archives[4],self, self.grid_sz)
                #oldRAR = - eob.calc_individual_entropy(archivegrid,self,self.grid_sz)
                #assert oldRAR == dicRAR #only fires when sampling rate =1
                #assert dicRAR == self.objs[RAR]
            
            if np.any([ r in self.selected4+recordObj for r in[ tRAR]]):
                self.objs[tRAR] = - eob.pos_entropy(archives[1],self, self.grid_sz)
            
            if np.any([ r in self.selected4+recordObj for r in[ smartRAR]]):
                self.objs[smartRAR] = - eob.smart_entropy(archives[0],self,self.grid_sz)
            
            if np.any([ r in self.selected4+recordObj for r in[ naiveRAR]]):
                self.objs[naiveRAR] = - eob.naive_entropy(archives[2],self,self.grid_sz)
            
            #frCUR
            if frCUR in self.selected4+recordObj:
                self.objs[frCUR] = - eob.frequency_of_objective(self,archives[4],
                                                                CUR,
                                                                math.log(1./self.grid_sz**2),
                                                                0)
            #FFA
            if FFA in self.selected4+recordObj:
                self.objs[FFA] = - eob.calc_FFA(FFAArchive,self)
                ffa = self.objs[FFA]
            #EVOLVABILITY 
            if probe_Evo or EVO in self.selected4:
                mutantgrid = eob.map_mutants_to_grid(self, EvoMuts, self.grid_sz)
                self.objs[EVO] = - eob.grid_entropy(mutantgrid)
                self.objs[REVO] = - eob.grid_contribution_to_population(mutantgrid, mutantgrid+archivegrid)
                if self.objs[EVO]==0:
                        print 'EVO is actually 0'
                del mutantgrid

            #Lineage rarity: passing on heredity from
            if LRAR in self.selected4 or LRAR in recordObj:
                    self.objs[LRAR] =  self.objs[RAR]+gammaLRAR *self.parentRar
            else: 
                    self.objs[LRAR] =  0

                    
            #Lineage Grid Entropy and Diversity
            #currently with current position, if it should be excluded uncomment this line and comment it in 152
            #self.grid[eob.map_into_grid(self, self.grid_sz)] += 1
            #self.history.append(eob.map_into_grid(self, self.grid_sz))

            if probe_RARs and (lineageCUR in self.selected4 + recordObj):
                self.objs[lineageCUR] = - eob.grid_entropy(self.grid)
            else:
                self.objs[lineageCUR] = 0

            if self.selected4 == [lineageCUR]:
                self.objs[lineageCUR] = - eob.grid_entropy(self.grid)

            if probe_RARs and (LGE in self.selected4 + recordObj or
                               LGD in self.selected4 + recordObj):
                lineagegridreduced = util.reduce_grid_sz(self.grid,gammaGrid)
                self.objs[LGE] =  - eob.grid_entropy(self.grid)
                self.objs[LGEr] =  - eob.grid_entropy(lineagegridreduced)
                self.objs[LGD] = - np.sum(self.grid>0)
                self.objs[LGDr] = - np.sum(lineagegridreduced>0)
                self.objs[LGDnd] =  -np.sum(self.grid>0)/float(np.sum(self.grid))

                #short term 
                #shSOL auf normalen grid
                recent_lineagegrid = eob.map_behaviors_to_grid(self.history[-shSOLSpan:], self.grid_sz)
                self.objs[shLGE] =  - eob.grid_entropy(recent_lineagegrid)
                self.objs[shLGD] =  -np.sum(recent_lineagegrid>0)
                self.objs[shLGDnd] =  -np.sum(recent_lineagegrid>0)/float(np.sum(recent_lineagegrid))
            else:

                self.objs[LGE] = 0
                self.objs[LGEr] =0
                self.objs[LGD] = 0
                self.objs[LGDr] =0
                self.objs[LGDnd] =0
                self.objs[shLGE] = 0
                self.objs[shLGD] = 0
                self.objs[shLGDnd] = 0

            #IRAR
            if probe_RARs and IRAR in self.selected4 + recordObj:
                    if not self.IRARflag:
                            metaGrid = util.reduce_grid_sz(self.grid,gammaGrid)
                            metaGrid[eob.map_into_grid(self,metaGrid.shape[0])] += 1
                            self.objs[IRAR] = -eob.calc_individual_entropy(metaGrid, self, metaGrid.shape[0])
                    else:
                            self.objs[IRAR] = 0
                    self.IRARflag = not self.IRARflag
            else:
                    self.objs[IRAR] = 0
            self.objs[ZERO]=0

            #if not probe_RARs: 
                #self.objs[VIAB]=0

            #self.grid[eob.map_into_grid(self, self.grid_sz)] += 1
            #self.history.append(eob.map_into_grid(self, self.grid_sz))
            
            for k in range(len(self.selected4)):
                self.objectives[k] = self.objs[self.selected4[k]]
                #VIAB is excluded here as its computation needs to be  carried on 
                # tanke false away to only select for VIAB in intervals
                if False and self.selected4[k]==VIAB and not probe_RARs :
                    self.objectives[k] = 0


    def set_grid_sz(self,gs,samples=2, historygrid=None):
       '''
        called  upon initialization of individuals
        sets grid_sz
        and indicates how many samples are taken throughout the
        walk through the maze
       '''
       self.grid_sz = gs
       self.BEHAV_DIM = samples*2
       #self.behaviorSamples=np.array([self.robot.get_data_at_x(i) for i in range(self.BEHAV_DIM)])
       if historygrid==None:
           self.grid = np.zeros((gs,gs))
       else:
            self.grid = np.copy(historygrid)

    def crossover(self, other):
        ''' Crossover of T1 solutions.
        '''
        child_solution = MazeSolution(self.selected4, self.robot.copy())
        child_solution.set_grid_sz(self.grid_sz,
                                   samples=self.BEHAV_DIM/2,
                                   historygrid=self.grid)
        child_solution.parentRar= int(np.copy(self.objs[LRAR]))
        child_solution.IRARflag = not self.IRARflag 
        child_solution.parentIDs.append(self.id)
        child_solution.history = list(self.history)
        child_solution.behaviorSamples = self.behaviorSamples.copy()
        self.childrenInQ += 1
        return child_solution
    def get_data_at_x(self,idx):
            return self.robot.get_data_at_x(idx)

    def mutate(self):
        '''
        Mutation of T1 solution.
        '''
        self.robot.mutate()


def write_params(paramlist, expname, trialnr):
   paramfile = './out/'+expname+str(trialnr)+'params.txt'
   with open(paramfile, 'w') as f:
      [ f.write(str(param)+"\n") for param in paramlist]


##### PARAMS #########
NNov = 15  # neigbours looked at when computing novelty
wallcondition = 'soft' #'soft'
wallpunish = False
NovTresh = 0.08
objsNoGrid =[[FIT],[FIT,DIV]]
objs2BRecorded = [RAR,LGD]
objsNoGrid =[]
No_grid_szs = [15]*len(objsNoGrid)
objsGr = [[RAR],[RAR,VIAB],[RAR,CUR],[RAR,EVO],[RAR,EVO,CUR],[RAR,CUR,EVO],[NOV,EVO] ]
objsGr = [[RAR,CUR,VIAB],[CUR,VIAB],[CUR],[FFA],[FIT,DIV],[FIT],[RAR,EVO]]
grid_szs = [15,18,20,23,25,30]
expName = 'gridComp'
expName = "medium" # there must be a directory with this name in /out
expName = "hard" # there must be a directory with this name in /out
gammaLRAR = .2
gridGamma = .4 #how much reduce the grid to measure SOL
shSOLSpan = 20
EvoBoosterIntervall= 50
evoMutants = 200
params = {}# 'grid_sz': grid_szs[0],'NMutation': evoMutants,'kNov':NNov, 'breakAfterSolved':breakflag,'wallpunish':wallpunish}
NPop = 100 # Population size
NovGamma = int(NPop*.03)

#### IMPORTANT PARAMS ###
breakflag =False #  stop trial after first success   
expName = "typicalRuns"
expName = "evoCorr" # there must be a directory with this name in /out
expName = "T"
mazelevels= [ 'hard','medium']
mazelevels= [ 'medium','hard']
NGens = [200]#,300] #according to maze level
objsGr=[[RAR,lineageCUR],[lineageCUR]]
objsGr=[[lineageCUR]]
sample_sz=200
grid_szs = [15]#,13,15,18,20,23,25,30]
trial_start=0
Ntrials = 5
disp=True
saveChronic=True

datapath = './out/'+wallcondition+'/'+expName +'/'
description ='experiment to whether or not the evoproxies have a good efefct on evolvability'
descr_file  = datapath+expName+'-description.txt'
with  open(descr_file, 'w') as f:
    f.write(description)

#Execution starts here
import sys

if __name__ == '__main__':
#maybe add random seeding, but make sure to save it later for reproducability
    for mazelevel,ngen in zip(mazelevels,NGens):
       statfile = datapath+expName+'-stats.csv'
       mazepy.mazenav.initmaze(mazelevel + '_maze_list.txt', "test.ne")
       mazepy.mazenav.random_seed()
       exp_list= zip( objsNoGrid,No_grid_szs)
       exp_list.extend(list(itertools.product(objsGr, grid_szs)))
       print exp_list
       for obj, gridsz in exp_list:
          exp_name = wallcondition + '/'+expName+'/'+mazelevel+ '/'
          for o in obj:
            exp_name += str(obj_names[o])
          exp_name += str(gridsz) +'samp'+str(sample_sz)+ '-'
          print exp_name
          print obj, ' with grid_sz: ', str(gridsz)
          with open(statfile,'a') as f:
             f.write(''+ exp_name + ',' + str(params) + '\n')
          for ti in range(trial_start,Ntrials):
             print 'trial: ', ti
             nsga2 = NSGAII(len(obj), mutation_rate=1,
                            crossover_rate=1.0,
                            grid_sz = gridsz,
                            thresNov=NovTresh,
                            NovGamma=NovGamma,
                            gridGamma= gridGamma,
                            gammaLRAR= gammaLRAR,
                            shSOLSpan = shSOLSpan,
                            saveChronic = saveChronic
                           )
             P = []
             for i in range(NPop):
                 P.append(MazeSolution(obj))
                 P[-1].set_grid_sz(gridsz,samples=sample_sz)
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
