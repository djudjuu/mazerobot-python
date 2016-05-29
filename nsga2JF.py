'''
Created on 07/01/2011

@author: 04610922479

modified by Julius 
'''

import sys, random
from entropy import *
import entropyObjectives as eob
import numpy as np
import pickle
import visuals
from scipy.stats.vonmises_cython import numpy
from fixedparams import *
Nslice = 100

class Solution:
    '''
    Abstract solution. To be implemented.
    '''
    
    def __init__(self, num_objectives):
        '''
        Constructor. Parameters: number of objectives. 
        '''
        self.num_objectives = num_objectives
        self.objectives = []
        for _ in range(num_objectives):
            self.objectives.append(None)
        self.attributes = []
        self.rank = sys.maxint
        self.distance = 0.0
        
    def evaluate_solution(self):
        '''
        Evaluate solution, update objectives values.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def crossover(self, other):
        '''
        Crossover operator.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def mutate(self):
        '''
        Mutation operator.
        '''
        raise NotImplementedError("Solution class have to be implemented.")
    
    def __rshift__(self, other):
        '''
        True if this solution dominates the other (">>" operator).
        '''
        dominates = False
        
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:
                return False
                
            elif self.objectives[i] < other.objectives[i]:
                dominates = True
        
        return dominates
        
    def __lshift__(self, other):
        '''
        True if this solution is dominated by the other ("<<" operator).
        '''
        return other >> self


def crowded_comparison(s1, s2):
    '''
    Compare the two solutions based on crowded comparison.
    '''
    if s1.rank < s2.rank:
        return 1
        
    elif s1.rank > s2.rank:
        return -1
        
    elif s1.distance > s2.distance:
        return 1
        
    elif s1.distance < s2.distance:
        return -1
        
    else:
        return 0


class NSGAII:
    '''
    Implementation of NSGA-II algorithm.
    '''
    current_evaluated_objective = 0

    def __init__(self, num_objectives, mutation_rate=0.1,
                 crossover_rate=1.0, grid_sz = 10,
                 thresNov =0.1, NovGamma=4,
                gridGamma =0.5,
                gammaLRAR=0.2,
                shSOLSpan=20):
        '''
        Constructor. Parameters: number of objectives, mutation rate (default value 10%) and crossover rate (default value 100%). 
        '''
        self.num_objectives = num_objectives
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.grid_sz = grid_sz
        self.thresNovelty = thresNov
        self.NovGamma = NovGamma #how many added per generations 
        self.NoveltyArchive = None
        self.gridGamma = gridGamma
        self.gammaLRAR = gammaLRAR
        self.shSOLSpan = shSOLSpan

        
        random.seed();
        
    def run(self, P, population_size, num_generations,
                                    visualization=False, title='oook',
                                     NovArchive = False,
                                        FFAArchive=False,
                                      select4SEVO=False,
                                      breakAfterSolved=False, NMutations=10,
                                        recordObj=[],
                                        EvoBoosterIntervall=50,
                                    probeEvoNmutants =150):
        '''
        Run NSGA-II. 
        '''
        for s in P:
            s.mutate()
            s.mutate()
            s.evaluate_solution()
        Q = []
        if NovArchive: 
           self.NoveltyArchive = []

        chronic = numpy.zeros((len(obj_names),population_size*2, Nslice))
        solved = {}
        archive_array = np.zeros((self.grid_sz, self.grid_sz))
        ffa_archive = np.zeros(self.grid_sz)
        pp = 0 # counter to keep track of chronics that are saved
        stats_array = np.zeros((100,100))
        EvoBoosterFlag = False 
        measureEvoFlag = False
        #map initial generation into archive
        archive_array = eob.map_population_to_grid(P,
                                                   grid_sz =self.grid_sz,
                                                   grid=archive_array)
        ffa_archive = eob.map_pop_to_array_by_objective(
                                        P, self.grid_sz,
                                         FIT,ffa_archive)
        
        for i in range(num_generations):
            if i>1 and (i)%(Nslice)==0:# save chronic so that it does not get to big and i have it in case of freeze
               self.save_objectives(chronic,title,pp, solved, archive_array,stats_array)
               pp += 1
           
            pfunc = [p for p in P if np.all(p.behavior>=0)]
            qfunc = [p for p in Q if np.all(p.behavior>=0)]
            

            #FEEDBACK from children to compute EVO
            #progress is measured as number of individuals added to the elite per generation
            progress = 0
            #rewarding feedback
            for p in P:
                    if p.newInArchive:
                            p.newInArchive = False
                            progress += 1
                            self.augmentParentEvo(p.parentIDs,P,propagate=False)
            #punishing feedback
            progress = float(progress)/len(P)
            for p in P:
                    p.objs[VIAB] += p.childrenInQ * progress
                    p.objs[VIABP] += p.childrenInQ * progress
                    p.objs[PROGRESS] = progress
                    p.childrenInQ = 0
           
            print "Iteracao ", i,'psize: ', len(pfunc), ', qsize: ', len(qfunc) , ' fraction added to elite: ', progress, '%'

            R = []
            R.extend(P)
            R.extend(Q)

            solvers = 0
            NovAdded = False
            NovAddedThisGen = 0
            before = EvoBoosterFlag
            EvoBoosterFlag = (i %(2*EvoBoosterIntervall)) > EvoBoosterIntervall and i> EvoBoosterIntervall
            if before != EvoBoosterFlag or i == num_generations -1:
                    measureEvoFlag = True
                    print "measuring evolvability..."
            else:
                    measureEvoFlag = False

            # evaluate relative criteria (those defined with respect to the population): Rarities, Novelty, Diversity
            #and writes it in the list used for selection
            for s in R:
               #evaluates rarity (wrt to archive) and novelty (current pop and archive)
               s.evaluate2( R, archive_array,
                           self.NoveltyArchive,
                           ffa_archive,
                           probe_Evo = measureEvoFlag,
                           EvoMuts = probeEvoNmutants,
                           recordObj = recordObj,
                           probe_RARs = (EvoBoosterFlag or measureEvoFlag),
                           gammaGrid = self.gridGamma,
                          gammaLRAR = self.gammaLRAR,
                           shSOLSpan=self.shSOLSpan)
               s.objs[PROGRESS] = progress
               if s.solver:
                  solvers +=1
           
            #Novelty 
            # questions: archive is unique? no double entries?
            # sample from parents and children or children only?) if not np.any([np.all(c.behavior==a) for a  in NoveltyArchive])]
            if NovArchive:
                    Rfunc = [q for q in Q if np.all(q.behavior >=0)] + [p for p in P if np.all(p.behavior >=0)]
                    RfuncNew = [r for r in Rfunc if not np.any([np.all(r.behavior == a) for a in self.NoveltyArchive])]
                    self.NoveltyArchive += [c.behavior for c in  random.sample(RfuncNew,min(self.NovGamma,len(RfuncNew)))]
           


            #save the end location and fitness of individuals throughout iterations
            robs =R# Q if i>1 else P #first generation Q is empty
            for r in range(len(robs)):
               chronic[:,r,i%Nslice] = robs[r].objs
            if NovArchive:
               chronic[ARCHIVESIZE,:,i%Nslice] = len(self.NoveltyArchive)
            Nweirdos = np.sum( chronic[WEIRDO,:,i%Nslice], axis=0)
            #if Nweirdos> 0:
                    #print "this generation weirdos: ",Nweirdos

            #check if maze got solved            
            if solvers !=0:
               solved.update({i:solvers})
               if breakAfterSolved:
                  break

            #selection of most interesting trade-offs
            fronts = self.fast_nondominated_sort(R)
            
            del P[:]
            for front in fronts.values():
                if len(front) == 0:
                    break
                
                self.crowding_distance_assignment(front);
                P.extend(front)
                
                if len(P) >= population_size:
                    break
            
            self.sort_crowding(P)
            
            if len(P) > population_size:
                del P[population_size:]
                
            Q = self.make_new_pop(P)
            
            #refQ = [q for q in Q if all(q.behavior>0)]
            #print 'refQ', len(refQ)
            archive_array = eob.map_population_to_grid(Q, self.grid_sz, archive_array)
            ffa_archive = eob.map_pop_to_array_by_objective(
                                        Q, self.grid_sz,
                                         FIT,ffa_archive)
            ### visualization
            if visualization:
                viz = visuals.Vizzer(title)
                if NovArchive:
                     viz.render_robots_and_archive(self.NoveltyArchive, [P,Q], color=[(0,255,0),(255,0,0),(0,0,180)])
                else:
                     viz.render_robots( [P,Q], color=[(0,255,0),(255,0,0)])
        self.save_objectives(chronic[:,:,:i%Nslice],title,pp, solved, archive_array)

        if solved!={}:
           ret = solved.keys()[0]
        else:
           ret = -1
        return ret
        
    def augmentParentEvo(self,pIDs, P, propagate = False):
            '''
            augments the Evolvability proxy VIAB of the parent by one.
            Does the same for all grandparents still in the archive if propagete is set to TRUE
            '''
            for p in P:
                    if p.id in pIDs:
                            p.objs[VIAB] -= 1
                            if propagate:
                                    p.objs[VIABP] -= 1
                                    self.augmentParentEvo(p.parentIDs,P,propagate)

    def evaluate_pevo(self,pop):
        popgrid = np.zeros((self.grid_sz, self.grid_sz) )
        for rob in pop:
                popgrid += rob.grid
        for r in pop:
                    r.objs[PEVO] = entropyJ2.grid_contribution_to_population(r.grid,popgrid)

    def evaluate3(self,pop, NMutations =20):
       '''
       generates mutants for all individuals and 
       maps them plus their offsprings into a grid.
       Then computes an individuals contribution 
       to the overall spread of all mutants
       and writes that into the mazenaw objective list
       ATTENTION: NMUTANTS IS FIXCODED is it still?
       also computes evolvability in between:
       '''        
       mutants = []
       for r in pop:
          l = []
          for i in range(NMutations):
            mutant = r.robot.copy()
            mutant.mutate()
            mutant.map()
            l.append(mutant)
          #use mutants to calculate evolvability of individual
          l.append(r.robot)
          #robofamiliygrid = entropyJ.population_to_grids(l,self.grid_sz)
          #r.objs[3] = entropyJ.calc_evolvability_entropy(r,
          #                                         robofamiliygrid,
           #                                        self.grid_sz)
          mutants.append(l)
       flattened_mutants = []
       for ml in mutants:
          flattened_mutants.extend(ml)
       grid = entropyJ.map_robots_to_grids(flattened_mutants, self.grid_sz)
       for ri in range(len(pop)):
         superevo = 0
         for m in mutants[ri]:
            superevo -= entropyJ.calc_individual_entropy(grid,
                                                          m,
                                                          len(pop)*(1+NMutations),
                                                          self.grid_sz)
         pop[ri].objs[SEVO] = superevo
               
    def save_objectives(self,chronic,title,pp,solved=None, archive=None, stats=None):
        print "saving chronic of this run...." 
        numpy.save('./out/'+ title +'-' +str(pp) +'.npy', chronic)
        if archive != None: 
           numpy.save('./out/'+ title +'Archive' +'.npy', archive)
        if stats != None: 
           numpy.save('./out/'+ title +'Stats' +'.npy', stats)
        if solved != None:
           with open('./out/'+ title+'Solver.pkl','wb') as f:
              pickle.dump(solved,f)
        chronic = numpy.zeros(chronic.shape) #reset chronic
           
    def sort_ranking(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.rank > s2.rank:
                    P[j - 1] = s2
                    P[j] = s1
                    
    def sort_objective(self, P, obj_idx):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if s1.objectives[obj_idx] > s2.objectives[obj_idx]:
                    P[j - 1] = s2
                    P[j] = s1
                    
    def sort_crowding(self, P):
        for i in range(len(P) - 1, -1, -1):
            for j in range(1, i + 1):
                s1 = P[j - 1]
                s2 = P[j]
                
                if crowded_comparison(s1, s2) < 0:
                    P[j - 1] = s2
                    P[j] = s1
                
    def make_new_pop(self, P):
        '''
        Make new population Q, offspring of P. 
        '''
        Q = []
        
        while len(Q) != len(P):
            selected_solutions = [None, None]
            
            while selected_solutions[0] == selected_solutions[1]:
                for i in range(2):
                    s1 = random.choice(P)
                    s2 = s1
                    while s1 == s2:
                        s2 = random.choice(P)
                    
                    if crowded_comparison(s1, s2) > 0:
                        selected_solutions[i] = s1
                        
                    else:
                        selected_solutions[i] = s2
            
            if random.random() < self.crossover_rate: #happens always
                child_solution = selected_solutions[0].crossover(selected_solutions[1])
                
                if random.random() < self.mutation_rate: #currently set to 0.9
                    child_solution.mutate()
                    
                child_solution.evaluate_solution()
                
                Q.append(child_solution)
        
        return Q
        
    def fast_nondominated_sort(self, P):
        '''
        Discover Pareto fronts in P, based on non-domination criterion. 
        '''
        fronts = {}
        
        S = {}
        n = {}
        for s in P:
            S[s] = []
            n[s] = 0
            
        fronts[1] = []
        
        for p in P:
            for q in P:
                if p == q:
                    continue
                
                if p >> q:
                    S[p].append(q)
                
                elif p << q:
                    n[p] += 1
            
            if n[p] == 0:
                fronts[1].append(p)
        
        i = 1
        
        while len(fronts[i]) != 0:
            next_front = []
            
            for r in fronts[i]:
                for s in S[r]:
                    n[s] -= 1
                    if n[s] == 0:
                        next_front.append(s)
            
            i += 1
            fronts[i] = next_front
                    
        return fronts
        
    def crowding_distance_assignment(self, front):
        '''
        Assign a crowding distance for each solution in the front. 
        '''
        for p in front:
            p.distance = 0
        
        for obj_index in range(self.num_objectives):
            self.sort_objective(front, obj_index)
            
            front[0].distance = float('inf')
            front[len(front) - 1].distance = float('inf')
            
            for i in range(1, len(front) - 1):
                front[i].distance += (front[i + 1].distance - front[i - 1].distance)


"""
#novelty archive with threshold:
        NoNovAdded = 0
        lastNovAdded = 0
       for s in R:
          if -s.objs[5] > self.thresNovelty and all(s.behavior>=0): #if behavior is novel and not outside the maze
             if not (any([np.all(s.behavior == x) for x in NoveltyArchive]) and NoveltyArchive != []): #not yet in archive
                NoveltyArchive.append(s.behavior)
                NoNovAdded = True
                lastNovAdded = i
                NovAddedThisGen += 1
             
       #adjust threshold for Novelty Archive
       if i - lastNovAdded > 4:      #too hard
          self.thresNovelty *= 0.95  #decrease
       elif NovAddedThisGen > 4:     #too easy
          self.thresNovelty *= 1.05  # increase
       #print 'size of archive: ', len(NoveltyArchive)
"""
