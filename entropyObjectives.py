import numpy as np
import math
from collections import defaultdict
import mazepy
import random
import itertools
from fixedparams import *

#MAPPING POPULATION,ROBOTS,MAZENAVS TO GRIDS

def map_pop_to_archives(P,grid_sz,archives,HD=True, test=False):
        '''
        maps the population into 4 different kinds of archives in this order
        smartarchive
        pos_archive
        naive_archive
        HD_archive (can be disabled because it is so expensive in high dimensions)
                can also be array or dictionary
        '''
        for p in P:
                idxs = tuple([ int(x*grid_sz) for x in p.behaviorSamples])
                
                #smart_archive (all timesamples interpreted as positions into one grid)
                idxpairs = [(idxs[i],idxs[i+1]) for i in range(0,len(idxs),2)]
                for idxpair in idxpairs:
                        archives[0][idxpair]+=1 

                #pos_archives (each timeslot gets its own grid)
                idxpairs = [(idxs[i],idxs[i+1]) for i in range(0,len(idxs),2)]
                for idxpair,archive in zip(idxpairs, archives[1]):
                        archive[idxpair]+=1 

                #naive_archive ( all  timesamples get their own grid, no relation between x and y position)
                for idx,archive in zip(idxs, archives[2]):
                        archive[idx]+=1
                
                #HD_Archive; (all dimensions relate to each other)
                if HD:
                        if type(archives[3]) == dict:
                                if idxs in archives[3].keys():
                                        archives[3][idxs]+=1
                                else:
                                        archives[3].update({idxs:1})
                        else:
                                archives[3][idxs]+=1
                if test:
                        archives[4][idxs]+=1
        return archives

def map_mazenav_behavior(mazenav,grid_sz):
        '''
        discretizes the mazenav's behavior into different tuples of indeces
        HD (x1,y1,...xn,yn)
        pos ((x1,y1),...(xn,yn))
        naive ((x1),...(y1))
        '''
        idxs = tuple([ int(x*grid_sz) for x in mazenav.behaviorSamples])
        idxpairs = [(idxs[i],idxs[i+1]) for i in range(0,len(idxs),2)]
        return idxs, idxpairs, idxs

def map_into_grid(mazenav, grid_sz):
        """discretizes the robots behavior into a grid and writes the original values into its .behavior attribute.
        expects to be given a MazeSolution class, but can also be given a behavior
        ATTENTION: also accepts degenerate (negative) behavior
        @grid cells: returns a tuple with x and y index of grid
        """
        return map_robot_into_grid(mazenav.robot,grid_sz)

def map_robot_into_grid(robot, grid_sz):
        x=mazepy.feature_detector.endx(robot)
        y=mazepy.feature_detector.endy(robot)
        x_grid=int(x*grid_sz)
        y_grid=int(y*grid_sz)
        robot.behavior=np.array([x,y])
        return (x_grid,y_grid)

def map_population_to_grid(pop, grid_sz, grid=None):
        """
        maps a given list of mazenavs into a grid
        mazenavs with dummy (negative) behavior will not be mapped
        @grid:if no grid is given, an array will be created and returned
        otherwise the given grid is incremented
        """
        if grid == None:
                print "given grid was empty"
                grid = np.zeros((grid_sz, grid_sz))
        refPop = [p for p in pop if np.all(p.behavior>=0)]
        for k in refPop:
                key = map_into_grid(k, grid_sz)
                grid[key] += 1
        return grid

def map_behaviors_to_grid(Bs, grid_sz):
        grid = np.zeros((grid_sz, grid_sz))
        for b in Bs:
                grid[b] += 1
        return grid

def map_pop_to_array_by_objective(pop,array_sz,obj,grid=None):
        '''maps a lst of given mazenas into an array of given size according 
        to a given objective
        mazenavs with dummy (negative) behavior will not be mapped
        if no grid is given it will be created
        assumes obj to be normalized to [0,1] 
        '''
        if grid == None:
                print "creating grid"
                grid = np.zeros(array_sz)
        refPop = [p for p in pop if np.all(p.behavior>=0)]
        for k in refPop:
                key = int(k.objs[obj]*(array_sz))
                grid[key] += 1
        return grid

## ENTROPY CALCULATIONS 
def HD_entropy(grid, mazenav,grid_sz):
        if type(grid) == dict:
                return HD_entropyDic(grid,mazenav,grid_sz)
        else:
                return individual_entropy(grid, map_mazenav_behavior(mazenav,grid_sz)[0])

def HD_entropyDic(grid, mazenav,grid_sz):
        fsamp = np.sum(grid.values())
        key = map_mazenav_behavior(mazenav,grid_sz)[0]
        p = grid[key]/float(fsamp)
        entr = p * math.log(p)
        entr /= grid[key]
        assert entr <=0
        return -entr

def pos_entropy(grids, mazenav, grid_sz):
        behavs =  map_mazenav_behavior(mazenav,grid_sz)[1]
        entr=0
        for behav,grid in zip(behavs,grids):
                entr += individual_entropy(grid,behav)
        return entr

def naive_entropy(grids, mazenav, grid_sz):
        behavs =  map_mazenav_behavior(mazenav,grid_sz)[2]
        entr=0
        for behav,grid in zip(behavs,grids):
                entr += individual_entropy(grid,behav)
        return entr

def smart_entropy(grid, mazenav, grid_sz):
        behavs =  map_mazenav_behavior(mazenav,grid_sz)[1]
        entr=0
        for behav in behavs:
                entr += individual_entropy(grid,behav)
        return entr



def calc_individual_entropy(grid, mazenav, grid_sz):
        return individual_entropy(grid,map_into_grid(mazenav,grid_sz))

def individual_entropy(grid, behavior):
        '''
        s the entropy of an behavior within a given grid
        behavior is a tuple that must be within the size of the grid
        when there is no other behavior in the grid, return something psitive too: 0.01
        '''
        if np.sum(grid>0) ==1:
                return .01
        assert np.sum(grid)>0
        fsamp = np.sum(grid)
        try:
                p = grid[behavior]/fsamp
                entr = p*math.log(p) # the entropy of all the complete cell
                entr /= grid[behavior] # spread entropy across individuals in that cells 
                assert entr<=0
                return - entr
        except ValueError:
                print 'ValueError', grid
                print behavior
                return 0

def grid_entropy(grid):
        '''
        returns the entropy of the given grid
        if there is only one behavior in the grid then returns 0.01
        if there is no behavior in the grid return 0
        '''
        fsamp=np.sum(grid)
        if fsamp ==1:
                return .01
        if fsamp ==0:
                return 0 
        p = grid/fsamp
        ps = p[p>0]
        entr = np.sum(ps * np.log(ps))
        if entr > 0:
                print 'ERROR: gridentropy positive:', entr
                print grid
                return 100
                assert entr <=0
        return - entr

def grid_contribution_to_population(robgrid, popgrid):
        '''
        returns the contribution of the  individuals in the small grid
        to overall evolvability of the population
        '''
        xs, ys = np.where(robgrid>0)
        stepstones = [[(x,y)]*robgrid[x,y] for x,y in zip(xs,ys)]
        stepstones = list(itertools.chain.from_iterable(stepstones))
        entr = 0
        for s in stepstones:
                entr += individual_entropy(popgrid,s)
        return entr

def calc_FFA(ffa_archive, mazenav):
        key = int(mazenav.objs[FIT]*len(ffa_archive))
        return individual_entropy(ffa_archive, key)

def map_mutants_to_grid(mazenav,Nmuts, grid_sz):
        grids = np.zeros((grid_sz,grid_sz))
        for x in range(Nmuts):
                mutant = mazenav.robot.copy()
                mutant.mutate()
                mutant.map()
                key=map_robot_into_grid(mutant,grid_sz)
                if np.all(mutant.behavior>0):
                        grids[key]+=1
        key=map_into_grid(mazenav,grid_sz)
        grids[key] += 1
        return grids











