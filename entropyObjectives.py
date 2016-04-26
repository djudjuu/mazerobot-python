import numpy as np
import math
from collections import defaultdict
import mazepy
import random
import itertools
from fixedparams import *

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

def calc_individual_entropy(grid, mazenav, grid_sz):
        return individual_entropy(grid,map_into_grid(mazenav,grid_sz))

def individual_entropy(grid, behavior):
        '''
        s the entropy of an behavior within a given grid
        behavior is a tuple that must be within the size of the grid
        '''
        assert np.sum(grid)>0
        fsamp = np.sum(grid)
        try:
                p = grid[behavior]/fsamp
                entr = p*math.log(p) # the entropy of all the complete cell
                entr /= grid[behavior] # spread entropy across individuals in that cells 
                assert entr<=0
                return - entr
        except ValueError:
                print grid
                print behavior
                return 0

def grid_entropy(grid):
        '''
        returns the entropy of the given grid
        '''
        fsamp=np.sum(grid)
        p = grid/fsamp
        ps = p[p>0]
        entr = np.sum(ps * np.log(ps))
        assert entr <= 0
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











