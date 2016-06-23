import numpy as np
import math
from collections import defaultdict
import mazepy
import random
import itertools
from fixedparams import *


#MAPPING BIPEDS TO DYNAMIC ARCHIVES

def map_pop_to_dyn_archives(P,archives,bin_sz, bdim=30):
        '''
        currently uses a 2dim archive for every  2 dimensions 
        of the behavior vector
        '''
        for p in P:
                keys = map_behaviors_to_key(p, bin_sz,bdim)[1]
                for key,archive in zip(keys,archives):
                        if key in archive.keys():
                                archive[key] += 1
                        else: 
                                archive.update({key:1})
        return archives

def map_behaviors_to_key(r, bin_sz, bdim=30, scaled=False):
        '''returns to sets of keys:
                i) all dimensions independantly
                ii) tupels of xy pairs
        '''
        if type(r) == type(np.zeros(3)):
                behav = r
        else: 
                behav = [r.get_data_at_x(i) for i in range(bdim)]
        single_keys = [ int(b/bin_sz) for b in behav]
        keypairs = [(single_keys[i*2],single_keys[(i*2)+1]) 
                    for i in range(len(single_keys)/2)]
        return single_keys, keypairs

##### BIPED ENTROPIES #####

def dyn_pos_entropy(r,archives, bin_sz, bdim=30):
        '''
        currently returns the tRAR
        '''
        keys = map_behaviors_to_key(r,bin_sz,bdim)[1]
        entr = 0
        for key,archive in zip(keys,archives):
                entr += individual_entropy(archive,key)
        return entr

def path_entropy(biped, bin_sz):
        keys = map_behaviors_to_key(biped,bin_sz,biped.BEHAV_DIM)[1]
        counts = np.asarray([float(keys.count(k)) for k in set(keys)])
        ret =  grid_entropy(counts)
        #print ret
        assert ret >= -0
        return ret

def entropy_diff_path(biped, prev_bhv,bin_sz, maze=False):
        '''
        measures how much the rarity of the steps taken on the current path
        differs from the one  previous ones
        '''
        #catch the first iteration
        if np.sum(prev_bhv)==0:
                return 0
        keysNew = 0
        keysOld = 0
        if maze:
                keysNew = map_mazenav_behavior(biped, biped.grid_sz)[1]
                keysOld = map_mazenav_behavior(prev_bhv, biped.grid_sz)[1]
        else:
                keysNew = map_behaviors_to_key(biped,bin_sz,
                                               biped.BEHAV_DIM)[1]
                keysOld = map_behaviors_to_key(prev_bhv,bin_sz,
                                               biped.BEHAV_DIM)[1]
        entr = 0
        fsamp = biped.BEHAV_DIM
        #all new behaviors
        for key in set(keysNew):
                pN = float(keysNew.count(key)) / fsamp
                entrN = - pN * math.log(pN)
                pO = float(keysOld.count(key)) / fsamp
                entrO = 0
                if pO != 0:
                        entrO = - pO * math.log(pO)
                entr += entrN - entrO
        #anyu old baehaviors missing?
        for key in (set(keysOld) - set(keysNew)):
                pO = float(keysOld.count(key)) / fsamp
                entrO = - pO * math.log(pO)
                entr += 0 - entrO
        #jKprint entr
        return  entr


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
        can also directly be given an array of  positions (need to be in [0,1])
        HD (x1,y1,...xn,yn)
        pos ((x1,y1),...(xn,yn))
        naive ((x1),...(y1))
        '''
        behavs=0
        if type(mazenav) == type(np.zeros(2)):
                behavs=mazenav
        else:
                behavs = mazenav.behaviorSamples
        idxs = tuple([ int(x*grid_sz) for x in behavs])
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

def map_pop_to_array_by_objective(pop,array_sz,obj,grid=None,scale=None ):
        '''maps a lst of given mazenas into an array of given size according 
        to a given objective
        mazenavs with dummy (negative) behavior will not be mapped
        if no grid is given it will be created
        assumes obj to be normalized to [0,1] 
        but if scale is not None, then it will be scaled, with scope = (min,max)
        '''
        if grid == None:
                print "creating grid"
                grid = np.zeros(array_sz)
        refPop = [p for p in pop if np.all(p.behavior>=0)]
        for k in refPop:
                if scale!=None:
                        key=scale_obj(k.objs[obj],scale[0],scale[1])
                        key =int(key*array_sz)
                else:
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
        fsamp =0
        if type(grid)==dict:
                if np.sum(grid.values())==1:
                        print 'WARNING: only one behavior in grid'
                        return .01
                fsamp = np.sum(grid.values())
        else:
                if np.sum(grid>0) ==1:
                        print 'WARNING: only one behavior in grid'
                        return .01
                fsamp = np.sum(grid)
        assert fsamp>0
        try:
                p = grid[behavior]/float(fsamp)
                entr = p*math.log(p) # the entropy of all the complete cell
                entr /= grid[behavior] # spread entropy across individuals in that cells 
                assert entr<=0
                return - entr
        except ValueError:
                print 'ValueError', grid[behavior]
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
                print 'no behav in grid'
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

def frequency_of_objective(mazenav, obj_archive, obj_idx,mino, maxo):
        '''
        returns the entropy of the mazenav for the given objective
        expects an archive that accumulated the behavior over time
        '''
        o = scale_obj(mazenav.objs[obj_idx], mino,maxo)
        key = int(o*len(obj_archive))
        return individual_entropy(obj_archive, key)

def scale_obj(o, mino, maxo):
        return (o-mino)/(maxo-mino)







