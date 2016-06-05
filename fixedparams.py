FIT=0    #plain fitness
CUR=1    # path entropy
NOV=2    # Novelty as mean distance to k nearest neighbours
EVO=3    # evolvability as entropy over xxx extra created offspring 
PEVO=4   #evo as an individuals contribution to the diversity of the stepping stones accumulated by the population
RAR=5    # Rarity wrt to accumulated history of behavior
SEVO=6   #one-step lookahead contribution to the diversity of the next generatio
DIV=7    # Diversity as mean distance to all the population
FFA=8   #frequency fitness assignment with respect to accumulated history of fitness values
IRAR=10  #individual rarity (how different is the last behavior from the last lineage?
LRAR=11 #Lineage rarity: discounted rarity is herited from parent
ARCHIVESIZE = 12 
WEIRDO = 13     # True if was disabled after hitting the wall
XEND=14
YEND=15
REVO= 16        #Useful (rare) evolvability, measures the capability to produce offsprings that do something new
VIAB= 17        # also evolvability in the 2nd sense. Measures how many offsprings have been taken into the archive, selected periodically
PROGRESS = 18   #how many individuals have come into the elite this generations
ZERO=19         # PLACEHOLDER
VIABP= 20        # VIAB selected for permanently
LGE=9          # evolvability as entropy over personal stepping stones, distribution over complete history, need to be apart
LGEr=21       #entropy of the recent behaviors, need to be farther apart
shLGE=22
LGD = 23      # how many different behaviors in complete history
LGDr = 24      # how many different behaviors in complete history, need to be apart
LGDnd=25      #just counts how many different recent behaviors, ignores distribution
shLGD=26     #how many different recent behacviors, those need to be farther apart
shLGDnd=27     #how many different recent behacviors, those need to be farther apart
evoMeasured =28    #1 if evo measured, 0 elsewise

obj_names =  ['FIT', 'CUR', 'NOV', 'EVO','PEVO','RAR', 'SEVO', 'DIV','FFA','LGE', 'IRAR','LRAR','ARCHIVESIZE','WEIRDO','XEND', 'YEND','REVO','VIAB','PROGRESS','ZERO', 'VIABP','LGEr','shLGE','LGD','LGDr','LGDnd','shLGD','shLGDnd','evoMeasured']

def get_obj_ID(string):
        return obj_names.index(string)
