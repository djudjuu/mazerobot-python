import pygame
from pygame.locals import *
import mazepy
SZX=SZY=400
global screen, background

class Vizzer():
   def __init__(self, title='Viz'):
      self.screen =None
      pygame.init()
      pygame.display.set_caption(title)
      self.screen = pygame.display.set_mode((SZX,SZY))
    
      self.background = pygame.Surface(self.screen.get_size())
      self.background = self.background.convert()
      self.background.fill((250, 250, 250))
   
   def render_robots_and_archive(self, archive, pops, color = [(255,0,0)]):
      self.screen.blit(self.background, (0, 0))
      i = 0
      for pop in pops:
       for s in pop:
        if all(s.behavior>0):
           x=mazepy.feature_detector.endx(s.robot)*SZX #feature detector is handling mazenav classes
           y=mazepy.feature_detector.endy(s.robot)*SZY
           rect=(int(x),int(y),5,5)
           pygame.draw.rect(self.screen,color[i],rect,0)
       i+=1
      for b in archive:
         rect=(int(b[0]*SZX),int(b[1]*SZY),3,3)
         pygame.draw.rect(self.screen, color[i],rect,0)
      pygame.display.flip()

   def render_robots(self,pops, color = [(255,0,0)]):
      """
      becomes a population of mazeSolution classes
      and renders them on the map
      """
      #global screen,background
      self.screen.blit(self.background, (0, 0))
      i = 0
      sizes=(5,3)
      for pop,sz in zip(pops,sizes):
       for s in pop:
        if all(s.behavior>0):
           x=mazepy.feature_detector.endx(s.robot)*SZX #feature detector is handling mazenav classes
           y=mazepy.feature_detector.endy(s.robot)*SZY
           rect=(int(x),int(y),sz,sz)
           pygame.draw.rect(self.screen,color[i],rect,0)
       i+=1
      pygame.display.flip()
