from vmad import operator
import numpy

@operator
class codot:
   ain = {'x1' : 'ndarray', 'x2' : 'ndarray'}
   aout = {'y' : 'ndarray'}

   def apl(self, x, axis):
       return 
