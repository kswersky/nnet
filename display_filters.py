from numpy import *
from scipy import *
from pylab import *
from matplotlib import *
import matplotlib.cm as cm
def print_aligned(w):
   n1 = int(ceil(sqrt(shape(w)[1])))
   n2 = n1
   r1 = int(sqrt(shape(w)[0]))
   r2 = r1
   Z = zeros(((r1+1)*n1, (r1+1)*n2), 'd')
   i1, i2 = 0, 0
   for i1 in range(n1):
       for i2 in range(n2):
           i = i1*n2+i2
           if i>=shape(w)[1]: break
           Z[(r1+1)*i1:(r1+1)*(i1+1)-1, (r2+1)*i2:(r2+1)*(i2+1)-1] = w[:,i].reshape(r1,r2)
   imshow(Z,cmap=cm.gray)
