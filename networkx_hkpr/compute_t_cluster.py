import math
import sys

size = float(sys.argv[1])
sig = float(sys.argv[2])
phi = float(sys.argv[3])
eps = float(sys.argv[4])

t = 1/phi*math.log(2*math.sqrt(sig)/(1-eps) + 2*eps*size)

print t
