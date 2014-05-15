import math
import sys

t = float(sys.argv[1])
eps = float(sys.argv[2])

num = t*(2+math.log(t)) + math.log(2/eps)
den = math.log(num)

print num/den
