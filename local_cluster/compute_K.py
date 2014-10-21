import math

def compute_K_noT(eps):
    num = math.log(1/eps)
    den = math.log(num)
    return num/den

def compute_K(t, eps):
    num = t*(2+math.log(t)) + math.log(2/eps)
    den = math.log(num)
    return num/den

def compute_K_new(t, eps):
    num = t*math.log(t) + math.log(2/eps)
    den = math.log(num)
    return num/den
