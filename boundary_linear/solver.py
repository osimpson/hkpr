import networkx as nx
import numpy as np
from scipy.integrate import quadrature
import math
import random
import hkpr

def laplacian_solver(G, b, eps, approx=True):
    n = G.size
    D = G.deg_mat
    b1 = np.dot(np.transpose(b),D)
    T = (n**3)*math.log((n**3)*(1./eps))
    # compute exact solution
    if approx is False:
        hkpr = lambda t: G.exp_hkpr(t, seed_vec=b1)
        return quadrature(hkpr, 0, T)
    # compute approximate with Riemann sum
    else:
        x = np.zeros((1,n))
        N = int(T/eps)
        r = int(1./(eps**2)*(math.log(n) + math.log(1./eps)))
        for i in range(r):
            j = random.randint(1,N)
            xi = G.exp_hkpr(j*(T/N), seed_vec=np.transpose(b1))
            x += xi
        return np.dot(x/r, np.linalg.inv(D))
