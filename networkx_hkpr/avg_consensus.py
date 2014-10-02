import math
import numpy as np
import hkpr

def avg_consensus_explicit(net, x_vec):
    num = sum([net.graph.degree(n)*x_vec[net.node_to_index[n]] for n in net.graph.nodes()])
    den = sum([net.graph.degree(n) for n in net.graph.nodes()])
    return float(num/den)

def avg_consensus(net, x_vec, t_force=None, eps=None):
    '''
    Computes the weighted-average consensus for a flock    
    input:
        net, hkpr.Network
        x_vec, a vector of initial vertex states
        eps, error parameter
    '''

    def fiedler_val(net):
        '''
        Compute the second smallest eigenvalue of the Laplace operator
        '''
        laplace = np.eye(net.size) - net.walk_mat #laplace operator
        w, v = np.linalg.eig(laplace) #compute the eigenvalues and eigenvectors
        fied = sorted(w)[1] #second smallest eigenvalue
        return fied
    
    if t_force is not None:
        t = t_force
    else:
        lam = fiedler_val(net)
        t = 1.0/lam
    f = np.array(x_vec).dot(net.deg_mat)
    hkpr = net.exp_hkpr(t, seed_vec=f)
    cons = hkpr.dot(np.linalg.inv(net.deg_mat))
    return t, cons

def disagreement(net, xinitial, xt):
    cons_val = avg_consensus_explicit(net, xinitial)
    delta = xt - cons_val #disagreement vector
    #return norm of disagreement vector
    return math.sqrt(sum([x**2 for x in delta]))

dolph = hkpr.Network(gml_file='dolphins.gml')

f = open('consensustrials.txt','wb')
f.write('=============== New Trial ===============\n')
x_e = np.random.rand(dolph.size)
true = avg_consensus_explicit(dolph, x_e)
f.write('\ntrue avg consensus: '+str(true)+'\n')
t, comp = avg_consensus(dolph, x_e)
disg = disagreement(dolph, x_e, comp)
f.write('computed value t: '+str(t)+'\n\n')
f.write('#t\t\tdisagreement\n\n')
f.write(str(t)+'\t'+str(disg))
t = 0
for i in range(200):
    t = t+1
    cons = avg_consensus(dolph, x_e, t_force=t)[1]
    disg = disagreement(dolph, x_e, cons)
    f.write('\n'+str(t)+'\t'+str(disg))
f.close()

#for i in range(5):
#    f.write('=============== New Trial ===============\n')
#    x_e = np.random.rand(dolph.size)
#    t, comp = avg_consensus(dolph, x_e)
#    f.write('computed avg consensus'+'\n'+str(comp)+'\n')
#    f.write('computed value t:'+str(t)+'\n')
#    for i in range(5):
#        t = t+50
#        cons = avg_consensus(dolph, x_e, t_force=t)[1]
#        f.write('\nnew t:'+str(t)+'\n')
#        f.write(str(cons))
#    true = avg_consensus_explicit(dolph, x_e)
#    f.write('\ntrue avg consensus'+'\n'+str(true)+'\n\n')
#f.close()
