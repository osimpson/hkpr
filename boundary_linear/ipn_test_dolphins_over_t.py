import sys

execfile('ipn_dolphins_example.py')

eps, gamma = sys.argv[1:]
eps = float(eps)
gamma = float(gamma)
s = len(subset)
T = (s**3)*(np.log((s**3)*(1./gamma)))
N = T/gamma

ts = np.logspace(0, np.log10(T), base=10, num=20)

err_unit = []
err_2norm = []
err_1norm = []

for t in ts:
    print
    print t
    print 'computing true'
    true_t = dolphins.exp_hkpr(subset, t, b2)
    print 'computing appr'
    appr_t = solver.approx_hkpr_mp(dolphins, subset, t, b2, eps, verbose=True)
    err_unit.append(solver.approx_hkpr_err_unit(true_t, appr_t, eps))
    err_2norm.append(solver.approx_hkpr_err(true_t, appr_t, b2, eps))
    err_1norm.append(solver.approx_hkpr_err_1norm(true_t, appr_t, b2, eps))

print 'err_unit'
print err_unit
print '\nerr_2norm'
print err_2norm
print '\nerr_1norm'
print err_1norm

f = open('dolphins_err_unit'+str(eps)+str(gamma)+'.txt', 'w')
f.write(err_unit)
f.close()
f = open('dolphins_err_2norm'+str(eps)+str(gamma)+'.txt', 'w')
f.write(err_2norm)
f.close()
f = open('dolphins_err_1norm'+str(eps)+str(gamma)+'.txt', 'w')
f.write(err_1norm)
f.close()
