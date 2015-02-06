execfile('Network.py')

eps = 0.1                               
target_cheeg = 0.05
target_size = 100
target_vol = 20*target_size
T = (1./target_cheeg)*np.log( (2*np.sqrt(target_vol))/(1-eps) + 2*eps*target_size )


print 'building gowalla network...'
gowalla = Network(edge_list=GRAPH_DATASETS['gowalla'])
sn_go = '307'
try:
    print 'computing heat kernel...'
    hkpr_go = gowalla.exp_hkpr(T, sn_go)
    f = open('gowalla_exp_hkpr_95.6466_307.txt', 'w')
    for n in hkpr_go:
        f.write(str(hkpr_go[n])+'\n')
    f.close()
except:
    print 'exception...'


print 'building dblp network...'
dblp = Network(edge_list=GRAPH_DATASETS['dblp'])
sn_db = '38868'
try:
    print 'computing heat kernel...'
    hkpr_db = dblp.exp_hkpr(T, sn_db)
    f = open('dblp_exp_hkpr_95.6466_38868.txt', 'w')
    for n in hkpr_db:
        f.write(str(hkpr_db[n])+'\n')
    f.close()
except:
    print 'exception...'


print 'building web network...'
web = Network(edge_list=GRAPH_DATASETS['webUND'])
sn_web = '12129'
try:
    print 'computing heat kernel...'
    hkpr_web = web.exp_hkpr(T, sn_web)
    f = open('webUND_exp_hkpr_95.6466_12129.txt', 'w')
    for n in hkpr_web:
        f.write(str(hkpr_web[n])+'\n')
    f.close()
except:
    print 'exception...'
