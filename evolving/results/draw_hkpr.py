import sys
import matplotlib.pyplot as plt

hkpr_file = sys.argv[1]
fig_file = hkpr_file+'.png'

hkpr_vec = []
for l in open(hkpr_file, 'r').readlines():
    row = map(float, l.strip().split('\t'))
    hkpr_vec.append(row)

hkpr_vec.sort(key=lambda x: x[0])

plt.figure(figsize=(35,5))
plt.bar([x[0] for x in hkpr_vec], [x[1] for x in hkpr_vec], align='center')
plt.xticks([x[0] for x in hkpr_vec], size='small', rotation='vertical')
plt.savefig(fig_file, dpi=300)
