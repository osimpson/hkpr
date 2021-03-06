from Network import *
import solver

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')

dolphins = Network(GRAPH_DATASETS['dolphins'])

subset = [6, 32, 41, 25, 9, 17, 26, 31, 54, 27, 13, 57, 60, 5, 48, 56, 7, 22, 19,
1]
vertex_boundary = dolphins.vertex_boundary(subset)
boundary_vec = np.array([[  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 73.20356034399658540224],
       [  0.                    ],
       [ 72.36707861296704891174],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  2.61707892868332514524],
       [  0.                    ],
       [  0.                    ],
       [ 67.56497031535104724753],
       [ 73.21537901342006193772],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ]])

b2 = solver.compute_b2(dolphins, boundary_vec, subset)
