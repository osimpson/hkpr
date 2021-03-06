from Network import *
import solver

execfile('/home/olivia/UCSD/projects/datasets/datasets.py')

polbooks = Network(GRAPH_DATASETS['polbooks'])

subset_pb = [8,
 47,
 40,
 23,
 54,
 13,
 53,
 32,
 25,
 11,
 38,
 43,
 39,
 35,
 56,
 12,
 33,
 26,
 102,
 21,
 3]

vertex_boundary_pb = polbooks.vertex_boundary(subset_pb)

boundary_vec_pb = np.array([[ 38.09847577253854922219],
       [ 16.58582358461767114477],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  3.2698134520221469046 ],
       [  7.30706711966300748173],
       [  0.                    ],
       [  0.                    ],
       [ 16.59252695571260005636],
       [ 10.86453930143835933109],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 41.11745732935456487667],
       [  9.78489172426849052044],
       [ 24.95095609638604372549],
       [ 35.1815914201047945653 ],
       [  6.1277634688402198293 ],
       [ 26.33677616276978028509],
       [ 43.18173032208536454846],
       [  0.                    ],
       [ 33.3286979105802387835 ],
       [  0.                    ],
       [  1.65675170761830359289],
       [  0.                    ],
       [  0.                    ],
       [ 18.54861000542786797496],
       [  0.                    ],
       [ 27.12769932103373804466],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 14.58034721384019505308],
       [  0.                    ],
       [ 25.83987172025535272724],
       [ 20.4358207791302994849 ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 25.17086317363338920927],
       [ 34.44929686682000635756],
       [  0.                    ],
       [  1.50311592145290773281],
       [ 10.88412066312254289357],
       [  8.45011285703157177807],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 26.80603845867506862533],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [ 44.96682701245633495546],
       [  0.                    ],
       [ 42.7842995781201977934 ],
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
       [  1.64797871976130050342],
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
       [ 48.35793277951719915109],
       [ 34.11471924060946747659],
       [ 12.02082000227010993854],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ],
       [  0.                    ]])


b2_pb = solver.compute_b2(polbooks, boundary_vec_pb, subset_pb)
