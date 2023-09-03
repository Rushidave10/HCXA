import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from NDimCountMaximization import NCountMaximization
import time

sns.set()

def reference_coverage(X):
    # for uniform distribution on a given shape the value range of the reference coverage is not important
    x0 = X[0]
    x1 = X[1]
    # x2 = X[2]
    # x3 = X[3]
    # x4 = X[4]
    # x5 = X[5]
    _sum = x0 ** 2 + x1 ** 2
    return np.less(_sum, 1)


oracle = NCountMaximization(box_constraints=[[-1, 1],
                                             [-1, 1],
                                             # [-1, 1],
                                             # [-1, 1],
                                             # [-1, 1],
                                             # [0, 0],
                                             ],
                            n_bins=[5, 4],
                            render_online=True,
                            render_options=dict(annot=True),
                            reference_pdf=reference_coverage,
                            # state_names=['d', 'q', 'z']
                            )

x = np.array([[0, 0],
              # [0, 1, -1, 1],
              # [1, 0, 1, .5],
              # [0, 1, 0, -.5],
              # [1, 1, 0, 1],
              # [-1, -1, -1, .3],
              # [-1, 0, -1, .8],
              # [0, -1, 0, -1],
              ])

num_points = 100
oracle.update_data(x)
for _ in range(num_points):
    point = oracle.sample_optimally()
    oracle.update_data(point)
    # if _ % 10 == 0:
    #     oracle.render()
oracle.render()
