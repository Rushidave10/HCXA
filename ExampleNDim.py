import numpy as np
import seaborn as sns

from NDimCountMaximization import NCountMaximization

sns.set()

def reference_coverage(X):
    # for uniform distribution on a given shape the value range of the reference coverage is not important
    x0 = X[0]
    x1 = X[1]
    _sum = x0 ** 2 + x1 ** 2
    return np.less(_sum, 1)


oracle = NCountMaximization(box_constraints=[[-1, 1],
                                             [-1, 1],
                                             ],
                            n_bins=[5, 4],
                            render_online=True,
                            render_options=dict(annot=True),
                            )

x = np.array([[0, 0],
              ])

num_points = 100
oracle.update_data(x)
for _ in range(num_points):
    point = oracle.sample_optimally()
    oracle.update_data(point)
oracle.render()
