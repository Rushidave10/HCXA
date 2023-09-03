import numpy as np
import seaborn as sns
import datetime
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from typing import Union

# sns.set()


class NCountMaximization:
    def __init__(self,
                 box_constraints,
                 n_bins: Union[list, int],
                 render_online=False,
                 render_options=None,
                 reference_pdf=None,
                 state_names=None,
                 ):

        self.range = box_constraints  # Range in each Dimension.
        self.dim = len(box_constraints)  # Total dimension of Sample Space.
        self.edges = None  # Start and Stop of each bin per Dimension.
        self.lower_bound = np.array(box_constraints)[:, 0]  # Lower bound in each Dimension.
        self.upper_bound = np.array(box_constraints)[:, -1]  # Upper bound in each Dimension.
        self.n_bins = n_bins  # Number of bins in each Dimension. It can be int or list.

        # If n_bins is int, then bins in each dimension are same. e.g., n_bins = 5 for 3D search space would mean
        # 5 bins per dimension.

        # If n_bins is list, then bins in each dimension cna be set individually. The order of dimension should be
        # monotonically decreasing. e.g., n_bins = [10, 5, 2] would lead to 10 bins in 1st dimension and 5 bins in
        # second and so on.

        self.render_online = render_online  # To activate live plot.

        self.coverage_data = np.empty((0, self.dim)) * np.nan  # Capture the points already visited.
        self.count_matrix = None  # Capture the number of points in each bin per dimension.

        if render_options is None:
            self.annot = True  # To display number of points in each bin in the live plot.
        else:
            self.annot = render_options['annot']

        if state_names is None:
            self.state_names = [f"$x_{i}$" for i in range(self.dim)]  # Name of variables in each dimension.
        else:
            self.state_names = state_names

        if reference_pdf is None:
            # If not specified sample points in order to achieve uniform distribution in entire space.
            _state_space_volume = np.product([_con[-1] - _con[0] for _con in box_constraints])  #

            def uniform_pdf(X):
                return 1 / _state_space_volume

            self.reference_pdf = uniform_pdf
        else:
            self.reference_pdf = reference_pdf  # User-defined probability distribution function.

        self.file_path = None

    def update_data(self, point: np.ndarray):
        """

        :param point: The point which is to be added to the coverage_data.
        The point could be a single point or a set of points. Depending on size of point either increment of single bin
        count is done or np.histogramdd() is called to recalculate the whole count matrix (for efficiency).
        """
        self.coverage_data = np.vstack((self.coverage_data, point))

        if self.count_matrix is None:
            self.count_matrix, self.edges = np.histogramdd(self.coverage_data,
                                                           range=self.range,
                                                           bins=self.n_bins)

        else:
            if point.shape[0] > 1:
                self.count_matrix = np.histogramdd(self.coverage_data,
                                                   range=self.range,
                                                   bins=self.n_bins)[0]

            else:
                _list = []
                for idx, edge in enumerate(self.edges):
                    _list.append(np.digitize(point[idx], edge) - 1)
                _tuple = tuple(_list)
                np.add.at(self.count_matrix, _tuple, 1)

        if self.render_online:
            self.render_live(point, annot=self.annot)

    def sample_optimally(self):
        """
        Using the count at each index, the bins are sorted and then index with the lowest count satisfying the reference
        pdf is chosen as sampling bin. Inside the chosen bin, point is sampled uniformly without knowledge of other
        points in the bin.
        """
        _sorted_idx = np.argsort(np.ravel(self.count_matrix, order="C"), axis=None, kind='mergesort')
        _sorted = np.asarray(np.unravel_index(_sorted_idx, shape=self.count_matrix.shape)).T

        for j in range(_sorted.shape[0]):
            _idx = _sorted[j]
            _low = [self.edges[i][_idx[i]] for i in range(self.dim)]
            _high = [self.edges[i][_idx[i] + 1] for i in range(self.dim)]
            _sample = np.random.uniform(low=_low,
                                        high=_high,
                                        )

            if self.reference_pdf(_sample):
                return _sample

    def render(self, query_point=None, annot=False):
        """
        :param query_point: Enter the point for which you would like to know the count.
        :param annot: Annotate all the bins.
        Plot and save a figure showing the distribution of points in each dimension.
        """
        # if not hasattr(self, 'scatter_fig'):
        # matplotlib.use('Agg')
        self.scatter_fig, self.scatter_axes = plt.subplots(self.dim, self.dim)
        self.scatter_axes = np.reshape(self.scatter_axes, (self.dim, self.dim))

        for i in reversed(range(self.dim)):
            for j in reversed(range(self.dim)):
                _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                _i_margin = (self.upper_bound[i] - self.lower_bound[i]) * 0.1

                if j < i:
                    self.scatter_axes[i, j].set_xlim([self.lower_bound[j] - _j_margin,
                                                      self.upper_bound[j] + _j_margin])
                    self.scatter_axes[i, j].set_ylim([self.lower_bound[i] - _i_margin,
                                                      self.upper_bound[i] + _i_margin])

                    if i == self.dim - 1:
                        self.scatter_axes[i, j].set_xlabel(self.state_names[j])
                    if j == 0:
                        self.scatter_axes[i, j].set_ylabel(self.state_names[i])

                elif i == j:
                    self.scatter_axes[i, j].set_ylabel(self.state_names[i])
                    self.scatter_axes[i, j].set(yticklabels=[])

                elif j > i:
                    pass

        for i in range(self.dim):
            for j in range(self.dim):
                _axes = tuple(ax for ax in range(self.dim) if ax not in (i, j))
                if j < i:
                    if query_point is not None:
                        _data = np.vstack((self.coverage_data, query_point))
                        colors = ['darkblue'] * len(self.coverage_data) + ['orange']
                        sns.scatterplot(x=_data[:, j], y=_data[:, i], ax=self.scatter_axes[i, j], color=colors)
                    else:
                        sns.scatterplot(x=self.coverage_data[:, j], y=self.coverage_data[:, i],
                                        ax=self.scatter_axes[i, j])
                        sns.rugplot(x=self.coverage_data[:, 0], y=self.coverage_data[:, 1], ax=self.scatter_axes[i, j])

                elif j == i:

                    _hist_2d = np.sum(self.count_matrix, axis=_axes)
                    sns.histplot(self.coverage_data[:, i], binrange=self.range[i], bins=self.n_bins[i],
                                 ax=self.scatter_axes[i, j],
                                 # kde=True,
                                 )


                else:

                    _hist_2d = np.sum(self.count_matrix, axis=_axes)
                    sns.heatmap(_hist_2d.T, annot=annot, ax=self.scatter_axes[i, j], cmap='Blues',
                                cbar=False).invert_yaxis()

        # fm = plt.get_current_fig_manager()
        # fm.window.showMaximized()
        plt.show()
        folder_name = 'plots'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        plt.savefig(folder_name + f'/Case1.pdf', dpi=300)

    def render_live(self, query_point=None, annot=False):
        if not hasattr(self, 'scatter_fig'):
            self.scatter_fig, self.scatter_axes = plt.subplots(self.dim, self.dim)
            self.scatter_axes = np.reshape(self.scatter_axes, (self.dim, self.dim))

            for i in reversed(range(self.dim)):
                for j in reversed(range(self.dim)):
                    _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                    _i_margin = (self.upper_bound[i] - self.lower_bound[i]) * 0.1

                    if j < i:
                        self.scatter_axes[i, j].set_xlim([self.lower_bound[j] - _j_margin,
                                                          self.upper_bound[j] + _j_margin])
                        self.scatter_axes[i, j].set_ylim([self.lower_bound[i] - _i_margin,
                                                          self.upper_bound[i] + _i_margin])

                        if i == self.dim - 1:
                            self.scatter_axes[i, j].set_xlabel(self.state_names[j])
                        if j == 0:
                            self.scatter_axes[i, j].set_ylabel(self.state_names[i])

                    elif i == j:
                        self.scatter_axes[i, j].set_ylabel(self.state_names[i])
                        self.scatter_axes[i, j].set(yticklabels=[])

                    elif j > i:
                        pass

        for i in range(self.dim):
            for j in range(self.dim):
                _axes = tuple(ax for ax in range(self.dim) if ax not in (i, j))
                if j < i:
                    self.scatter_axes[i, j].cla()
                    _j_margin = (self.upper_bound[j] - self.lower_bound[j]) * 0.1
                    _i_margin = (self.upper_bound[i] - self.lower_bound[i]) * 0.1
                    self.scatter_axes[i, j].set_xlim([self.lower_bound[j] - _j_margin,
                                                      self.upper_bound[j] + _j_margin])
                    self.scatter_axes[i, j].set_ylim([self.lower_bound[i] - _i_margin,
                                                      self.upper_bound[i] + _i_margin])
                    if i == self.dim - 1:
                        self.scatter_axes[i, j].set_xlabel(self.state_names[j])
                    if j == 0:
                        self.scatter_axes[i, j].set_ylabel(self.state_names[i])

                    if query_point is not None:
                        _data = np.vstack((self.coverage_data, query_point))
                        colors = ['darkblue'] * len(self.coverage_data) + ['orange']
                        sns.scatterplot(x=_data[:, j], y=_data[:, i], ax=self.scatter_axes[i, j], color=colors)
                    else:
                        sns.scatterplot(x=self.coverage_data[:, j], y=self.coverage_data[:, i],
                                        ax=self.scatter_axes[i, j])
                        sns.rugplot(x=self.coverage_data[:, 0], y=self.coverage_data[:, 1], ax=self.scatter_axes[i, j])

                elif j == i:

                    self.scatter_axes[i, j].cla()
                    self.scatter_axes[i, j].set_ylabel(self.state_names[i])
                    _hist_2d = np.sum(self.count_matrix, axis=_axes)
                    sns.histplot(self.coverage_data[:, i], binrange=self.range[i], bins=self.n_bins[i],
                                 ax=self.scatter_axes[i, j],
                                 # kde=True,
                                 )
                    self.scatter_axes[i, j].set(yticklabels=[])

                else:
                    self.scatter_axes[i, j].cla()
                    _hist_2d = np.sum(self.count_matrix, axis=_axes).T
                    sns.heatmap(_hist_2d, annot=True, ax=self.scatter_axes[i, j], cmap='Blues',
                                cbar=False).invert_yaxis()

        # fm = plt.get_current_fig_manager()
        # fm.window.showMaximized()
        plt.pause(.1)
