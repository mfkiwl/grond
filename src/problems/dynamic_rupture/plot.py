import numpy as num
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from pyrocko.gf import PseudoDynamicRupture

from pyrocko.guts import Tuple, Float
from pyrocko.plot import mpl_init

from grond.plot.config import PlotConfig
from grond.plot.collection import PlotItem

km = 1e3


def km_fmt(x, p):
    return '%.2f' % (x / km)


class DynamicRuptureSlipMap(PlotConfig):
    '''
    Slip map of best solution.
    '''
    name = 'rupture_slip_map'
    size_cm = Tuple.T(2, Float.T(), default=(20., 20.))

    def make(self, environ):
        environ.setup_modelling()
        cm = environ.get_plot_collection_manager()
        history = environ.get_history(subset='harvest')
        problem = environ.get_problem()

        mpl_init(fontsize=self.font_size)
        cm.create_group_mpl(
            self,
            self.draw_figures(history, problem),
            title=u'Slip Map',
            section='solution',
            feather_icon='grid',  # alternatively: wind
            description=u'''
Slip distribution and rake of the finite slip solution. The absolute slip
is color coded. The black star marks the nucleation point.
''')

    def draw_figures(self, history, problem):
        source = history.get_best_source()

        store_ids = problem.get_gf_store_ids()
        store = problem.get_gf_store(store_ids[0])

        interpolation = 'nearest_neighbor'

        source.ensure_tractions()
        source.discretize_patches(store, interpolation)
        dislocations = source.get_okada_slip(scale_slip=True)

        fig = plt.figure()
        ax = fig.gca()

        abs_disloc = num.linalg.norm(dislocations, axis=1)
        abs_disloc = abs_disloc.reshape(source.nx, source.ny)

        im = ax.imshow(
            abs_disloc.T,
            cmap='YlOrRd',
            origin='upper',
            aspect='auto',
            extent=(0., source.length, 0., source.width))

        slip_strike = dislocations[:, 0]
        slip_dip = dislocations[:, 1]

        patch_length = source.length / source.nx
        patch_width = source.width / source.ny

        x_quiver = num.repeat(num.arange(source.nx), source.ny)\
            * patch_length + patch_length/2.
        y_quiver = num.tile(num.arange(source.ny), source.nx)\
            * patch_width + patch_width/2.

        ax.quiver(
            x_quiver, y_quiver, slip_strike, -slip_dip,
            facecolor='white', edgecolor='k', linewidth=.5,
            alpha=.7)

        ax.invert_yaxis()

        nucleation_x = ((source.nucleation_x + 1.) / 2.) * source.length
        nucleation_y = ((source.nucleation_y + 1.) / 2.) * source.width

        ax.scatter(
            nucleation_x, nucleation_y,
            s=20, color='k', marker='*',
            alpha=.7)

        ax.xaxis.set_major_formatter(FuncFormatter(km_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(km_fmt))

        ax.set_xlabel('Length [km]')
        ax.set_ylabel('Down-dip width [km]')

        cmap = fig.colorbar(im)
        cmap.set_label('Slip [m]')

        yield PlotItem(name='fig_1'), fig
