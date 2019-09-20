import logging

import numpy as num

from pyrocko.guts import Float, Tuple, String
from pyrocko import gf

from ..base import (
    MisfitTarget, TargetGroup, MisfitResult)
from grond.meta import has_get_plot_classes

guts_prefix = 'grond'
logger = logging.getLogger('grond.targets.phase_pick.target')


def log_exclude(target, reason):
    logger.debug('Excluding potential target %s: %s' % (
        target.string_id(), reason))


class PhasePickTargetGroup(TargetGroup):

    '''
    Generate targets to fit phase arrival times.
    '''

    distance_min = Float.T(optional=True)
    distance_max = Float.T(optional=True)
    distance_3d_min = Float.T(optional=True)
    distance_3d_max = Float.T(optional=True)
    depth_min = Float.T(optional=True)
    depth_max = Float.T(optional=True)
    store_id = gf.StringID.T(optional=True)
    pick_synthetic_traveltime = gf.Timing.T(
        help='Synthetic phase arrival definition.')
    pick_phasename = String.T(
        help='Name of phase in pick file.')

    def get_targets(self, ds, event, default_path='none'):
        logger.debug('Selecting phase pick targets...')
        origin = event
        targets = []

        for st in ds.get_stations():

            target = PhasePickTarget(
                codes=st.nsl(),
                lat=st.lat,
                lon=st.lon,
                north_shift=st.north_shift,
                east_shift=st.east_shift,
                depth=st.depth,
                store_id=self.store_id,
                manual_weight=self.weight,
                normalisation_family=self.normalisation_family,
                path=self.path or default_path,
                pick_synthetic_traveltime=self.pick_synthetic_traveltime,
                pick_phasename=self.pick_phasename)

            if self.distance_min is not None and \
               target.distance_to(origin) < self.distance_min:
                log_exclude(target, 'distance < distance_min')
                continue

            if self.distance_max is not None and \
               target.distance_to(origin) > self.distance_max:
                log_exclude(target, 'distance > distance_max')
                continue

            if self.distance_3d_min is not None and \
               target.distance_3d_to(origin) < self.distance_3d_min:
                log_exclude(target, 'distance_3d < distance_3d_min')
                continue

            if self.distance_3d_max is not None and \
               target.distance_3d_to(origin) > self.distance_3d_max:
                log_exclude(target, 'distance_3d > distance_3d_max')
                continue

            if self.depth_min is not None and \
               target.depth < self.depth_min:
                log_exclude(target, 'depth < depth_min')
                continue

            if self.depth_max is not None and \
               target.depth > self.depth_max:
                log_exclude(target, 'depth > depth_max')
                continue

            target.set_dataset(ds)
            targets.append(target)

        return targets


class PhasePickResult(MisfitResult):
    pass


@has_get_plot_classes
class PhasePickTarget(gf.Location, MisfitTarget):

    '''
    Target to fit phase arrival times.
    '''

    codes = Tuple.T(
        3, String.T(),
        help='network, station, location codes.')

    store_id = gf.StringID.T(
        help='ID of Green\'s function store (only used for earth model).')

    pick_synthetic_traveltime = gf.Timing.T(
        help='Synthetic phase arrival definition.')

    pick_phasename = String.T(
        help='Name of phase in pick file.')

    can_bootstrap_weights = True

    def __init__(self, **kwargs):
        gf.Location.__init__(self, **kwargs)
        MisfitTarget.__init__(self, **kwargs)

    @classmethod
    def get_plot_classes(cls):
        from . import plot
        plots = super(PhasePickTarget, cls).get_plot_classes()
        plots.extend(plot.get_plot_classes())
        return plots

    def string_id(self):
        return '.'.join(x for x in (self.path,) + self.codes)

    def get_plain_targets(self, engine, source):
        return self.prepare_modelling(engine, source, None)

    def prepare_modelling(self, engine, source, targets):
        return []

    def get_times(self, engine, source):
        tobs = None
        tsyn = None
        ds = self.get_dataset()

        store = engine.get_store(self.store_id)
        tsyn = source.time + store.t(
            self.pick_synthetic_traveltime, source, self)

        marker = ds.get_pick(
            source.name,
            self.codes[:3],
            self.pick_phasename)

        if marker:
            tobs = marker.tmin

        return tobs, tsyn

    def finalize_modelling(
            self, engine, source, modelling_targets, modelling_results):

        ds = self.get_dataset()  # noqa

        tobs, tsyn = self.get_times(engine, source)
        misfit = abs(tobs - tsyn)

        norm = 1.0
        result = PhasePickResult(
            misfits=num.array([[misfit, norm]], dtype=num.float))

        return result


__all__ = '''
    PhasePickTargetGroup
    PhasePickTarget
    PhasePickResult
'''.split()