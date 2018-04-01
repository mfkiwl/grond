import logging

import numpy as num

from pyrocko.guts import Float, Tuple, String
from pyrocko import gf

from ..base import (
    MisfitTarget, TargetGroup, MisfitResult)
from . import measure as fm
from grond import dataset

guts_prefix = 'grond'
logger = logging.getLogger('grond.targets.waveform.phase_ratio')


class PhaseRatioTargetGroup(TargetGroup):
    distance_min = Float.T(optional=True)
    distance_max = Float.T(optional=True)
    distance_3d_min = Float.T(optional=True)
    distance_3d_max = Float.T(optional=True)
    depth_min = Float.T(optional=True)
    depth_max = Float.T(optional=True)
    measure_a = fm.FeatureMeasure.T()
    measure_b = fm.FeatureMeasure.T()

    def get_targets(self, ds, event, default_path):
        logger.debug('Selecting waveform targets...')
        origin = event
        targets = []

        for st in ds.get_stations():
            exclude = False
            for measure in [self.measure_a, self.measure_b]:
                for cha in measure.channels:
                    if ds.is_blacklisted((st.nsl() + (cha,))):
                        exclude = True

            if exclude:
                continue

            target = PhaseRatioTarget(
                codes=st.nsl(),
                lat=st.lat,
                lon=st.lon,
                depth=st.depth,
                interpolation=self.interpolation,
                store_id=self.store_id,
                measure_a=self.measure_a,
                measure_b=self.measure_b,
                manual_weight=self.weight,
                normalisation_family=self.normalisation_family,
                path=self.path or default_path,
                backazimuth=0.0)

            if self.distance_min is not None and \
               target.distance_to(origin) < self.distance_min:
                continue

            if self.distance_max is not None and \
               target.distance_to(origin) > self.distance_max:
                continue

            if self.distance_3d_min is not None and \
               target.distance_3d_to(origin) < self.distance_3d_min:
                continue

            if self.distance_3d_max is not None and \
               target.distance_3d_to(origin) > self.distance_3d_max:
                continue

            if self.depth_min is not None and \
               target.depth < self.depth_min:
                continue

            if self.depth_max is not None and \
               target.depth > self.depth_max:
                continue

            bazi, _ = target.azibazi_to(origin)
            target.backazimuth = bazi
            target.set_dataset(ds)
            targets.append(target)

        return targets


class PhaseRatioResult(MisfitResult):
    pass


class PhaseRatioTarget(gf.Location, MisfitTarget):

    codes = Tuple.T(
        3, String.T(),
        help='network, station, location codes.')

    store_id = gf.StringID.T(
        help='ID of Green\'s function store to use for the computation.')

    backazimuth = Float.T(optional=True)

    interpolation = gf.InterpolationMethod.T()

    measure_a = fm.FeatureMeasure.T()
    measure_b = fm.FeatureMeasure.T()

    def __init__(self, **kwargs):
        gf.Location.__init__(self, **kwargs)
        MisfitTarget.__init__(self, **kwargs)

    def string_id(self):
        return '.'.join(x for x in (self.path,) + self.codes if x)

    def get_plain_targets(self, engine, source):
        return self.prepare_modelling(engine, source)

    def prepare_modelling(self, engine, source):
        modelling_targets = []
        for measure in [self.measure_a, self.measure_b]:
            modelling_targets.extend(measure.get_modelling_targets(
                self.codes, self.lat, self.lon, self.depth, self.store_id,
                self.backazimuth))

        return modelling_targets

    def finalize_modelling(
            self, engine, source, modelling_targets, modelling_results):

        ds = self.get_dataset()

        try:
            imt = 0
            amps = []
            for measure in [self.measure_a, self.measure_b]:
                nmt_this = measure.get_nmodelling_targets()
                amp_obs, _ = measure.evaluate(
                    engine, source,
                    modelling_targets[imt:imt+nmt_this],
                    dataset=ds)

                amp_syn, _ = measure.evaluate(
                    engine, source,
                    modelling_targets[imt:imt+nmt_this],
                    trs=[r.trace.pyrocko_trace()
                         for r
                         in modelling_results[imt:imt+nmt_this]])

                amps.append((amp_obs, amp_syn))

                imt += nmt_this

            (a_obs, a_syn), (b_obs, b_syn) = amps

            res_a = a_obs / (a_obs + b_obs) - a_syn / (a_syn + b_syn)

            misfit = num.abs(res_a)
            norm = 1.0

            result = PhaseRatioResult(
                misfits=num.array([[misfit, norm]], dtype=num.float))

            return result

        except dataset.NotFound as e:
            logger.debug(str(e))
            return gf.SeismosizerError('no waveform data, %s' % str(e))


__all__ = '''
    PhaseRatioTargetGroup
    PhaseRatioTarget
    PhaseRatioResult
'''.split()
