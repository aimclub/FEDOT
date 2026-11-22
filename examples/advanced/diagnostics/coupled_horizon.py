"""Prospective classification-only LightGBM horizon profiles for isolated copies."""
from numbers import Integral

PROFILES = ((100, .1), (150, .1), (300, .05), (450, 1 / 30))


def apply_profile(parameters):
    profile = parameters.get('horizon_profile')
    if profile is None:
        return
    if isinstance(profile, bool) or not isinstance(profile, Integral) or not 0 <= profile < len(PROFILES):
        raise ValueError('horizon_profile must be an integer in [0, 3]')
    rounds, rate = PROFILES[profile]
    parameters.update(n_estimators=rounds, learning_rate=rate)
