Gap Filling in Time Series
===========================

Classes for filling in the gaps in time series data.

Classes:
    SimpleGapFiller: Base class used for filling in the gaps in time series with simple methods.
    ModelGapFiller: Class used for filling in the gaps in time series with more complex methods.

Methods (SimpleGapFiller):
    linear_interpolation(): Fill gaps using linear interpolation.
    local_poly_approximation(): Fill gaps using Savitzky-Golay filter.
    batch_poly_approximation(): Fill gaps using batch polynomial approximations.

Methods (ModelGapFiller):
    forward_inverse_filling(): Fill gaps using forward and inverse directions of predictions.
    forward_filling(): Fill gaps using only forward direction predictions.
