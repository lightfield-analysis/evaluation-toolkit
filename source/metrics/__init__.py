from general_metrics import BaseMetric, BadPix, MSE, Runtime, Quantile

from region_metrics import Discontinuities, FineFattening, FineThinning, \
    BumpinessContinSurf, BumpinessPlanes, MAEPlanes, MAEContinSurf

from stratified_metrics import BackgammonThinning, BackgammonFattening, \
    PyramidsParallelBumpiness, PyramidsSlantedBumpiness, \
    MissedDots, DotsBackgroundMSE, \
    StripesLowTexture, BrightStripes, DarkStripes