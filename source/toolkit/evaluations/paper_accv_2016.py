# -*- coding: utf-8 -*-

############################################################################
#  This file is part of the 4D Light Field Benchmark.                      #
#                                                                          #
#  This work is licensed under the Creative Commons                        #
#  Attribution-NonCommercial-ShareAlike 4.0 International License.         #
#  To view a copy of this license,                                         #
#  visit http://creativecommons.org/licenses/by-nc-sa/4.0/.                #
#                                                                          #
#  Authors: Katrin Honauer & Ole Johannsen                                 #
#  Contact: contact@lightfield-analysis.net                                #
#  Website: www.lightfield-analysis.net                                    #
#                                                                          #
#  The 4D Light Field Benchmark was jointly created by the University of   #
#  Konstanz and the HCI at Heidelberg University. If you use any part of   #
#  the benchmark, please cite our paper "A dataset and evaluation          #
#  methodology for depth estimation on 4D light fields". Thanks!           #
#                                                                          #
#  @inproceedings{honauer2016benchmark,                                    #
#    title={A dataset and evaluation methodology for depth estimation on   #
#           4D light fields},                                              #
#    author={Honauer, Katrin and Johannsen, Ole and Kondermann, Daniel     #
#            and Goldluecke, Bastian},                                     #
#    booktitle={Asian Conference on Computer Vision},                      #
#    year={2016},                                                          #
#    organization={Springer}                                               #
#    }                                                                     #
#                                                                          #
############################################################################


from toolkit.evaluations import radar_chart
from toolkit.metrics import MSE, BadPix, Runtime, \
    BumpinessPlanes, BumpinessContinSurf, FineThinning, FineFattening, Discontinuities, \
    PyramidsSlantedBumpiness, PyramidsParallelBumpiness, DotsBackgroundMSE, MissedDots, \
    StripesLowTexture, BrightStripes, DarkStripes, BackgammonFattening, BackgammonThinning
from toolkit.scenes import Pyramids, Dots, Stripes, Backgammon
from toolkit.utils import misc


def plot_radar_charts(algorithms, log_runtime=True, with_test_scenes=False, subdir="radar"):

    # photorealistic training
    photorealistic_metrics = [MSE(),
                              BadPix(0.07),
                              BumpinessPlanes(name="Planar\nSurfaces"),
                              BumpinessContinSurf(name="Continuous\nSurfaces"),
                              FineThinning(name="Fine Structure\nThinning"),
                              FineFattening(name="Fine Structure\nFattening"),
                              Discontinuities(name="Discontinuity\nRegions"),
                              Runtime(log=log_runtime)]

    metric_names = [m.get_display_name() for m in photorealistic_metrics]
    max_per_metric = [12, 40, 4, 4, 40, 80, 80, 6]
    radar_chart.plot(algorithms,
                     scenes=misc.get_training_scenes(),
                     metrics=photorealistic_metrics,
                     axis_labels=metric_names,
                     average="mean",
                     max_per_metric=max_per_metric,
                     title="Mean Scores for Training Scenes",
                     fig_name="radar_training",
                     subdir=subdir)

    # stratified
    metrics = [MSE(), BadPix(0.07),
               DotsBackgroundMSE(), MissedDots(),
               BackgammonFattening(), BackgammonThinning(),
               DarkStripes(), BrightStripes(), StripesLowTexture(),
               PyramidsSlantedBumpiness(), PyramidsParallelBumpiness(),
               Runtime(log=log_runtime)]

    scenes = [Backgammon(gt_scale=10.0), Pyramids(gt_scale=1.0),
              Dots(gt_scale=10.0), Stripes(gt_scale=10.0)]

    metric_names = [m.get_display_name().replace(":", ":\n") for m in metrics]
    max_per_metric = [12, 32, 6, 120, 40, 8, 64, 64, 64, 4, 4, 6]
    radar_chart.plot(algorithms,
                     scenes=scenes,
                     metrics=metrics,
                     axis_labels=metric_names,
                     average="mean",
                     max_per_metric=max_per_metric,
                     title="Mean Scores for Stratified Scenes",
                     fig_name="radar_stratified",
                     subdir=subdir)

    # photorealistic test
    if with_test_scenes:
        metric_names = [m.get_display_name() for m in photorealistic_metrics]
        max_per_metric = [16, 40, 4, 4, 16, 80, 80, 6]
        radar_chart.plot(algorithms,
                         scenes=misc.get_test_scenes(),
                         metrics=photorealistic_metrics,
                         axis_labels=metric_names,
                         average="mean",
                         max_per_metric=max_per_metric,
                         title="Mean Scores for Test Scenes",
                         fig_name="radar_test",
                         subdir=subdir)
