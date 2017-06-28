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

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import misc
from evaluations import bad_pix_series, metric_overviews
from utils import plotting
import settings

SUBDIR = "cvprws_2017_survey"


def plot_scene_overview(scenes):

    # prepare grid figure
    fig = plt.figure(figsize=(21.6, 4))
    fs = 16

    rows, cols = 2, len(scenes)
    hscale, wscale = 5, 7
    grids = gridspec.GridSpec(rows, cols, height_ratios=[hscale] * rows, width_ratios=[wscale] * cols)

    # plot center view and ground truth for each scene
    for idx_s, scene in enumerate(scenes):

        center_view = scene.get_center_view()
        plt.subplot(grids[idx_s])
        plt.imshow(center_view)
        plt.title("\n\n" + scene.get_display_name(), fontsize=fs)

        gt = scene.get_gt()
        plt.subplot(grids[cols+idx_s])
        if scene.hidden_gt():
            gt = plotting.pixelize(gt, noise_factor=0.5)
        plt.imshow(gt, **settings.disp_map_args(scene))

    # add text
    height = 785
    plt.gca().annotate("(a) Stratified Scenes", (400, 420), (500, height), fontsize=fs, xycoords='figure pixels')
    plt.gca().annotate("(b) Training Scenes", (400, 420), (1910, height), fontsize=fs, xycoords='figure pixels')
    plt.gca().annotate("(c) Test Scenes (Hidden Ground Truth)", (400, 420), (3070, height), fontsize=fs, xycoords='figure pixels')

    # save figure
    fig_path = plotting.get_path_to_figure("scenes", subdir=SUBDIR)
    plotting.save_tight_figure(fig, fig_path, hide_frames=True, remove_ticks=True, hspace=0.02, wspace=0.02, dpi=200)


def plot_bad_pix_series(algorithms, with_cached_scores=False, penalize_missing_pixels=False):
    scene_sets = [[misc.get_stratified_scenes(), "Stratified Scenes", "stratified"],
                  [misc.get_training_scenes() + misc.get_test_scenes(), "Test and Training Scenes", "photorealistic"]]

    for scene_set, title, fig_name in scene_sets:
        bad_pix_series.plot(algorithms, scene_set,
                            with_cached_scores=with_cached_scores,
                            penalize_missing_pixels=penalize_missing_pixels,
                            title=title, subdir=SUBDIR, fig_name="bad_pix_series_" + fig_name)


def plot_normals_overview(algorithms, scenes):
    metric_overviews.plot_normals(algorithms, scenes, subdir=SUBDIR)