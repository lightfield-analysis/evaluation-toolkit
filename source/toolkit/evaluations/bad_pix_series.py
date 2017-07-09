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


import os.path as op
import pickle

import matplotlib.pyplot as plt
import numpy as np

from toolkit import settings
from toolkit.metrics import BadPix
from toolkit.utils import log, misc, plotting, file_io


THRESHOLDS = np.arange(0, 0.102, 0.002)


def get_fname_scores(scenes):
    descr = "_".join([s.get_name() for s in scenes])
    return op.join(settings.TMP_PATH, "bad_pix_series_scores_%s.pickle" % descr)


def plot(algorithms, scenes, thresholds=THRESHOLDS, with_cached_scores=False,
         penalize_missing_pixels=False, title=None, subdir="bad_pix_series", fig_name=None,
         fig_size=(16, 6), legend_pos=(1.19, -0.04), marker_size=2.3, fs=16):

    # prepare scores
    fname_scores = get_fname_scores(scenes)
    if not op.isfile(fname_scores) or not with_cached_scores:
        percentages_algo_thresh = compute_scores(algorithms, scenes, thresholds,
                                                 penalize_missing_pixels=penalize_missing_pixels)
        if with_cached_scores:
            fname_scores = get_fname_scores(scenes)
            file_io.check_dir_for_fname(fname_scores)
            with open(fname_scores, "w") as f:
                pickle.dump(percentages_algo_thresh, f)
    else:
        with open(fname_scores, "r") as f:
            percentages_algo_thresh = pickle.load(f)

    # prepare figure
    fig = plt.figure(figsize=fig_size)
    x_ticks = np.arange(len(thresholds))

    # plot BadPix scores per algorithm
    for idx_a, algorithm in enumerate(algorithms):
        plt.plot(x_ticks, percentages_algo_thresh[idx_a, :],
                 alpha=0.9, color=algorithm.get_color(),
                 lw=1.3, ls=algorithm.get_line_style(), label=algorithm.get_display_name(),
                 marker="D", markersize=marker_size, markeredgecolor="none")

    # add vertical lines for special thresholds
    indices = [i for i, t in enumerate(thresholds) if t in [0.01, 0.03, 0.07]]
    for idx in indices:
        plt.plot([idx, idx], [0, 100], lw=1., c="k", alpha=0.9, ls=":")

    # add horizontal line for Q25
    plt.plot(x_ticks, [25]*len(x_ticks), lw=1., c="k", alpha=0.9, ls=":")

    # add axis ticks and labels
    plt.xticks(x_ticks, ["%0.03f" % t for t in thresholds], rotation=90, fontsize=fs)
    plt.xlabel("Threshold for absolute disparity error", fontsize=fs)
    plt.ylabel("Percentage of pixels\nbelow threshold", fontsize=fs)
    plt.ylim([0, 103])

    # finalize figure
    if title is None:
        title = "Scenes: %s" % ", ".join(scene.get_display_name() for scene in scenes)
    plt.title(title, fontsize=fs)
    plotting.hide_upper_right()
    plt.legend(frameon=False, loc="lower right", bbox_to_anchor=legend_pos,
               prop={'size': fs}, labelspacing=0.2)

    # save figure
    if fig_name is None:
        fig_name = "bad_pix_series_%s" % ("_".join(scene.get_name() for scene in scenes))
    fig_path = plotting.get_path_to_figure(fig_name, subdir)
    plotting.save_tight_figure(fig, fig_path, hide_frames=False, remove_ticks=False, hspace=0.07)


def compute_scores(algorithms, scenes, thresholds=THRESHOLDS, penalize_missing_pixels=True):
    percentages_algo_thresh = np.full((len(algorithms), len(thresholds)), fill_value=np.nan)
    bad_pix_metric = BadPix()
    max_diff = np.max(thresholds)

    for idx_a, algorithm in enumerate(algorithms):
        combined_diffs = np.full(0, fill_value=np.nan)
        log.info('Computing BadPix scores for: %s' % algorithm.get_display_name())

        for scene in scenes:
            gt = scene.get_gt()
            algo_result = misc.get_algo_result(algorithm, scene)
            diffs = np.abs(algo_result - gt)

            mask_valid = misc.get_mask_valid(algo_result) * misc.get_mask_valid(diffs)
            mask_eval = bad_pix_metric.get_evaluation_mask(scene)

            if penalize_missing_pixels:
                # penalize all invalid algorithm pixels with maximum error
                diffs[~mask_valid] = max_diff + 100
                diffs = diffs[mask_eval]
            else:
                diffs = diffs[mask_eval*mask_valid]

            combined_diffs = np.concatenate((combined_diffs, diffs))

        # compute BadPix score for each threshold
        for idx_t, t in enumerate(thresholds):
            bad_pix_metric.thresh = t
            bad_pix_score = bad_pix_metric.get_score_from_diffs(combined_diffs)
            percentages_algo_thresh[idx_a, idx_t] = 100 - bad_pix_score

    return percentages_algo_thresh
