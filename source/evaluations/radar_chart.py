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
import numpy as np

from utils import plotting, misc
from utils.logger import log


def plot(algorithms, scenes, metrics, average="median", axis_labels=None, max_per_metric=None,
         fig_name="radar", subdir="radar", title="", fs=16):

    scores_scenes_metrics_algos = misc.collect_scores(algorithms, scenes, metrics, masked=True)

    if average == "median":
        scores_metrics_algos = np.ma.median(scores_scenes_metrics_algos, axis=0)
    elif average == "mean":
        scores_metrics_algos = np.ma.average(scores_scenes_metrics_algos, axis=0)
    else:
        raise Exception('Only "median" and "mean" are handled, got "%s".' % average)

    if axis_labels is None:
        axis_labels = [m.get_display_name().replace(":", "\n") for m in metrics]

    plot_scores(scores_metrics_algos, algorithms, axis_labels,
                fig_name, subdir, title, fs, max_per_metric)


def plot_scores(scores_metrics_algos, algorithms, axis_labels, fig_name, subdir, title, fs, max_per_metric=None):

    # prepare figure
    fig = plt.figure(figsize=(10, 10))
    n_circles = 4 + 1

    rect = [0.1, 0.1, 0.8, 0.8]  # left, bottom, width, height
    n_axes = np.shape(scores_metrics_algos)[0]
    axes = [fig.add_axes(rect, projection="polar", label="axes%d" % i) for i in range(n_axes)]
    angles = np.arange(90, 90+360, 360.0/n_axes) % 360

    # add metric labels
    axes[0].set_thetagrids(angles, labels=axis_labels, fontsize=16)

    # hide default labeling
    for ax in axes[1:]:
        ax.patch.set_visible(False)
        ax.grid("off")
        ax.xaxis.set_visible(False)

    # compute metric values at axis intersections
    axes_values = []

    if max_per_metric is None:
        max_per_metric = np.max(scores_metrics_algos, axis=1) * 1.2

    for vmax in max_per_metric:
        steps = list(np.linspace(0, vmax, n_circles))
        steps = steps[:-1]
        axes_values.append(steps)

    # add metric values at axis intersections
    for ax, angle, labels in zip(axes, angles, axes_values):
        decimals = int(np.ceil(np.log10(np.max(labels))))
        if decimals <= -1:
            str_labels = [('%0.3f' % e) for e in labels[1:]]
        if decimals == 0:
            str_labels = [('%0.2f' % e) for e in labels[1:]]
        elif decimals >= 1:
            str_labels = [('%0.1f' % e) for e in labels[1:]]

        # add zero to inner circle
        str_labels = ["0"] + str_labels

        ax.set_rgrids(range(1, n_circles+1), angle=angle, labels=str_labels, fontsize=14)
        ax.spines["polar"].set_visible(False)
        ax.set_ylim(0, n_circles)

    # add first angle at the end to close the line-loop
    angle = np.deg2rad(np.r_[angles, angles[0]])

    # plot one line per algorithm, passing through all metrics
    for idx_a, algorithm in enumerate(algorithms):
        metric_scores_cur_algo = scores_metrics_algos[:, idx_a]

        for idx_m in range(len(axis_labels)):
            step = axes_values[idx_m][1] - axes_values[idx_m][0]
            scale_factor = n_circles / (float(max_per_metric[idx_m]) + step)
            adjusted_value = (metric_scores_cur_algo[idx_m] + step) * scale_factor
            metric_scores_cur_algo[idx_m] = adjusted_value

        # add first score at end to close the line-loop
        metric_scores_cur_algo = np.r_[metric_scores_cur_algo, metric_scores_cur_algo[0]]
        ax.plot(angle, metric_scores_cur_algo, label=algorithm.get_display_name(),
                ls=algorithm.get_line_style(), lw=2, alpha=1, color=algorithm.get_color())

    ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1), frameon=False, prop={'size': fs}, labelspacing=0.2)
    plt.title(title + "\n\n", fontsize=fs+4)
    plotting.save_fig(fig, plotting.get_path_to_figure(fig_name, subdir=subdir))


def compare_relative_performances(algorithms, scenes, metrics, all_but=0):
    scores_scenes_metrics_algos = misc.collect_scores(algorithms, scenes, metrics, masked=True)
    scores_metrics_algos = np.ma.median(scores_scenes_metrics_algos, axis=0)

    n_metrics = np.shape(scores_metrics_algos)[0]
    winners = dict()

    for idx_a1, algorithm1 in enumerate(algorithms):
        scores_a1 = scores_metrics_algos[:, idx_a1]
        worse_on_all_but_n = []

        for idx_a2, algorithm2 in enumerate(algorithms):
            scores_a2 = scores_metrics_algos[:, idx_a2]
            n_better = np.sum(scores_a1 < scores_a2)

            if n_better == n_metrics - all_but:
                worse_on_all_but_n.append(algorithm2)

        if len(worse_on_all_but_n) > 0:
            winners[algorithm1] = worse_on_all_but_n

    n_winners = len(winners.keys())
    log.info("%d Algorithm(s) better on all but %d score(s)." % (n_winners, all_but))

    for idx_a, (algorithm, better_than) in enumerate(winners.items()):
        log.info("%d) %s is better than: %s" % (idx_a+1, algorithm.get_display_name(), ", ".join(a.get_display_name() for a in better_than)))

    return winners
