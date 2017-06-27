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


import abc

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import settings
from evaluations import radar_chart
from metrics import MSE, BadPix
from scenes import BaseScene
from utils.logger import log
from utils import plotting, misc


class BaseStratified(BaseScene):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, **kwargs):
        super(BaseStratified, self).__init__(name, category=settings.STRATIFIED_SCENE, **kwargs)

    def set_scale_for_algo_overview(self):
        self.set_high_gt_scale()

    @staticmethod
    def plot_radar_chart(algorithms):
        from scenes import Backgammon, Pyramids, Dots, Stripes
        from metrics import DotsBackgroundMSE, MissedDots, BackgammonThinning, BackgammonFattening, \
            DarkStripes, BrightStripes, StripesLowTexture, PyramidsSlantedBumpiness, PyramidsParallelBumpiness, Runtime

        backgammon = Backgammon(gt_scale=10.0)
        pyramids = Pyramids(gt_scale=1.0)
        dots = Dots(gt_scale=10.0)
        stripes = Stripes(gt_scale=10.0)

        dm_pairs = [[dots, DotsBackgroundMSE(), 6],
                    [dots, MissedDots(), 120],
                    [backgammon, BackgammonFattening(), 60],
                    [backgammon, BackgammonThinning(), 8],
                    [stripes, DarkStripes(), 64],
                    [stripes, BrightStripes(), 64],
                    [stripes, StripesLowTexture(), 64],
                    [pyramids, PyramidsSlantedBumpiness(), 6],
                    [pyramids, PyramidsParallelBumpiness(), 6]]

        scores_a_m = np.full((len(algorithms), len(dm_pairs)+3), fill_value=np.nan)

        # averaged scores of pixel metrics over all scenes
        all_scenes = [backgammon, pyramids, dots, stripes]

        metric_names = [MSE().get_display_name(),  BadPix().get_display_name()]
        scores_a_m[:, 0] = BaseScene.get_average_scores(algorithms, MSE(), all_scenes)
        scores_a_m[:, 1] = BaseScene.get_average_scores(algorithms, BadPix(), all_scenes)
        max_per_metric = [12, 32]

        # scene specific scores
        for idx_m, (scene, metric, vmax) in enumerate(dm_pairs):
            log.info("Computing scores for: %s" % metric.get_display_name())
            scores_a_m[:, idx_m+2] = scene.get_scores(algorithms, metric)
            metric_names.append(metric.get_display_name().replace(":", ":\n"))
            max_per_metric.append(vmax)

        runtime = Runtime(log=True)
        metric_names.append(runtime.get_display_name())
        scores_a_m[:, 2+len(dm_pairs)] = BaseScene.get_average_scores(algorithms, runtime, all_scenes)
        max_per_metric.append(6)

        fig_path = plotting.get_path_to_figure("radar_stratified")
        radar_chart.plot(scores_a_m, metric_names, algorithms, fig_path, max_per_metric)

    def plot_algo_overview(self, algorithms, with_metric_vis=True):
        self.set_scale_for_algo_overview()
        metrics = self.get_scene_specific_stratified_metrics()
        n_metrics = len(metrics)

        if not with_metric_vis:
            rows, cols = 2 + n_metrics, len(algorithms) + 2
            fig = plt.figure(figsize=(2.6*len(algorithms), 4.9))
            offset = 0
        else:
            rows, cols = 2 + 2*n_metrics, len(algorithms) + 2
            fig = plt.figure(figsize=(2.6*len(algorithms), rows+3))
            offset = n_metrics

        fontsize = 14
        labelpad = -15
        hscale = 7
        wscale = 5
        width_ratios = [wscale] * (len(algorithms) + 1) + [1]
        height_ratios = [hscale] * (rows - n_metrics) + [1] * n_metrics
        gs = gridspec.GridSpec(rows, cols, height_ratios=height_ratios, width_ratios=width_ratios)

        gt = self.get_gt()
        dummy = np.ones((self.get_height() / hscale, self.get_width()))
        colorbar_height, w = np.shape(gt)
        colorbar_width = w / float(wscale)

        # first column (gt, center view, ...)
        plt.subplot(gs[0])
        plt.imshow(gt, **settings.disp_map_args(self))
        plt.title("Ground Truth", fontsize=fontsize)
        plt.ylabel("Disparity Map", fontsize=fontsize)

        plt.subplot(gs[cols])
        plt.imshow(self.get_center_view())
        plt.ylabel("diff: GT - Algo", fontsize=fontsize)

        for idx_m, metric in enumerate(metrics):
            plt.subplot(gs[(2+idx_m+offset)*cols])
            plt.xlabel(metric.get_short_name(), labelpad=labelpad, fontsize=fontsize)
            plt.imshow(dummy, cmap="gray_r")

        # algorithm columns
        for idx_a, algorithm in enumerate(algorithms):
            log.info("Processing algorithm: %s" % algorithm)
            algo_result = misc.get_algo_result(self, algorithm)

            # algorithm disparities
            plt.subplot(gs[idx_a+1])
            plt.title(algorithm.get_display_name(), fontsize=fontsize)
            cm1 = plt.imshow(algo_result, **settings.disp_map_args(self))

            # algorithm diff map
            plt.subplot(gs[cols+idx_a+1])
            cm2 = plt.imshow(gt - algo_result, **settings.diff_map_args())

            # add colorbar if last column
            if idx_a == (len(algorithms) - 1):
                plotting.add_colorbar(gs[idx_a + 2], cm1, colorbar_height, colorbar_width,
                                   colorbar_bins=5, fontsize=10, img_width=1)
                plotting.add_colorbar(gs[cols + idx_a + 2], cm2, colorbar_height, colorbar_width,
                                   colorbar_bins=5, fontsize=10, img_width=1)

            # score + background color for metrics
            for idx_m, metric in enumerate(metrics):

                if with_metric_vis:
                    plt.subplot(gs[(2+idx_m)*cols+idx_a+1])
                    score, vis = metric.get_score(algo_result, gt, self, with_visualization=True)
                    cm3 = plt.imshow(vis, **settings.metric_args(metric))

                    if idx_a == 0:
                        plt.ylabel(metric.get_short_name(), fontsize=fontsize)
                    elif idx_a == (len(algorithms) - 1):
                        plotting.add_colorbar(gs[(2+idx_m)*cols+idx_a+2], cm3, colorbar_height, colorbar_width,
                                           colorbar_bins=metric.colorbar_bins, fontsize=10, img_width=1)

                else:
                    score = metric.get_score(algo_result, gt, self)

                plt.subplot(gs[(2+idx_m+offset)*cols+idx_a+1])
                plt.imshow(dummy * score, **settings.score_color_args(vmin=metric.vmin, vmax=metric.vmax))
                plt.xlabel(metric.format_score(score), labelpad=labelpad, fontsize=fontsize)

        fig_path = plotting.get_path_to_figure("algo_overview_" + self.get_name() + with_metric_vis * "_vis")
        plotting.save_tight_figure(fig, fig_path, wspace=0.04, hide_frames=True, remove_ticks=True)