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
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

import settings
from metrics import MSE, BadPix,  BumpinessPlanes, Discontinuities, BumpinessContinSurf, FineFattening, \
    FineThinning, Runtime, MAEPlanes, MAEContinSurf
from scenes import BaseScene
from utils import plotting, misc
from utils.logger import log



class PhotorealisticScene(BaseScene):

    def __init__(self, name, **kwargs):
        super(PhotorealisticScene, self).__init__(name, **kwargs)

    def get_applicable_metrics_high_res(self):
        return [metric for metric in misc.get_region_metrics() if metric.is_applicable_for_high_res_scene(self)]

    def get_applicable_metrics_low_res(self):
        metrics = [metric for metric in misc.get_region_metrics() if metric.is_applicable_for_low_res_scene(self)]
        #  add general metrics to low resolution evaluation
        return misc.get_general_metrics() + metrics

    @staticmethod
    def get_applicable_scenes(all_scenes, metric):
        if metric.is_general():
            return all_scenes
        return [s for s in all_scenes if metric.mask_exists(s, settings.LOWRES) or metric.mask_exists(s, settings.HIGHRES)]

    def plot_algo_overview(self, algorithms, subdir="algo_overview"):
        accv_metrics = [MSE(), BadPix(0.07), BumpinessPlanes(), BumpinessContinSurf(),
                        Discontinuities(), FineFattening(), FineThinning()]
        metrics_low_res = [m for m in self.get_applicable_metrics_low_res() if m in accv_metrics]
        metrics_high_res = [m for m in self.get_applicable_metrics_high_res() if m in accv_metrics]

        # prepare figure
        fontsize = 6
        rows = len(metrics_low_res + metrics_high_res) + 1
        cols = len(algorithms) + 1
        fig = plt.figure(figsize=(cols, rows*1.1))
        grids = _get_grids(fig, rows, cols, axes_pad=-0.2)

        # center view on top left grid cell
        self.set_high_gt_scale()
        plt.sca(grids[0][0])
        plt.imshow(self.get_center_view())
        plt.title("Center View", fontsize=fontsize)
        plt.ylabel("Disparity Map", fontsize=fontsize)

        # mask visualizations + algorithm disparity maps + metric visualizations
        log.info("Computing scores and visualizations for LOW resolution metrics.")
        self.set_low_gt_scale()
        self.plot_metric_rows(grids, algorithms, metrics_low_res, offset=0, fontsize=fontsize)

        log.info("Computing scores and visualizations for HIGH resolution metrics.")
        self.set_high_gt_scale()
        self.plot_metric_rows(grids, algorithms, metrics_high_res, offset=len(metrics_low_res), fontsize=fontsize)

        # finalize figure
        for grid in grids:
            plotting.remove_ticks_from_axes(grid.axes_all)
            plotting.remove_frames_from_axes(grid.axes_all)
        plt.suptitle(self.get_display_name(), fontsize=fontsize+2)

        fig_path = plotting.get_path_to_figure("algo_overview_%s" % self.get_name(), subdir=subdir)
        plotting.save_fig(fig, fig_path, pad_inches=0.1)

    @staticmethod
    def _get_grids(fig, rows, cols, axes_pad=0):
        grids = []
        for row in range(rows):
            grid_id = int("%d%d%d" % (rows, 1, row + 1))
            grid = ImageGrid(fig, grid_id,
                             nrows_ncols=(1, cols),
                             axes_pad=(0.05, axes_pad),
                             share_all=True,
                             cbar_location="right",
                             cbar_mode="single",
                             cbar_size="10%",
                             cbar_pad="5%")
            grids.append(grid)
        return grids

    def plot_metric_rows(self, grids, algorithms, metrics, offset, fontsize):
        gt = self.get_gt()
        center_view = self.get_center_view()

        for idx_a, algorithm in enumerate(algorithms):
            log.info("Algorithm: %s" % algorithm)
            algo_result = misc.get_algo_result(self, algorithm)

            # add algorithm disparity map
            plt.sca(grids[0][idx_a + 1])
            cm = plt.imshow(algo_result, **settings.disp_map_args(self))
            plt.title(algorithm.get_display_name(), fontsize=fontsize)

            # add colorbar to last disparity map in row
            if idx_a == (len(algorithms) - 1):
                plotting.create_colorbar(cm, cax=grids[0].cbar_axes[0], colorbar_bins=7, fontsize=fontsize)

            # add algorithm metric visualizations
            for idx_m, metric in enumerate(metrics):
                log.info(metric.get_display_name())
                mask = metric.get_evaluation_mask(self)

                plt.sca(grids[idx_m + offset + 1][idx_a + 1])
                cm = self.plot_algo_vis_for_metric(metric, algo_result, gt, mask, self.hidden_gt(), fontsize)

                # add colorbar to last metric visualization in row
                if idx_a == len(algorithms) - 1:
                    plotting.create_colorbar(cm, cax=grids[idx_m + offset + 1].cbar_axes[0],
                                             colorbar_bins=metric.colorbar_bins, fontsize=fontsize)

                # add mask visualizations as 1st column
                if idx_a == 0:
                    plt.sca(grids[idx_m + offset + 1][0])
                    plotting.plot_img_with_transparent_mask(center_view, mask, alpha=0.7, color=settings.color_mask)
                    plt.ylabel(metric.get_short_name(), fontsize=fontsize)
                    plt.title("Region Mask", fontsize=fontsize)

    def plot_algo_vis_for_metric(self, metric, algo_result, gt, mask, hide_gt=False, fontsize=10):
        score, vis = metric.get_score(algo_result, gt, self, with_visualization=True)

        if hide_gt and metric.pixelize_results():
            vis = misc.pixelize(vis)

        # plot algorithm disparity map as background
        plt.imshow(algo_result, **settings.disp_map_args(self, cmap="gray"))

        # plot masked metric visualization on top
        cm = plt.imshow(np.ma.masked_array(vis, mask=~mask), **settings.metric_args(metric))
        plt.title(metric.format_score(score), fontsize=fontsize)

        return cm


# convenience classes for test and training scenes

class Bedroom(PhotorealisticScene):
    def __init__(self, name="bedroom", category=settings.TEST_SCENE, **kwargs):
        super(Bedroom, self).__init__(name, category=category, **kwargs)


class Bicycle(PhotorealisticScene):
    def __init__(self, name="bicycle", category=settings.TEST_SCENE, **kwargs):
        super(Bicycle, self).__init__(name, category=category,  **kwargs)


class Herbs(PhotorealisticScene):
    def __init__(self, name="herbs", category=settings.TEST_SCENE, **kwargs):
        super(Herbs, self).__init__(name, category=category,  **kwargs)


class Origami(PhotorealisticScene):
    def __init__(self, name="origami", category=settings.TEST_SCENE, **kwargs):
        super(Origami, self).__init__(name, category=category,  **kwargs)


class Boxes(PhotorealisticScene):
    def __init__(self, name="boxes", category=settings.TRAINING_SCENE, **kwargs):
        super(Boxes, self).__init__(name, category=category, **kwargs)


class Cotton(PhotorealisticScene):
    def __init__(self, name="cotton", category=settings.TRAINING_SCENE, **kwargs):
        super(Cotton, self).__init__(name, category=category, **kwargs)


class Dino(PhotorealisticScene):
    def __init__(self, name="dino", category=settings.TRAINING_SCENE, **kwargs):
        super(Dino, self).__init__(name, category=category, **kwargs)


class Sideboard(PhotorealisticScene):
    def __init__(self, name="sideboard", category=settings.TRAINING_SCENE, **kwargs):
        super(Sideboard, self).__init__(name, category=category, **kwargs)
