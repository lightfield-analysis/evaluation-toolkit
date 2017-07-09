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
import skimage.morphology as skmorph

from toolkit.metrics import BackgammonThinning, BackgammonFattening
from toolkit.scenes import BaseStratified
from toolkit.utils import misc, plotting


class Backgammon(BaseStratified):

    mn_foreground = "mask_foreground"
    mn_background = "mask_background"
    mn_fg_thin = "mask_foreground_thinning"
    mn_fg_fat = "mask_foreground_fattening"
    mn_vertical_bins = "mask_vertical_bins"

    def __init__(self, name="backgammon", general_metrics_high_res=True, **kwargs):
        super(Backgammon, self).__init__(name, general_metrics_high_res=general_metrics_high_res,
                                         **kwargs)

    @staticmethod
    def get_scene_specific_metrics():
        return [BackgammonFattening(), BackgammonThinning()]

    def plot_fattening_thinning(self, algorithms, n_bins=15, subdir="stratified"):
        self.set_high_gt_scale()
        gt = self.get_gt()

        # prepare masks
        m_eval = self.get_boundary_mask()
        m_extrapolated_bg = self.get_bg_extrapolation()
        m_fg_thin = self.get_fg_thinning_mask() * m_eval
        m_extrapolated_fg = self.get_fg_extrapolation()
        m_fg_fat = self.get_fg_fattening_mask() * m_eval
        m_bins = self.get_vertical_bins()

        # prepare metrics
        fattening = BackgammonFattening()
        thinning = BackgammonThinning()

        fig = plt.figure(figsize=(15, 4))
        rows, cols = 1, 3

        # create visualization of vertical bins
        plt.subplot(rows, cols, 1)
        plt.title("Blue = Fattening Eval Area\n Orange = Thinning Eval Area")
        plt.xlabel("White = column separators", labelpad=5)

        alpha = 0.5
        bin_vis = np.zeros(np.shape(m_fg_thin))
        bin_vis[:, 1:] = (m_bins[:, 1:] - m_bins[:, :-1]) > 0
        bin_vis[:, 1:] += (m_bins[:, :-1] - m_bins[:, 1:]) > 0  # right most bar
        bin_vis = skmorph.binary_dilation(bin_vis, np.ones((3*self.gt_scale, 3*self.gt_scale)))

        plt.imshow(self.get_center_view(), cmap="gray")

        plt.imshow(np.ma.masked_array(m_fg_fat, mask=~m_fg_fat),
                   alpha=alpha, vmin=0.4, vmax=2, cmap="jet")

        plt.imshow(np.ma.masked_array(m_fg_thin, mask=~m_fg_thin),
                   alpha=alpha, vmin=-1, vmax=1.4, cmap="jet")

        plt.imshow(np.ma.masked_array(bin_vis, mask=~bin_vis),
                   alpha=1, vmin=0, vmax=1, cmap="gray")

        plt.yticks([])
        plt.xticks([])

        # compute scores for vertical bins
        x_values = np.arange(0, n_bins, 1)
        for algorithm in algorithms:
            algo_result = misc.get_algo_result(algorithm, self)
            props = {"color": algorithm.get_color(), "lw": 2,
                     "alpha": 0.8, "markersize": 7, "markeredgewidth": 0}

            plt.subplot(rows, cols, 2)
            mask_fattening = fattening.get_fattening(algo_result, gt, m_extrapolated_fg) * m_eval
            y_values_fat = self.get_bin_scores(x_values, m_bins, n_bins, m_fg_fat, mask_fattening)
            plt.plot(x_values, y_values_fat, "o-", **props)

            plt.subplot(rows, cols, 3)
            mask_thinning = thinning.get_thinning(algo_result, gt, m_extrapolated_bg) * m_eval
            y_values_thin = self.get_bin_scores(x_values, m_bins, n_bins, m_fg_thin, mask_thinning)
            plt.plot(x_values, y_values_thin, "o-", label=algorithm.get_display_name(), **props)

        for idx_m, metric in enumerate([fattening, thinning]):
            plt.subplot(rows, cols, idx_m+2)
            plt.xlabel("%d columns from left to right" % n_bins)
            plt.ylabel(metric.get_short_name(), labelpad=-5)
            plt.title(metric.get_display_name())
            plotting.hide_upper_right()
            plt.ylim([0, 105])
            plt.xlim([0, n_bins - 0.5])

        plt.legend(frameon=False, title="Algorithms:", bbox_to_anchor=(1.57, 1.1))
        fig_path = plotting.get_path_to_figure("backgammon_fattening_thinning", subdir=subdir)
        plotting.save_fig(fig, fig_path)

    @staticmethod
    def get_bin_scores(x_values, m_bins, n_bins, m_eval, m_algo_result):
        y_values = np.full(np.shape(x_values), fill_value=np.nan)
        for i in range(n_bins):
            mask = (m_bins == i+1) * m_eval
            y_values[i] = misc.percentage(np.sum(mask), np.sum(m_algo_result * mask))

        return y_values

    # -------------------------
    # evaluation masks
    # -------------------------

    def get_fg_extrapolation(self):
        fg_extr = np.zeros(self.get_shape(), dtype=np.float)
        fg_extr[:, :] = self.get_gt()[:, int(14*self.gt_scale):int(14*self.gt_scale)+1]
        return fg_extr

    def get_bg_extrapolation(self):
        bg_extr = np.zeros(self.get_shape(), dtype=np.float)
        bg_extr[:, :] = self.get_gt()[:, self.get_width()-11:self.get_width()-10]
        return bg_extr

    def get_fg_fattening_mask(self):
        return self.get_mask(self.mn_fg_fat)

    def get_fg_thinning_mask(self):
        return self.get_mask(self.mn_fg_thin)

    def get_foreground_mask(self):
        return self.get_mask(self.mn_foreground)

    def get_background_mask(self):
        return self.get_mask(self.mn_background)

    def get_vertical_bins(self):
        return self.get_mask(self.mn_vertical_bins, binary=False)
