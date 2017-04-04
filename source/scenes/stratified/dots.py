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

import settings
from metrics import DotsBackgroundMSE, MissedDots, MSE
from scenes import BaseStratified, BaseScene
from utils import plotting, misc


class Dots(BaseStratified):

    mn_background = "mask_background"
    mn_dots_by_size = "mask_dots_by_size"
    mn_boxes = "mask_boxes"

    def __init__(self, img_name="dots", **kwargs):
        super(Dots, self).__init__(img_name, **kwargs)

    @staticmethod
    def get_applicable_metrics_high_res():
        return BaseScene.get_general_metrics() + Dots.get_overview_metrics()

    @staticmethod
    def get_applicable_metrics_low_res():
        return []

    @staticmethod
    def get_overview_metrics():
        return [DotsBackgroundMSE(), MissedDots()]

    def plot_error_vs_noise(self, algo_names):
        self.set_low_gt_scale()
        fig = plt.figure(figsize=(8, 4))

        grid = self.get_boxes()
        box_ids = sorted(list(np.unique(grid)))
        box_ids.remove(0)
        n_boxes = len(box_ids)
        mse = MSE()
        gt = self.get_gt()
        m_basic = self.get_boundary_mask()
        m_eval = self.get_background_mask() * m_basic

        x_values = np.arange(1, n_boxes + 1)

        for idx_a, algo_name in enumerate(algo_names):
            algo_result = misc.get_algo_result(self, algo_name)
            y_values = np.full(n_boxes, fill_value=np.nan)

            for idx_b, box_id in enumerate(box_ids):
                m_current = m_eval * (grid == box_id)
                y_values[idx_b] = mse.get_masked_score(algo_result, gt, m_current)

            color = settings.get_algo_color(algo_name)
            plt.plot(x_values, y_values, "o-", color=color, label=settings.get_algo_display_name(algo_name),
                     lw=2, alpha=0.9, markeredgewidth=0)

        plt.legend(frameon=False, loc="upper right", ncol=1,
                   title="Algorithms:", bbox_to_anchor=(1.25, 1), borderaxespad=0.0)
        plt.xlabel("Cell IDs (increasing noise from left to right)")
        plt.ylabel("MSE on cell background")
        plt.title("%s: Error per Cell Background" % (self.get_display_name()))
        plotting.hide_upper_right()

        fig_path = plotting.get_path_to_figure("dots_per_box")
        plotting.save_tight_figure(fig, fig_path, remove_ticks=False)

    # -------------------------
    # evaluation masks
    # -------------------------

    def get_background_mask(self):
        return self.get_mask(self.mn_background)

    def get_boxes(self):
        return self.get_mask(self.mn_boxes, binary=False)

    def get_dots_by_size(self):
        return self.get_mask(self.mn_dots_by_size, binary=False)



