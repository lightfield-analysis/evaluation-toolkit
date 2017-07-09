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

from toolkit import settings
from toolkit.scenes import BaseStratified
from toolkit.utils import plotting


class Stripes(BaseStratified):
    mn_high_contrast = "mask_high_contrast"
    mn_low_contrast = "mask_low_contrast"
    mn_low_texture = "mask_low_texture"

    def __init__(self, name="stripes", general_metrics_high_res=True, **kwargs):
        super(Stripes, self).__init__(name, general_metrics_high_res=general_metrics_high_res,
                                      **kwargs)

    @staticmethod
    def get_scene_specific_metrics():
        from toolkit.metrics import StripesLowTexture, DarkStripes, BrightStripes
        return [StripesLowTexture(), DarkStripes(), BrightStripes()]

    def visualize_masks(self, subdir="stratified"):
        self.set_high_gt_scale()

        rows, cols = 1, 3
        fig = plt.figure(figsize=(9, 3))

        center_view = self.get_center_view()
        m_eval = self.get_boundary_mask()

        plt.subplot(rows, cols, 1)
        plt.title("Low Texture")
        plotting.plot_img_with_transparent_mask(center_view,
                                                self.get_low_texture() * m_eval,
                                                **settings.mask_vis_args())

        plt.subplot(rows, cols, 2)
        plt.title("High Contrast\n(Dark Stripes)")
        plotting.plot_img_with_transparent_mask(center_view,
                                                self.get_high_contrast() * m_eval,
                                                **settings.mask_vis_args())

        plt.subplot(rows, cols, 3)
        plt.title("Low Contrast\n(Bright Stripes)")
        plotting.plot_img_with_transparent_mask(center_view,
                                                self.get_low_contrast() * m_eval,
                                                **settings.mask_vis_args())

        fig_path = plotting.get_path_to_figure("stripes_masks", subdir=subdir)
        plotting.save_tight_figure(fig, fig_path, hide_frames=True)

    # -------------------------
    # evaluation masks
    # -------------------------

    def get_high_contrast(self):
        return self.get_mask(self.mn_high_contrast)

    def get_low_contrast(self):
        return self.get_mask(self.mn_low_contrast)

    def get_low_texture(self):
        return self.get_mask(self.mn_low_texture)
