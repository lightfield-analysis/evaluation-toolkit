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


import numpy as np

from metrics import BadPix, MSE, BumpinessPlanes
import settings
from utils import misc, plotting


class StratifiedBadPix(BadPix):
    def __init__(self, descr, vmin, vmax, scene_display_name, thresh=settings.BAD_PIX_THRESH):
        super(StratifiedBadPix, self).__init__(descr=descr, vmin=vmin, vmax=vmax, thresh=thresh)
        self.category = settings.STRATIFIED
        self.scene_display_name = scene_display_name

    def get_display_name(self):
        if self.descr == "BadPix":
            return "%s: BadPix(%0.2f)" % (self.scene_display_name, self.thresh)
        return "%s: %s" % (self.scene_display_name, self.descr)

    def get_short_name(self):
        return self.descr


# --------------------------------------
# BACKGAMMON
# --------------------------------------


class BackgammonFattening(StratifiedBadPix):
    def __init__(self, descr="Fattening", vmin=0, vmax=22):
        super(BackgammonFattening, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Backgammon")

    def get_description(self):
        return "The percentage of pixels around fine structures " \
               "whose disparity estimate is closer to the foreground than to the background."

    def get_short_name(self):
        return "Fattening"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_fg_fattening_mask() * scene.get_boundary_mask(ignore_boundary)

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_fattening = self.get_fattening(algo_result, gt, scene.get_fg_extrapolation())
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_fattening)
        score = misc.percentage(np.sum(mask), np.sum(m_fattening * mask))

        if not with_visualization:
            return score
        else:
            vis = np.ma.masked_array(plotting.adjust_binary_vis(m_fattening), mask=~mask)
            return score, vis

    @staticmethod
    def get_fattening(algo_result, gt, extrapolated_foreground):
        half_distance = 0.5 * (extrapolated_foreground + gt)  # GT + 0.5 * (FG - GT)
        with np.errstate(invalid="ignore"):
            m_fattening = (algo_result > half_distance)
        return m_fattening

    @staticmethod
    def evaluate_on_high_res():
        return True


class BackgammonThinning(StratifiedBadPix):
    def __init__(self, descr="Thinning", vmin=0, vmax=3):
        super(BackgammonThinning, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Backgammon")

    def get_description(self):
        return "The percentage of pixels at fine structures " \
               "whose disparity estimate is closer to the background than to the foreground."

    def get_short_name(self):
        return "Thinning"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_fg_thinning_mask() * scene.get_boundary_mask(ignore_boundary)

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_thinning = self.get_thinning(algo_result, gt, scene.get_bg_extrapolation())
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_thinning)
        score = misc.percentage(np.sum(mask), np.sum(m_thinning * mask))

        if not with_visualization:
            return score
        else:
            m_thinning = plotting.adjust_binary_vis(m_thinning)
            vis = np.ma.masked_array(m_thinning, mask=~mask)
            return score, vis

    @staticmethod
    def get_thinning(algo_result, gt, extrapolated_background):
        half_distance = 0.5 * (extrapolated_background + gt)  # GT - 0.5 * (GT - BG)
        with np.errstate(invalid="ignore"):
            m_thinning = (algo_result < half_distance)
        return m_thinning

    @staticmethod
    def evaluate_on_high_res():
        return True

# --------------------------------------
# PYRAMIDS
# --------------------------------------


class PyramidsBaseBumpiness(BumpinessPlanes):

    def __init__(self, descr, vmax):
        super(PyramidsBaseBumpiness, self).__init__(descr=descr, vmax=vmax)
        self.category = settings.STRATIFIED
        self.scene_display_name = "Pyramids"
        self.cmin = 0
        self.cmax = 5

    def get_display_name(self):
        return "%s: %s" % (self.scene_display_name, self.descr)

    def get_short_name(self):
        return self.descr


class PyramidsSlantedBumpiness(PyramidsBaseBumpiness):

    def __init__(self, descr="Bump. Slanted", vmax=3.5):
        super(PyramidsSlantedBumpiness, self).__init__(descr=descr, vmax=vmax)

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_objects() * scene.get_boundary_mask(ignore_boundary)


class PyramidsParallelBumpiness(PyramidsBaseBumpiness):

    def __init__(self, descr="Bump. Parallel", vmax=3.5):
        super(PyramidsParallelBumpiness, self).__init__(descr=descr, vmax=vmax)

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_plane_mask() * scene.get_boundary_mask(ignore_boundary)


# --------------------------------------
# DOTS
# --------------------------------------


class MissedDots(StratifiedBadPix):

    def __init__(self, descr="Missed Dots", vmin=0, vmax=70, thresh=0.4):
        super(MissedDots, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Dots", thresh=thresh)

    def get_description(self):
        return "The percentage of dots with a BadPix(%0.2f) score > 50%%." % self.thresh

    def get_short_name(self):
        return "Missed Dots"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_boundary_mask(ignore_boundary)

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        grid = scene.get_boxes()
        dots_by_size = scene.get_dots_by_size()
        bad_pix = BadPix(thresh=self.thresh)

        vis = np.zeros(np.shape(algo_result), dtype=np.bool)
        diffs = np.abs(gt - algo_result)

        box_ids = sorted(list(np.unique(grid)))
        box_ids.remove(0)
        n_boxes = np.size(box_ids)

        dot_labels = list(np.unique(dots_by_size))
        dot_labels = [dl for dl in dot_labels if 0 < dl < 9]
        n_dots = len(dot_labels)
        total_dots = n_dots * n_boxes
        detected_dots = 0

        for idx_b, box_id in enumerate(box_ids):
            m_box = (grid == box_id)

            for idx_d in range(n_dots):
                dot_mask = (dots_by_size == idx_d+1) * m_box
                bad_pix_on_dot = bad_pix.get_score_from_diffs(diffs[dot_mask])
                if bad_pix_on_dot < 50:
                    detected_dots += 1
                else:
                    vis[dot_mask] = 1

        missed_dots = total_dots - detected_dots
        score = misc.percentage(total_dots, missed_dots)

        if not with_visualization:
            return score
        else:
            vis = plotting.adjust_binary_vis(vis)
            return score, vis

    @staticmethod
    def evaluate_on_high_res():
        return True


class DotsBackgroundMSE(MSE):
    def __init__(self, descr="Background MSE", vmin=0, vmax=4):
        super(DotsBackgroundMSE, self).__init__(descr=descr, vmin=vmin, vmax=vmax)
        self.category = settings.STRATIFIED
        self.scene_display_name = "Dots"
        self.cmin = settings.DMIN
        self.cmax = settings.DMAX

    def get_display_name(self):
        return "%s: %s" % (self.scene_display_name, self.descr)

    def get_short_name(self):
        return "Background"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_background_mask() * scene.get_boundary_mask(ignore_boundary)


# --------------------------------------
# STRIPES
# --------------------------------------


class StripesLowTexture(StratifiedBadPix):
    def __init__(self, descr="Low Texture", vmin=0, vmax=60):
        super(StripesLowTexture, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Stripes")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_low_texture() * scene.get_boundary_mask(ignore_boundary)


class DarkStripes(StratifiedBadPix):
    def __init__(self, descr="Dark Stripes", vmin=0, vmax=60):
        super(DarkStripes, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Stripes")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_high_contrast() * scene.get_boundary_mask(ignore_boundary)


class BrightStripes(StratifiedBadPix):
    def __init__(self, descr="Bright Stripes", vmin=0, vmax=60):
        super(BrightStripes, self).__init__(descr=descr, vmin=vmin, vmax=vmax, scene_display_name="Stripes")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_low_contrast() * scene.get_boundary_mask(ignore_boundary)