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

from toolkit import settings
from toolkit.metrics import BadPix, MSE, BumpinessPlanes
from toolkit.utils import misc, plotting


class StratifiedBadPix(BadPix):
    def __init__(self, thresh, name, vmin, vmax, scene_display_name, eval_on_high_res, **kwargs):
        super(StratifiedBadPix, self).__init__(name=name, vmin=vmin, vmax=vmax, thresh=thresh,
                                               eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.STRATIFIED_METRIC
        self.scene_display_name = scene_display_name

    def get_display_name(self):
        if self.name == "BadPix":
            return "%s: BadPix(%0.2f)" % (self.scene_display_name, self.thresh)

        return "%s: %s" % (self.scene_display_name, self.name)

    def get_short_name(self):
        return self.name


# --------------------------------------
# BACKGAMMON
# --------------------------------------


class BackgammonFattening(StratifiedBadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Fattening", vmin=0, vmax=22,
                 eval_on_high_res=True, **kwargs):
        super(BackgammonFattening, self).__init__(thresh=thresh, eval_on_high_res=eval_on_high_res,
                                                  name=name, vmin=vmin, vmax=vmax,
                                                  scene_display_name="Backgammon", **kwargs)

    def get_id(self):
        return ("backgammon_fattening_%0.3f" % self.thresh).replace(".", "")

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

        vis = np.ma.masked_array(plotting.adjust_binary_vis(m_fattening), mask=~mask)
        return score, vis

    @staticmethod
    def get_fattening(algo_result, gt, extrapolated_foreground):
        half_distance = 0.5 * (extrapolated_foreground + gt)  # GT + 0.5 * (FG - GT)
        with np.errstate(invalid="ignore"):
            m_fattening = (algo_result > half_distance)
        return m_fattening


class BackgammonThinning(StratifiedBadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Thinning", vmin=0, vmax=3,
                 eval_on_high_res=True, **kwargs):
        super(BackgammonThinning, self).__init__(thresh=thresh, eval_on_high_res=eval_on_high_res,
                                                 name=name, vmin=vmin, vmax=vmax,
                                                 scene_display_name="Backgammon", **kwargs)

    def get_id(self):
        return ("backgammon_thinning_%0.3f" % self.thresh).replace(".", "")

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

        m_thinning = plotting.adjust_binary_vis(m_thinning)
        vis = np.ma.masked_array(m_thinning, mask=~mask)
        return score, vis

    @staticmethod
    def get_thinning(algo_result, gt, extrapolated_background):
        half_distance = 0.5 * (extrapolated_background + gt)  # GT - 0.5 * (GT - BG)
        with np.errstate(invalid="ignore"):
            m_thinning = (algo_result < half_distance)
        return m_thinning

# --------------------------------------
# PYRAMIDS
# --------------------------------------


class PyramidsBaseBumpiness(BumpinessPlanes):

    def __init__(self, clip, factor, name, vmin, vmax, eval_on_high_res, **kwargs):
        super(PyramidsBaseBumpiness, self).__init__(clip=clip, factor=factor, name=name,
                                                    vmin=vmin, vmax=vmax,
                                                    eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.STRATIFIED_METRIC
        self.scene_display_name = "Pyramids"
        self.cmin = 0
        self.cmax = 5

    def get_display_name(self):
        return "%s: %s" % (self.scene_display_name, self.name)

    def get_short_name(self):
        return self.name


class PyramidsSlantedBumpiness(PyramidsBaseBumpiness):

    def __init__(self, clip=0.05, factor=100, name="Bump. Slanted", vmin=0, vmax=3.5,
                 eval_on_high_res=False, **kwargs):
        super(PyramidsSlantedBumpiness, self).__init__(clip=clip, factor=factor,
                                                       name=name, vmin=vmin, vmax=vmax,
                                                       eval_on_high_res=eval_on_high_res, **kwargs)

    def get_id(self):
        return ("bumpiness_slanted_%d_%0.3f" % (self.factor, self.clip)).replace(".", "")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_objects() * scene.get_boundary_mask(ignore_boundary)


class PyramidsParallelBumpiness(PyramidsBaseBumpiness):

    def __init__(self, clip=0.05, factor=100, name="Bump. Parallel", vmin=0, vmax=3.5,
                 eval_on_high_res=False, **kwargs):
        super(PyramidsParallelBumpiness, self).__init__(clip=clip, factor=factor,
                                                        name=name, vmin=vmin, vmax=vmax,
                                                        eval_on_high_res=eval_on_high_res, **kwargs)

    def get_id(self):
        return ("bumpiness_parallel_%d_%0.3f" % (self.factor, self.clip)).replace(".", "")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_plane_mask() * scene.get_boundary_mask(ignore_boundary)


# --------------------------------------
# DOTS
# --------------------------------------


class MissedDots(StratifiedBadPix):

    def __init__(self, thresh=0.4, missed_dot_bad_pix=50, name="Missed Dots",
                 vmin=0, vmax=70, eval_on_high_res=True, **kwargs):
        super(MissedDots, self).__init__(thresh=thresh, name=name, vmin=vmin, vmax=vmax,
                                         scene_display_name="Dots",
                                         eval_on_high_res=eval_on_high_res, **kwargs)
        self.missed_dot_bad_pix = missed_dot_bad_pix

    def get_id(self):
        return ("missed_dots_%d_%0.3f" % (self.missed_dot_bad_pix, self.thresh)).replace(".", "")

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
        # use only the nine biggest dots per box
        dot_labels = [dl for dl in dot_labels if 0 < dl < 9]
        n_dots = len(dot_labels)
        total_dots = n_dots * n_boxes
        detected_dots = 0

        for box_id in box_ids:
            m_box = (grid == box_id)

            for idx_d in range(n_dots):
                dot_mask = (dots_by_size == idx_d+1) * m_box
                bad_pix_on_dot = bad_pix.get_score_from_diffs(diffs[dot_mask])
                if bad_pix_on_dot < self.missed_dot_bad_pix:
                    detected_dots += 1
                else:
                    vis[dot_mask] = 1

        missed_dots = total_dots - detected_dots
        score = misc.percentage(total_dots, missed_dots)

        if not with_visualization:
            return score

        vis = plotting.adjust_binary_vis(vis)
        return score, vis


class DotsBackgroundMSE(MSE):
    def __init__(self, factor=100, name="Background MSE", vmin=0, vmax=4,
                 eval_on_high_res=True, **kwargs):
        super(DotsBackgroundMSE, self).__init__(factor=factor, name=name, vmin=vmin, vmax=vmax,
                                                eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.STRATIFIED_METRIC
        self.scene_display_name = "Dots"
        self.cmin = settings.DMIN
        self.cmax = settings.DMAX

    def get_id(self):
        return "background_mse_%d" % self.factor

    def get_display_name(self):
        return "%s: %s" % (self.scene_display_name, self.name)

    def get_short_name(self):
        return "Background"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_background_mask() * scene.get_boundary_mask(ignore_boundary)


# --------------------------------------
# STRIPES
# --------------------------------------


class StripesLowTexture(StratifiedBadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Low Texture", vmin=0, vmax=60,
                 eval_on_high_res=True, **kwargs):
        super(StripesLowTexture, self).__init__(thresh=thresh, eval_on_high_res=eval_on_high_res,
                                                name=name, vmin=vmin, vmax=vmax,
                                                scene_display_name="Stripes", **kwargs)

    def get_id(self):
        return ("low_texture_%0.3f" % self.thresh).replace(".", "")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_low_texture() * scene.get_boundary_mask(ignore_boundary)

    @staticmethod
    def eval_on_high_res():
        return True


class DarkStripes(StratifiedBadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Dark Stripes", vmin=0, vmax=60,
                 eval_on_high_res=True, **kwargs):
        super(DarkStripes, self).__init__(thresh=thresh, eval_on_high_res=eval_on_high_res,
                                          name=name, vmin=vmin, vmax=vmax,
                                          scene_display_name="Stripes", **kwargs)

    def get_id(self):
        return ("dark_stripes_%0.3f" % self.thresh).replace(".", "")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_high_contrast() * scene.get_boundary_mask(ignore_boundary)


class BrightStripes(StratifiedBadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Bright Stripes", vmin=0, vmax=60,
                 eval_on_high_res=True, **kwargs):
        super(BrightStripes, self).__init__(thresh=thresh, eval_on_high_res=eval_on_high_res,
                                            name=name, vmin=vmin, vmax=vmax,
                                            scene_display_name="Stripes", **kwargs)

    def get_id(self):
        return ("bright_stripes_%0.3f" % self.thresh).replace(".", "")

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_low_contrast() * scene.get_boundary_mask(ignore_boundary)
