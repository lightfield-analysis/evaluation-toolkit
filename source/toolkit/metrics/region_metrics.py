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
import skimage.filters as skf

from toolkit import settings
from toolkit.metrics import BadPix, BaseMetric
from toolkit.utils import misc, plotting


class Discontinuities(BadPix):
    def __init__(self, thresh=settings.BAD_PIX_THRESH, name="Discontinuities",
                 eval_on_high_res=True, **kwargs):
        super(Discontinuities, self).__init__(name=name, thresh=thresh,
                                              eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_discontinuities"

    def get_id(self):
        return ("discontinuities_%0.3f" % self.thresh).replace(".", "")

    def get_description(self):
        return "The percentage of pixels at discontinuity regions " \
               "with abs(gt - algo) > %0.2f." % self.thresh

    def get_short_name(self):
        return "Discont."

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_mask(self.mask_name) * scene.get_boundary_mask(ignore_boundary)


class BumpinessPlanes(BaseMetric):
    def __init__(self, clip=0.05, factor=100, name="Bumpiness Planes",
                 vmin=0, vmax=5, eval_on_high_res=False):
        super(BumpinessPlanes, self).__init__(name=name, vmin=vmin, vmax=vmax,
                                              colorbar_bins=5, cmap=settings.CMAP_DISP,
                                              eval_on_high_res=eval_on_high_res)
        self.clip = clip
        self.factor = factor
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_planes"

    def get_id(self):
        return ("bumpiness_planes_%d_%0.3f" % (self.factor, self.clip)).replace(".", "")

    def get_description(self):
        return "The average Frobenius norm of the Hessian matrix of (gt - algo) " \
               "at the given plane regions, multiplied with %d." % self.factor

    def get_legend(self):
        return "purple = smooth, yellow = bumpy"

    def get_short_name(self):
        return "Planes"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_mask(self.mask_name) * scene.get_boundary_mask(ignore_boundary)

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        bumpiness = self.get_bumpiness(gt, algo_result)
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(bumpiness)
        score = self.factor * np.sum(bumpiness[mask]) / float(np.sum(mask))

        if not with_visualization:
            return score

        vis = np.ma.masked_array(bumpiness * self.factor, mask=~mask)
        return score, vis

    def get_bumpiness(self, gt, algo_result):
        # Frobenius norm of the Hesse matrix
        diff = np.asarray(algo_result - gt, dtype='float64')
        dx = skf.scharr_v(diff)
        dy = skf.scharr_h(diff)
        dxx = skf.scharr_v(dx)
        dxy = skf.scharr_h(dx)
        dyy = skf.scharr_h(dy)
        dyx = skf.scharr_v(dy)
        bumpiness = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
        bumpiness = np.clip(bumpiness, 0, self.clip)
        return bumpiness


class BumpinessContinSurf(BumpinessPlanes):

    def __init__(self, clip=0.05, factor=100, name="Bumpiness Contin. Surfaces",
                 eval_on_high_res=False, **kwargs):
        super(BumpinessContinSurf, self).__init__(clip=clip, factor=factor, name=name,
                                                  eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_smooth_surfaces"

    def get_id(self):
        return ("bumpiness_contin_surfaces_%d_%0.3f" % (self.factor, self.clip)).replace(".", "")

    def get_description(self):
        return "The average Frobenius norm of the Hessian matrix of (gt - algo) " \
               "at smooth non-planar regions, multiplied with %d." % self.factor

    def get_short_name(self):
        return "Surfaces"


class MAEPlanes(BaseMetric):
    def __init__(self, name="MAE Planes", vmin=0, vmax=80, eval_on_high_res=False):
        super(MAEPlanes, self).__init__(name=name, vmin=vmin, vmax=vmax,
                                        eval_on_high_res=eval_on_high_res,
                                        colorbar_bins=5, cmap=settings.CMAP_ABS_ERROR)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_planes"

    def get_description(self):
        return "The median angular error of the surface normals at the given plane regions."

    def get_legend(self):
        return "green = good, red = bad"

    def get_short_name(self):
        return "MAE Planes"

    def get_id(self):
        return "mae_planes"

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_mask(self.mask_name) * scene.get_boundary_mask(ignore_boundary)

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        mask = self.get_evaluation_mask(scene)
        return self.get_score_from_mask(algo_result, gt, scene, mask,
                                        with_visualization=with_visualization)

    def get_score_from_mask(self, algo_result, gt, scene, mask, with_visualization=False):
        angular_error = self.get_angular_error(algo_result, gt, scene)
        mask = mask * misc.get_mask_valid(algo_result) * misc.get_mask_valid(angular_error)
        score = np.median(angular_error[mask])

        if not with_visualization:
            return score

        vis = np.ma.masked_array(angular_error, mask=~mask)
        return score, vis

    @staticmethod
    def get_angular_error(algo_result, gt, scene):
        algo_normals = scene.get_depth_normals(scene.disp2depth(algo_result))
        gt_normals = scene.get_depth_normals(scene.disp2depth(gt))

        ssum = np.sum(algo_normals * gt_normals, axis=2)
        ssum = np.clip(ssum, -1., 1.)
        angular_error = np.degrees(np.arccos(ssum))

        return angular_error


class MAEContinSurf(MAEPlanes):

    def __init__(self, name="MAE Contin. Surfaces", eval_on_high_res=False, **kwargs):
        super(MAEContinSurf, self).__init__(name=name, eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_smooth_surfaces"

    def get_description(self):
        return "The median angular error of the surface normals at smooth, non-planar regions."

    def get_short_name(self):
        return "MAE Surfaces"

    def get_id(self):
        return "mae_contin_surfaces"


class FineFattening(BadPix):
    def __init__(self, thresh=-0.15, name="Fine Fattening", eval_on_high_res=True, **kwargs):
        super(FineFattening, self).__init__(thresh=thresh, name=name,
                                            eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_fine_surrounding"

    def get_id(self):
        return ("fine_fattening_%0.3f" % abs(self.thresh)).replace(".", "")

    def get_short_name(self):
        return "Fine Fat"

    def get_description(self):
        return "The percentage of pixels around fine structures " \
               "with (gt - algo) < %0.2f." % self.thresh

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_fattening = self.get_fattening(algo_result, gt)
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_fattening)
        score = misc.percentage(np.sum(mask), np.sum(m_fattening * mask))

        if not with_visualization:
            return score

        vis = np.ma.masked_array(plotting.adjust_binary_vis(m_fattening), mask=~mask)
        return score, vis

    def get_fattening(self, algo_result, gt):
        with np.errstate(invalid="ignore"):
            m_fattening = (gt - algo_result) < self.thresh
        return m_fattening

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_mask(self.mask_name) * scene.get_boundary_mask(ignore_boundary)


class FineThinning(BadPix):
    def __init__(self, thresh=0.15, name="Fine Thinning", eval_on_high_res=True, **kwargs):
        super(FineThinning, self).__init__(thresh=thresh, name=name,
                                           eval_on_high_res=eval_on_high_res, **kwargs)
        self.category = settings.PHOTOREALISTIC_METRIC
        self.mask_name = "mask_fine"

    def get_id(self):
        return ("fine_thinning_%0.3f" % self.thresh).replace(".", "")

    def get_short_name(self):
        return "Fine Thin"

    def get_description(self):
        return "The percentage of pixels at fine structures " \
               "with (gt - algo) > %0.2f." % self.thresh

    def get_score(self, algo_result, gt, scene, with_visualization=False):
        m_thinning = self.get_thinning(algo_result, gt)
        mask = self.get_evaluation_mask(scene) * misc.get_mask_valid(m_thinning)
        score = misc.percentage(np.sum(mask), np.sum(m_thinning * mask))

        if not with_visualization:
            return score

        vis = np.ma.masked_array(plotting.adjust_binary_vis(m_thinning), mask=~mask)
        return score, vis

    def get_thinning(self, algo_result, gt):
        with np.errstate(invalid="ignore"):
            mask_thinning = (gt - algo_result) > self.thresh
        return mask_thinning

    def get_evaluation_mask(self, scene, ignore_boundary=True):
        return scene.get_mask(self.mask_name) * scene.get_boundary_mask(ignore_boundary)
