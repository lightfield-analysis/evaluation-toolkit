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


import copy
import os
import os.path as op
from random import randint

import matplotlib.cm as cm

base_path = os.getcwd()
DATA_PATH = op.normpath(op.join(base_path, "../data"))
ALGO_PATH = op.normpath(op.join(base_path, "../algo_results"))
EVAL_PATH = op.normpath(op.join(base_path, "../evaluation"))

HEIGHT = 512
WIDTH = 512
BAD_PIX_THRESH = 0.07

DISP_MAP_DIR = "disp_maps"
RUNTIME_DIR = "runtimes"

STRATIFIED = "Stratified"
PHOTOREALISTIC = "Photorealistic"
GENERAL = "General"

LOWRES = "lowres"
HIGHRES = "highres"


# plotting properties
DMIN = -0.2
DMAX = 0.2

fig_type = "png"
color_invalid = [1, 1, 1]
color_mask = tuple([c / 255.0 for c in (23, 190, 207)])

fontsize = 18
disp_cmap = copy.deepcopy(cm.viridis)
error_cmap = copy.deepcopy(cm.seismic)
abs_error_cmap = copy.deepcopy(cm.RdYlGn_r)
mask_cmap = copy.deepcopy(cm.seismic)

quantile_cmap = copy.deepcopy(cm.YlOrRd)
quantile_cmap.set_under(color=(0.5, 0.5, 0.5), alpha=1.0)

tableau20 = [(255, 127, 14),  # orange
             (44, 160, 44),  # green
             (148, 103, 189),  # purple
             (23, 190, 207),  # cyan
             (90, 80, 60),  # black
             (31, 119, 180),  # blue
             (140, 86, 75),  # brown
             (214, 39, 40),  # red
             (152, 223, 138),  # light green
             (174, 199, 232),  # light blue
             (255, 152, 150),  # rose
             (127, 127, 127),  # grey
             (219, 219, 141)  # olive
             ]


def get_algo_names_accv_paper():
    return ["epi1", "epi2", "lf_occ", "lf", "mv"]


def get_algo_display_name(algo_name):
    return algo_name.upper()


def diff_map_args():
    return {"vmin": DMIN,
            "vmax": DMAX,
            "interpolation": "none",
            "cmap": error_cmap}


def disp_map_args(scene, factor=0.9, cmap=disp_cmap):
    return {"vmin": scene.disp_min * factor,
            "vmax": scene.disp_max * factor,
            "interpolation": "none",
            "cmap": cmap}


def metric_args(metric):
    return {"vmin": metric.cmin,
            "vmax": metric.cmax,
            "interpolation": "none",
            "cmap": metric.cmap}


def mask_vis_args(alpha=0.8):
    return {"color": color_mask,
            "alpha": alpha}


def score_color_args(vmin, vmax, alpha=0.8):
    return {"vmin": vmin,
            "vmax": vmax,
            "alpha": alpha,
            "cmap": abs_error_cmap}


def get_algo_color(algo_prefix):
    color_indices = {
              "lf": 0,
              "lf_occ": 1,
              "epi2": 2,
              "epi1": 3,
              "mv": 4
              }
    algo_index = color_indices.get(algo_prefix, randint(len(color_indices.keys()), len(tableau20)))
    color = make_color(tableau20[algo_index % len(tableau20)])
    return color


def make_color(color):
    # scale RGB values to [0, 1] range
    return tuple([c/255.0 for c in color])
