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

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

base_path = os.getcwd()
DATA_PATH = op.normpath(op.join(base_path, "../data"))
ALGO_PATH = op.normpath(op.join(base_path, "../algo_results"))
EVAL_PATH = op.normpath(op.join(base_path, "../evaluation"))
ALGO_EVAL_PATH = op.normpath(op.join(EVAL_PATH, "algorithms"))
FIG_PATH = op.normpath(op.join(EVAL_PATH, "figures"))
TMP_PATH = op.normpath(op.join(base_path, "../tmp"))

PATH_TO_ALGO_META_DATA = op.normpath(op.join(ALGO_PATH, "meta_data.json"))

HEIGHT = 512
WIDTH = 512
BAD_PIX_THRESH = 0.07

FIG_SIZE_EVALUATION = (8, 4)

DISP_MAP_DIR = "disp_maps"
RUNTIME_DIR = "runtimes"

STRATIFIED_METRIC = "Stratified"
PHOTOREALISTIC_METRIC = "Photorealistic"
GENERAL_METRIC = "General"

LOWRES = "lowres"
HIGHRES = "highres"
PIXELIZE = True
USE_TEST_SCENE_GT = False

TEST = "test"
TRAINING = "training"
ADDITIONAL = "additional"
STRATIFIED = "stratified"
BENCHMARK = "benchmark"  # test + training + stratified
OTHER = "other"

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


def get_scene_names_test():
    return ["bedroom", "bicycle", "herbs", "origami"]


def get_scene_names_training():
    return ["boxes", "cotton", "dino", "sideboard"]


def get_scene_names_stratified():
    return ["backgammon", "dots", "pyramids", "stripes"]


def get_scene_names_additional():
    return ["antinous", "boardgames", "dishes", "greek",
            "kitchen", "medieval2", "museum", "pens",
            "pillows", "platonic", "rosemary", "table",
            "tomb", "tower", "town", "vinyl"]


def diff_map_args(vmin=DMIN, vmax=DMAX):
    return {"vmin": vmin,
            "vmax": vmax,
            "interpolation": "none",
            "cmap": error_cmap}


def abs_diff_map_args(vmin=0, vmax=0.1):
    return {"vmin": vmin,
            "vmax": vmax,
            "interpolation": "none",
            "cmap": cm.YlOrRd}


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

colors = [
    ( 31, 119, 180),  # blue
    (255, 127,  14),  # orange
    ( 44, 160,  44),  # green
    (214,  39,  40),  # red
    (148, 103, 189),  # violet
    (140,  86,  75),  # brown
    (227, 119, 194),  # rose
    (127, 127, 127),  # grey
    (188, 189,  34),  # yellow green
    ( 23, 190, 207),  # cyan
    (153,   0, 153),  # magenta
    (255, 210,   0),  # yellow
    (152, 223, 138),  # light lime green
    ( 46,   6, 224),  # purple
    (196, 156, 148),  # misty rose
    (247, 182, 210),  # rose
    (199, 199, 199),  # light grey
    (219, 219, 141),  # light green
    (158, 218, 229),  # light blue
           ]


def make_color(color):
    # scale RGB values to [0, 1] range
    return tuple([c/255.0 for c in color])


def get_color(idx):
    return make_color(colors[idx % len(colors)])

