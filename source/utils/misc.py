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


import os
import os.path as op

import numpy as np
import scipy.ndimage.interpolation as sci

import settings
import file_io


def median_downsampling(img, tile_height=10, tile_width=10):
    h, w = np.shape(img)
    if w % tile_width or h % tile_height:
        raise Exception("Image dimensions must be multiple of tile dimensions.")

    n_tiles_horiz = w / tile_width
    n_tiles_vert = h / tile_height
    n_tiles = n_tiles_horiz * n_tiles_vert

    # split vertically into tiles with height=tile_height, width=img_width
    tiles_vert = np.asarray(np.split(img, n_tiles_vert, 0))  # n_tiles_vert x tile_height x w
    tiles_vert = tiles_vert.transpose([1, 0, 2]).reshape(tile_height, n_tiles_vert * w)

    # split horizontally into tiles with height=tile_height, width=tile_width
    tiles = np.asarray(np.split(tiles_vert, n_tiles, 1))
    tiles = tiles.reshape(n_tiles, tile_width * tile_height)  # n_tiles x px_per_tile

    # compute median per tile (without averaging for even N)
    tiles = np.sort(tiles, axis=1)[:, tile_width*tile_height/2]
    small_img = tiles.reshape(n_tiles_vert, n_tiles_horiz)

    return small_img


def get_mask_invalid(matrix):
    mask = np.isposinf(matrix) + np.isneginf(matrix) + np.isnan(matrix)
    return mask


def get_mask_valid(matrix):
    return ~get_mask_invalid(matrix)


def make_rgba(img, norm=False):
    if norm:
        img /= np.max(img)
    return np.dstack((img, img, img, np.ones(np.shape(img))))


def rgb2gray(img):
    n_dims = len(np.shape(img))
    if 2 == n_dims:
        return img
    elif 3 == n_dims:
        n_channels = np.shape(img)[2]
        if 3 == n_channels or 4 == n_channels:
            new_img = 0.2125*img[:, :, 0] + 0.7154*img[:, :, 1] + 0.0721*img[:, :, 2]
            new_img = np.asarray(new_img, dtype=img.dtype)
            return new_img
        else:
            raise ValueError("unexpected number of channels: %d" % n_channels)
    else:
        raise ValueError("unexpected number of dimensions: %d" % n_dims)


def resize(data, scale_factor, order=1):
    if len(np.shape(data)) == 3:
        scale_factor = [scale_factor, scale_factor, 1.0]
    return sci.zoom(data, scale_factor, order=order)


def resize_to_shape(data, height, width, order=1):
    h, w = np.shape(data)[0:2]
    factor_h = height / float(h)
    factor_w = width / float(w)
    if len(np.shape(data)) == 3:
        scale_factor = [factor_h, factor_w, 1.0]
    else:
        scale_factor = [factor_h, factor_w]
    return sci.zoom(data, scale_factor, order=order)


def percentage(total, part):
    if total == 0:
        return np.nan
    return 100 * part / float(total)


def get_scene_dict():
    from scenes import BaseScene
    scene_dict = dict()
    default_scenes = BaseScene.get_training_scenes() + BaseScene.get_stratified_scenes()
    for scene in default_scenes:
        scene_dict[scene.get_name()] = scene
    return scene_dict


# project file handling


def get_available_algo_names():
    return [a for a in os.listdir(settings.ALGO_PATH) if op.isdir(op.join(settings.ALGO_PATH, a))]


def get_path_to_algo_data(algo_name):
    return op.join(settings.ALGO_PATH, algo_name)


def get_algo_result(scene, algo_name):
    return get_algo_result_from_dir(scene, get_path_to_algo_data(algo_name))


def get_algo_result_from_dir(scene, algo_dir):
    fname = op.normpath(op.join(*[algo_dir, settings.DISP_MAP_DIR, "%s.pfm" % scene.get_name()]))
    algo_result = file_io.read_file(fname)
    if scene.gt_scale != 1:
        algo_result = sci.zoom(algo_result, scene.gt_scale, order=0)
    return algo_result


def get_runtime(scene, algo_name):
    return get_runtime_from_dir(scene, get_path_to_algo_data(algo_name))


def get_runtime_from_dir(scene, algo_dir):
    fname = op.normpath(op.join(*[algo_dir, settings.RUNTIME_DIR, "%s.txt" % scene.get_name()]))
    return file_io.read_runtime(fname)
