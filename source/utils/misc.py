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
from utils import file_io


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
    if n_dims == 2:
        return img
    elif n_dims == 3:
        n_channels = np.shape(img)[2]
        if n_channels == 3 or n_channels == 4:
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


# scenes

def get_available_scenes_with_categories(categories=None, data_path=settings.DATA_PATH):
    if categories is None:
        categories = [d for d in os.listdir(data_path) if op.isdir(op.join(data_path, d))]

    scenes_to_categories = dict()
    for category in categories:
        category_dir = op.join(data_path, category)

        # at least parameter file is required for scene to be "available"
        scene_names = [d for d in os.listdir(category_dir) if
                       op.isfile(op.join(category_dir, d, "parameters.cfg"))]

        for scene_name in scene_names:
            if scene_name not in scenes_to_categories:
                scenes_to_categories[scene_name] = category
            else:
                raise Exception("Scene names must be unique across all categories. "
                                "Found duplicate for %s." % scene_name)

    return scenes_to_categories


def infer_scene_category(scene_name):
    if scene_name in settings.get_scene_names_stratified():
        category = settings.STRATIFIED
    elif scene_name in settings.get_scene_names_training():
        category = settings.TRAINING
    elif scene_name in settings.get_scene_names_test():
        category = settings.TEST
    elif scene_name in settings.get_scene_names_additional():
        category = settings.ADDITIONAL
    else:
        category = settings.OTHER
    return category


def get_benchmark_scenes(gt_scale=1.0, data_path=None):
    return get_stratified_scenes(gt_scale, data_path) + \
           get_training_scenes(gt_scale, data_path) + \
           get_test_scenes(gt_scale, data_path)


def get_training_scenes(gt_scale=1.0, data_path=None):
    return _get_photorealistic_scenes_by_name(settings.get_scene_names_training(),
                                              settings.TRAINING, gt_scale, data_path)


def get_test_scenes(gt_scale=1.0, data_path=None):
    return _get_photorealistic_scenes_by_name(settings.get_scene_names_test(),
                                              settings.TEST, gt_scale, data_path)


def get_additional_scenes(gt_scale=1.0, data_path=None):
    return _get_photorealistic_scenes_by_name(settings.get_scene_names_additional(),
                                              settings.ADDITIONAL, gt_scale, data_path)


def _get_photorealistic_scenes_by_name(scene_names, category, gt_scale=1.0, data_path=None):
    scenes = [get_photorealistic_scene(s, category, gt_scale, data_path) for s in scene_names]
    return scenes


def get_stratified_scenes(gt_scale=1.0, data_path=None):
    scene_names = settings.get_scene_names_stratified()
    scenes = [get_stratified_scene(scene_name, gt_scale, data_path) for scene_name in scene_names]
    return scenes


def get_scene(scene_name, category, gt_scale=1.0, data_path=None):
    if category == settings.STRATIFIED:
        scene = get_stratified_scene(scene_name, gt_scale, data_path)
    else:
        scene = get_photorealistic_scene(scene_name, category, gt_scale, data_path)
    return scene


def get_photorealistic_scene(scene_name, category, gt_scale=1.0, data_path=None):
    from scenes import PhotorealisticScene
    return PhotorealisticScene(name=scene_name, category=category,
                               data_path=data_path, gt_scale=gt_scale)


def get_stratified_scene(scene_name, gt_scale=1.0, data_path=None):
    from scenes import Backgammon, Pyramids, Dots, Stripes
    if scene_name == "backgammon":
        scene = Backgammon(data_path=data_path, gt_scale=gt_scale)
    elif scene_name == "pyramids":
        scene = Pyramids(data_path=data_path, gt_scale=gt_scale)
    elif scene_name == "dots":
        scene = Dots(data_path=data_path, gt_scale=gt_scale)
    elif scene_name == "stripes":
        scene = Stripes(data_path=data_path, gt_scale=gt_scale)
    else:
        raise Exception("Unknown stratified scene: %s." % scene_name)
    return scene


# metrics

def get_all_metrics(log_runtime=True):
    from metrics import Runtime
    return get_all_metrics_wo_runtime() + [Runtime(log=log_runtime)]


def get_all_metrics_wo_runtime():
    all_without_runtime = get_general_metrics() + get_stratified_metrics() + get_region_metrics()
    return all_without_runtime


def get_general_metrics():
    from metrics import MSE, BadPix, Quantile
    return [MSE(), BadPix(0.01), BadPix(0.03), BadPix(0.07), Quantile(25)]


def get_region_metrics():
    from metrics import Discontinuities, FineFattening, FineThinning, \
        BumpinessPlanes, BumpinessContinSurf, MAEPlanes, MAEContinSurf

    return [BumpinessPlanes(), BumpinessContinSurf(), MAEPlanes(), MAEContinSurf(),
            Discontinuities(), FineFattening(), FineThinning()]


def get_stratified_metrics():
    from scenes import Backgammon, Pyramids, Dots, Stripes
    metrics = []
    for scene in [Backgammon, Pyramids, Dots, Stripes]:
        metrics += scene.get_scene_specific_metrics()
    return metrics


# scores
def collect_scores(algorithms, scenes, metrics, masked=False):
    scores_scenes_metrics_algos = np.full((len(scenes), len(metrics), len(algorithms)),
                                          fill_value=np.nan)

    for idx_a, algorithm in enumerate(algorithms):
        fname_json = op.join(settings.ALGO_EVAL_PATH, algorithm.get_name(), "results.json")
        results = file_io.read_file(fname_json)

        for idx_s, scene in enumerate(scenes):
            scene_scores = results[scene.get_name()]["scores"]

            for idx_m, metric in enumerate(metrics):
                metric_score = scene_scores.get(metric.get_id(), None)

                if metric_score is not None:
                    scores_scenes_metrics_algos[idx_s, idx_m, idx_a] = metric_score["value"]

    if masked:
        mask = get_mask_invalid(scores_scenes_metrics_algos)
        scores_scenes_metrics_algos = np.ma.masked_array(scores_scenes_metrics_algos, mask=mask)

    return scores_scenes_metrics_algos


# project file handling


def get_available_algo_names():
    return [a for a in os.listdir(settings.ALGO_PATH) if op.isdir(op.join(settings.ALGO_PATH, a))]


def get_path_to_algo_data(algorithm):
    return op.join(settings.ALGO_PATH, algorithm.get_name())


def save_algo_result(algo_result, scene, algorithm):
    fname = op.normpath(op.join(get_path_to_algo_data(algorithm),
                                settings.DIR_NAME_DISP_MAPS,
                                "%s.pfm" % scene.get_name()))
    file_io.write_file(algo_result, fname)


def get_algo_result(scene, algorithm):
    return get_algo_result_from_dir(scene, get_path_to_algo_data(algorithm))


def get_algo_result_from_dir(scene, algo_dir):
    fname = op.normpath(op.join(algo_dir,
                                settings.DIR_NAME_DISP_MAPS,
                                "%s.pfm" % scene.get_name()))
    algo_result = file_io.read_file(fname)
    if scene.gt_scale != 1:
        algo_result = sci.zoom(algo_result, scene.gt_scale, order=0)
    return algo_result


def get_algo_results(scene, algorithms):
    algo_results = np.full((scene.get_height(), scene.get_width(), len(algorithms)),
                           fill_value=np.nan)

    for idx_a, algorithm in enumerate(algorithms):
        algo_results[:, :, idx_a] = get_algo_result(scene, algorithm)

    algo_results = np.ma.masked_array(algo_results, mask=get_mask_invalid(algo_results))
    return algo_results


def get_runtime(scene, algorithm):
    return get_runtime_from_dir(scene, get_path_to_algo_data(algorithm))


def get_runtime_from_dir(scene, algo_dir):
    fname = op.normpath(op.join(algo_dir,
                                settings.DIR_NAME_RUNTIMES,
                                "%s.txt" % scene.get_name()))
    return file_io.read_runtime(fname)


def get_runtimes(scene, algorithms):
    runtimes = []
    for algorithm in algorithms:
        runtimes.append(get_runtime(scene, algorithm))
    return runtimes


def save_runtime(runtime, scene, algorithm):
    fname = op.normpath(op.join(get_path_to_algo_data(algorithm),
                                settings.DIR_NAME_RUNTIMES,
                                "%s.txt" % scene.get_name()))
    file_io.write_runtime(runtime, fname)


def get_stacked_gt(scene, n):
    return np.tile(scene.get_gt()[:, :, np.newaxis], (1, 1, n))
