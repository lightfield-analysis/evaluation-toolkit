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


from toolkit import settings
from toolkit.utils import file_io, log


def get_mask_invalid(matrix):
    mask = np.isposinf(matrix) + np.isneginf(matrix) + np.isnan(matrix)
    return mask


def get_mask_valid(matrix):
    return ~get_mask_invalid(matrix)


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

def get_available_scenes_by_category(categories=None, data_path=settings.DATA_PATH):

    if categories is None:
        categories = [d for d in os.listdir(data_path) if op.isdir(op.join(data_path, d))]

    scenes_by_category = dict()
    for category in categories:
        category_dir = op.join(data_path, category)

        # at least parameter file is required for scene to be "available"
        scene_names = [d for d in os.listdir(category_dir) if
                       op.isfile(op.join(category_dir, d, "parameters.cfg"))]
        scenes_by_category[category] = scene_names

    return scenes_by_category


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
    from toolkit.scenes import PhotorealisticScene
    return PhotorealisticScene(name=scene_name, category=category,
                               data_path=data_path, gt_scale=gt_scale)


def get_stratified_scene(scene_name, gt_scale=1.0, data_path=None):
    from toolkit.scenes import Backgammon, Pyramids, Dots, Stripes
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

def get_all_metrics():
    from toolkit.metrics import Runtime
    return get_all_metrics_wo_runtime() + [Runtime(log=True), Runtime(log=False)]


def get_all_metrics_wo_runtime():
    all_without_runtime = get_general_metrics() + get_stratified_metrics() + get_region_metrics()
    return all_without_runtime


def get_general_metrics():
    from toolkit.metrics import MSE, BadPix, Quantile
    return [MSE(), BadPix(0.01), BadPix(0.03), BadPix(0.07), Quantile(25)]


def get_region_metrics():
    from toolkit.metrics import Discontinuities, FineFattening, FineThinning, \
        BumpinessPlanes, BumpinessContinSurf, MAEPlanes, MAEContinSurf

    return [BumpinessPlanes(), BumpinessContinSurf(), MAEPlanes(), MAEContinSurf(),
            Discontinuities(), FineFattening(), FineThinning()]


def get_stratified_metrics():
    from toolkit.scenes import Backgammon, Pyramids, Dots, Stripes
    metrics = []
    for scene in [Backgammon, Pyramids, Dots, Stripes]:
        metrics += scene.get_scene_specific_metrics()
    return metrics


def get_metric_groups_by_name():
    metrics = {
        "general": get_general_metrics(),
        "stratified": get_stratified_metrics(),
        "regions": get_region_metrics(),
        "all_wo_runtime": get_all_metrics_wo_runtime(),
        "all": get_all_metrics()
    }
    return metrics


# scores

def collect_scores(algorithms, scenes, metrics, masked=False):
    scores_scenes_metrics_algos = np.full((len(scenes), len(metrics), len(algorithms)),
                                          fill_value=np.nan)

    for idx_a, algorithm in enumerate(algorithms):
        fname_json = op.join(settings.ALGO_EVAL_PATH, algorithm.get_name(), "results.json")

        try:
            results = file_io.read_file(fname_json)
        except IOError:
            log.error("Could not find scores at: %s. \n"
                      "Please execute 'run_evaluation.py' with the algorithms, scenes and metrics "
                      "that you want to use in your figure." % fname_json)
            exit()

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


# project file handling: algorithms


def get_available_algo_names():
    return [a for a in os.listdir(settings.ALGO_PATH) if op.isdir(op.join(settings.ALGO_PATH, a))]


def get_path_to_algo_data(algorithm):
    return op.join(settings.ALGO_PATH, algorithm.get_name())


def get_fname_algo_result(algo_dir, scene):
    fname = op.normpath(op.join(algo_dir, settings.DIR_NAME_DISP_MAPS, "%s.pfm" % scene.get_name()))
    return fname


def save_algo_result(algo_result, algorithm, scene):
    fname = get_fname_algo_result(get_path_to_algo_data(algorithm), scene)
    file_io.write_file(algo_result, fname)


def get_algo_result(algorithm, scene):
    return get_algo_result_from_dir(get_path_to_algo_data(algorithm), scene)


def get_algo_result_from_dir(algo_dir, scene):
    fname = get_fname_algo_result(algo_dir, scene)
    algo_result = file_io.read_file(fname)
    if scene.gt_scale != 1:
        algo_result = sci.zoom(algo_result, scene.gt_scale, order=0)
    return algo_result


def get_algo_results(algorithms, scene):
    algo_results = np.full((scene.get_height(), scene.get_width(), len(algorithms)),
                           fill_value=np.nan)

    for idx_a, algorithm in enumerate(algorithms):
        algo_results[:, :, idx_a] = get_algo_result(algorithm, scene)

    algo_results = np.ma.masked_array(algo_results, mask=get_mask_invalid(algo_results))
    return algo_results


# project file handling: runtimes


def get_fname_runtime(algo_dir, scene):
    fname = op.normpath(op.join(algo_dir, settings.DIR_NAME_RUNTIMES, "%s.txt" % scene.get_name()))
    return fname


def get_runtime(algorithm, scene):
    return get_runtime_from_dir(get_path_to_algo_data(algorithm), scene)


def get_runtime_from_dir(algo_dir, scene):
    fname = get_fname_runtime(algo_dir, scene)
    return file_io.read_runtime(fname)


def get_runtimes(algorithms, scene):
    runtimes = []
    for algorithm in algorithms:
        runtimes.append(get_runtime(algorithm, scene))
    return runtimes


def save_runtime(runtime, algorithm, scene):
    fname = get_fname_runtime(get_path_to_algo_data(algorithm), scene)
    file_io.write_runtime(runtime, fname)
