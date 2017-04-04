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


import os.path as op

import matplotlib.pyplot as plt
from matplotlib import ticker

from metrics import Runtime
from scenes import BaseScene
import settings
from utils.logger import log
from utils import plotting, file_io, misc


THUMB_FORMAT = "png"


def evaluate(evaluation_output_path, algorithm_input_path, ground_truth_path,
             selected_scenes=None, metrics=None, visualize=False, with_test_scenes=False):
    """
    :param evaluation_output_path: target directory for all evaluation results
    :param algorithm_input_path: input directory for algorithm results with expected directories: runtimes, disp_maps
    :param ground_truth_path: input directory for ground truth data
    :param selected_scenes: the given subset of the scenes, otherwise: all benchmark scenes
    :param metrics: the given subset of the metrics, otherwise: all applicable metrics
    :param visualize: whether to save visualizations (otherwise just the scores)
    :param with_test_scenes: whether to evaluate on the test scenes too
    :return: success, {"messages": ["error 1", "error 2", ...]}
    """

    admin_errors = []
    eval_json = dict()

    log.info("Evaluating algorithm results in: %s" % algorithm_input_path)
    log.info("Writing results to: %s" % evaluation_output_path)
    log.info("Using ground truth data from: %s" % ground_truth_path)

    # prepare metrics
    if metrics is None:
        metrics = BaseScene.get_all_metrics()
    metric_ids = [m.get_identifier() for m in metrics]
    with_runtime = Runtime().get_identifier() in metric_ids
    log.info("Metrics: %s" % ", ".join(metric_ids))

    # prepare scenes
    scenes = get_scenes_for_evaluation(selected_scenes, data_path=ground_truth_path)
    log.info("Scenes: %s" % ", ".join(s.get_display_name() for s in scenes))

    # evaluate
    for scene in scenes:
        scene_data = dict()

        try:
            if visualize:
                log.info("Visualizing algorithm result on %s" % scene.get_display_name())
                scene_data["algorithm_result"] = visualize_algo_result(scene, algorithm_input_path, evaluation_output_path)

            if with_test_scenes or not scene.is_test():
                log.info("Processing scene: %s" % scene.get_display_name())
                log.info("Using data from: %s" % scene.get_data_path())
                scene_data["scores"] = compute_scores(scene, metric_ids, algorithm_input_path, evaluation_output_path, visualize, with_runtime)

        except IOError as e:
            admin_errors.append(e)
            log.error(e)
            continue

        eval_json[scene.get_name()] = scene_data

    # save json with scores and paths to visualizations
    file_io.write_file(eval_json, op.join(evaluation_output_path, "results.json"))
    log.info("Done!")

    success = not admin_errors
    error_json = {"messages": admin_errors}
    return success, error_json


def get_scenes_for_evaluation(selected_scenes, gt_scale=1.0, data_path=None):
    # collect all potential scenes
    scenes_for_eval = BaseScene.get_benchmark_scenes(gt_scale=gt_scale, data_path=data_path)

    # select subset of selected and available scenes
    if selected_scenes is not None:
        selected_scene_names = [scene.get_name() for scene in selected_scenes]
        scenes_for_eval = [scene for scene in scenes_for_eval if scene.get_name() in selected_scene_names]

    return scenes_for_eval


def get_relative_path(scene, descr, file_type=THUMB_FORMAT):
    return "%s/%s_%s.%s" % (scene.get_type(), scene.get_name(), descr, file_type)


def visualize_algo_result(scene, algo_dir, tgt_dir):
    algo_result = misc.get_algo_result_from_dir(scene, algo_dir)

    # visualize
    fig = init_figure()
    cm = plt.imshow(algo_result, **settings.disp_map_args(scene))
    add_colorbar(cm, bins=8)

    # save fig
    relative_fname = get_relative_path(scene, "dispmap")
    fpath = op.normpath(op.join(tgt_dir, relative_fname))
    plotting.save_tight_figure(fig, fpath, hide_frames=True, remove_ticks=True, pad_inches=0.01)

    # path info for django importer
    disp_map_data = {"thumb": relative_fname}

    return disp_map_data


def compute_scores(scene, metric_ids, algo_name, tgt_dir, visualize, with_runtime):
    scores = dict()

    # resolution for evaluation is metric specific
    metrics_low_res = [m for m in scene.get_applicable_metrics_low_res() if m.get_identifier() in metric_ids]
    scene.set_low_gt_scale()
    scores = add_scores(metrics_low_res, scene, algo_name, tgt_dir, scores, visualize)

    metrics_high_res = [m for m in scene.get_applicable_metrics_high_res() if m.get_identifier() in metric_ids]
    scene.set_high_gt_scale()
    scores = add_scores(metrics_high_res, scene, algo_name, tgt_dir, scores, visualize)

    if with_runtime:
        scores = add_runtime(scene, algo_name, scores)

    return scores


def add_runtime(scene, algo_name, scores):
    metric = Runtime(log=True)
    scores[metric.get_identifier()] = {"value": metric.get_score(scene, algo_name)}
    return scores


def add_scores(metrics, scene, algo_dir, tgt_dir, scores, visualize):
    gt = scene.get_gt()
    algo_result = misc.get_algo_result_from_dir(scene, algo_dir)

    for idx_m, metric in enumerate(metrics):
        log.info("Computing score for metric: %s, scale: %0.2f" % (metric.get_display_name(), scene.gt_scale))

        if visualize:
            score, vis = metric.get_score(algo_result, gt, scene, with_visualization=True)
            relative_fname = save_visualization(algo_result, vis, metric, scene, tgt_dir)
            metric_data = {"value": score, "visualization": {"thumb": relative_fname}}
        else:
            score = metric.get_score(algo_result, gt, scene)
            metric_data = {"value": score}

        scores[metric.get_identifier()] = metric_data

    return scores


def save_visualization(algo_result, metric_vis, metric, scene, tgt_dir):
    fig = init_figure()

    # algorithm result as background
    plt.imshow(algo_result, **settings.disp_map_args(scene, cmap="gray"))

    # metric visualization on top
    if scene.hidden_gt() and metric.pixelize_results():
        metric_vis = plotting.pixelize(metric_vis, noise_factor=0.05)
    cm = plt.imshow(metric_vis, **settings.metric_args(metric))
    add_colorbar(cm, metric.colorbar_bins)

    # save fig
    relative_fname = get_relative_path(scene, metric.get_identifier())
    fpath = op.normpath(op.join(tgt_dir, relative_fname))
    plotting.save_tight_figure(fig, fpath, hide_frames=True, remove_ticks=True, pad_inches=0.01)

    return relative_fname


def add_colorbar(cm, bins, fontsize=5):
    cb = plt.colorbar(cm, shrink=0.9)
    cb.outline.set_linewidth(0)
    cb.locator = ticker.MaxNLocator(nbins=bins)
    cb.ax.tick_params(labelsize=fontsize)
    cb.update_ticks()


def init_figure():
    fig = plt.figure(figsize=(4, 2))
    return fig