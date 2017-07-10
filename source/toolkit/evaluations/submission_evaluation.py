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


import shutil
import os.path as op

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from toolkit import settings
from toolkit.utils import file_io, log, misc, plotting


def evaluate(evaluation_output_path, algorithm_input_path, scenes, metrics,
             visualize=False, add_to_existing_results=True, add_pfms_to_result=True):
    """
    :param evaluation_output_path: target directory for all evaluation results
    :param algorithm_input_path: input directory for algorithm results,
                                 expected directories: runtimes, disp_maps
    :param scenes: scenes to be evaluated
    :param metrics: metrics to be evaluated
    :param visualize: whether to save visualizations (otherwise just the scores)
    :param add_to_existing_results: if set to True, will try to read results.json and add/replace entries,
                                    keeping existing scores of other scenes/metrics as is
    :param add_pfms_to_result: when executed on evaluation server, pfms are prepared for 3D point cloud view
    :return: success, {"messages": ["error 1", "error 2", ...]}
    """

    log.info("Evaluating algorithm results in:\n  %s" % algorithm_input_path)
    log.info("Writing results to:\n  %s" % evaluation_output_path)
    log.info("Using ground truth data from:\n  %s" % settings.DATA_PATH)
    log.info("Metrics:\n  %s" % ", ".join(m.get_display_name() for m in metrics))
    log.info("Scenes:\n  %s" % ", ".join(s.get_display_name() for s in scenes))

    file_name_results = op.join(evaluation_output_path, "results.json")
    admin_errors = []

    eval_json = dict()
    if add_to_existing_results:
        try:
            eval_json = file_io.read_file(file_name_results)
        except IOError:
            pass

    # evaluate
    for scene in scenes:
        scene_data = eval_json.get(scene.get_name(), dict())

        try:
            if visualize:
                log.info("Visualizing algorithm result on %s" % scene.get_display_name())
                scene_data["algorithm_result"] = visualize_algo_result(scene, algorithm_input_path,
                                                                       evaluation_output_path, add_pfms_to_result)

            log.info("Processing scene: %s" % scene.get_display_name())
            log.info("Using data from:\n  %s" % scene.get_data_path())
            scene_scores = compute_scores(scene, metrics, algorithm_input_path, evaluation_output_path, visualize)

            if add_to_existing_results:
                existing_scores = scene_data.get("scores", dict())
                existing_scores.update(scene_scores)
                scene_scores = existing_scores

            scene_data["scores"] = scene_scores

        except IOError as e:
            admin_errors.append(e)
            log.error(e)
            continue

        eval_json[scene.get_name()] = scene_data

    # save json with scores and paths to visualizations
    file_io.write_file(eval_json, file_name_results)
    log.info("Done!")

    success = not admin_errors
    error_json = {"messages": admin_errors}
    return success, error_json


def get_relative_path(scene, descr, file_type=settings.FIG_TYPE):
    return "%s/%s_%s.%s" % (scene.get_category(), scene.get_name(), descr, file_type)


def visualize_algo_result(scene, algo_dir, tgt_dir, add_pfms_to_result):
    algo_result = misc.get_algo_result_from_dir(algo_dir, scene)

    # visualize
    fig = init_figure()
    cm = plt.imshow(algo_result, **settings.disp_map_args(scene))
    add_colorbar(cm, bins=8)

    # save fig
    relative_fname_thumb = get_relative_path(scene, "dispmap")
    fpath = op.normpath(op.join(tgt_dir, relative_fname_thumb))
    plotting.save_tight_figure(fig, fpath, hide_frames=True, pad_inches=0.01)

    # path info
    height, width = np.shape(algo_result)[:2]
    disp_map_data = {"thumb": relative_fname_thumb,
                     "channels": 3,
                     "height": height,
                     "width": width}

    # save raw disparity map
    if add_pfms_to_result and not scene.is_test():
        relative_fname_raw = get_relative_path(scene, "dispmap", file_type="pfm")
        fpath_tgt = op.normpath(op.join(tgt_dir, relative_fname_raw))
        fpath_src = misc.get_fname_algo_result(algo_dir, scene)
        log.info("Copying disp map file from %s to %s" % (fpath_src, fpath_tgt))
        shutil.copyfile(fpath_src, fpath_tgt)
        disp_map_data["raw"] = relative_fname_raw

    return disp_map_data


def compute_scores(scene, metrics, algo_dir, tgt_dir, visualize):
    scores = dict()

    # resolution for evaluation depends on metric
    low_res_metrics = scene.get_applicable_metrics_low_res(metrics)
    if low_res_metrics:
        scene.set_low_gt_scale()
        scores = add_scores(low_res_metrics, scene, algo_dir, tgt_dir, scores, visualize)

    high_res_metrics = scene.get_applicable_metrics_high_res(metrics)
    if high_res_metrics:
        scene.set_high_gt_scale()
        scores = add_scores(high_res_metrics, scene, algo_dir, tgt_dir, scores, visualize)

    scores = add_runtime(scene, algo_dir, scores, metrics)

    return scores


def add_runtime(scene, algo_dir, scores, metrics):
    runtime_metrics = [m for m in metrics if "runtime" in m.get_id()]
    for metric in runtime_metrics:
        score = metric.get_score_from_dir(scene, algo_dir)
        scores[metric.get_id()] = {"value": score}
        log.info("Score %5.2f for: %s, %s, Scale: %0.2f" %
                 (score, metric.get_display_name(), scene.get_display_name(), scene.gt_scale))
    return scores


def add_scores(metrics, scene, algo_dir, tgt_dir, scores, visualize):
    gt = scene.get_gt()
    algo_result = misc.get_algo_result_from_dir(algo_dir, scene)

    for metric in metrics:

        if visualize:
            score, vis = metric.get_score(algo_result, gt, scene, with_visualization=True)
            relative_fname = save_visualization(algo_result, vis, metric, scene, tgt_dir)
            metric_data = {"value": float(score), "visualization": {"thumb": relative_fname}}
        else:
            score = metric.get_score(algo_result, gt, scene)
            metric_data = {"value": float(score)}

        log.info("Score %5.2f for: %s, %s, Scale: %0.2f" %
                 (score, metric.get_display_name(), scene.get_display_name(), scene.gt_scale))

        scores[metric.get_id()] = metric_data

    return scores


def save_visualization(algo_result, metric_vis, metric, scene, tgt_dir):
    fig = init_figure()

    # algorithm result as background
    plt.imshow(algo_result, **settings.disp_map_args(scene, cmap="gray"))

    # metric visualization on top
    if scene.hidden_gt() and metric.pixelize_results() and settings.PIXELIZE:
        metric_vis = plotting.pixelize(metric_vis, noise_factor=0.05)
    cm = plt.imshow(metric_vis, **settings.metric_args(metric))
    add_colorbar(cm, metric.colorbar_bins)

    # save fig
    relative_fname = get_relative_path(scene, metric.get_id())
    fpath = op.normpath(op.join(tgt_dir, relative_fname))
    plotting.save_tight_figure(fig, fpath, hide_frames=True, pad_inches=0.01)

    return relative_fname


def add_colorbar(cm, bins, fontsize=5):
    cb = plt.colorbar(cm, shrink=0.9)
    cb.outline.set_linewidth(0)
    cb.locator = ticker.MaxNLocator(nbins=bins)
    cb.ax.tick_params(labelsize=fontsize)
    cb.update_ticks()


def init_figure():
    fig = plt.figure(figsize=settings.FIG_SIZE_EVALUATION)
    return fig
