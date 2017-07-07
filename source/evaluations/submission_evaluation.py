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

import settings
from utils import file_io, log, misc, plotting


def evaluate(evaluation_output_path, algorithm_input_path, ground_truth_path,
             scenes, metrics, visualize=False):
    """
    :param evaluation_output_path: target directory for all evaluation results
    :param algorithm_input_path: input directory for algorithm results,
                                 expected directories: runtimes, disp_maps
    :param ground_truth_path: input directory for ground truth data
    :param scenes: scenes to be evaluated
    :param metrics: metrics to be evaluated
    :param visualize: whether to save visualizations (otherwise just the scores)
    :return: success, {"messages": ["error 1", "error 2", ...]}
    """

    admin_errors = []
    eval_json = dict()

    log.info("Evaluating algorithm results in:\n  %s" % algorithm_input_path)
    log.info("Writing results to:\n  %s" % evaluation_output_path)
    log.info("Using ground truth data from:\n  %s" % ground_truth_path)
    log.info("Metrics: %s" % ", ".join(m.get_display_name() for m in metrics))
    log.info("Scenes: %s" % ", ".join(s.get_display_name() for s in scenes))

    # evaluate
    for scene in scenes:
        scene_data = dict()

        try:
            if visualize:
                log.info("Visualizing algorithm result on %s" % scene.get_display_name())
                scene_data["algorithm_result"] = visualize_algo_result(scene, algorithm_input_path,
                                                                       evaluation_output_path)

            log.info("Processing scene: %s" % scene.get_display_name())
            log.info("Using data from:\n  %s" % scene.get_data_path())
            scene_data["scores"] = compute_scores(scene, metrics, algorithm_input_path,
                                                  evaluation_output_path, visualize)

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


def get_relative_path(scene, descr, file_type=settings.fig_type):
    return "%s/%s_%s.%s" % (scene.get_category(), scene.get_name(), descr, file_type)


def visualize_algo_result(scene, algo_dir, tgt_dir):
    algo_result = misc.get_algo_result_from_dir(scene, algo_dir)

    # visualize
    fig = init_figure()
    cm = plt.imshow(algo_result, **settings.disp_map_args(scene))
    add_colorbar(cm, bins=8)

    # save fig
    relative_fname = get_relative_path(scene, "dispmap")
    fpath = op.normpath(op.join(tgt_dir, relative_fname))
    plotting.save_tight_figure(fig, fpath, hide_frames=True, pad_inches=0.01)

    # path info for django importer
    disp_map_data = {"thumb": relative_fname}

    return disp_map_data


def compute_scores(scene, metrics, algo_dir, tgt_dir, visualize):
    scores = dict()

    # resolution for evaluation is metric specific
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
        scores[metric.get_id()] = {"value": metric.get_score_from_dir(scene, algo_dir)}
    return scores


def add_scores(metrics, scene, algo_dir, tgt_dir, scores, visualize):
    gt = scene.get_gt()
    algo_result = misc.get_algo_result_from_dir(scene, algo_dir)

    for idx_m, metric in enumerate(metrics):
        log.info("Computing score for metric: %s, scale: %0.2f" %
                 (metric.get_display_name(), scene.gt_scale))

        if visualize:
            score, vis = metric.get_score(algo_result, gt, scene, with_visualization=True)
            relative_fname = save_visualization(algo_result, vis, metric, scene, tgt_dir)
            metric_data = {"value": float(score), "visualization": {"thumb": relative_fname}}
        else:
            score = metric.get_score(algo_result, gt, scene)
            metric_data = {"value": float(score)}

        scores[metric.get_id()] = metric_data

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