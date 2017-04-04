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


import optparse
import os.path as op


def get_metric_dict():
    from scenes import BaseScene, BasePhotorealistic, BaseStratified
    md = {
        "general": BaseScene.get_general_metrics(),
        "stratified": BaseStratified.get_stratified_metrics(),
        "regions": BasePhotorealistic.get_region_metrics(),
        "all_wo_runtime": BaseScene.get_all_metrics_wo_runtime(),
        "all": BaseScene.get_all_metrics()
    }
    return md


def parse_evaluation_options():
    parser = optparse.OptionParser()

    def get_comma_separated_args(option, opt, value, parser):
        values = [v for v in value.split(',')]
        setattr(parser.values, option.dest, values)

    # algorithms, scenes, and metrics
    parser.add_option("-a", type="string", action="callback", callback=get_comma_separated_args,
                      dest="algo_names", help="comma separated list of algorithm names, e.g. 'epi1,lf,mv', "
                                              "default: all directories in ALGO_RESULTS")

    parser.add_option("-s", type="string", action="callback", callback=get_comma_separated_args,
                      dest="scene_names", help="comma separated list of scene names, e.g. 'bedroom,dots'")

    parser.add_option("-m", type="string",  dest="metric_set",
                      help="set of metrics, options: "
                           "'stratified': all metrics of the stratified scenes "
                           "'regions': all region metrics of the photorealistic scenes "
                           "'general': MSE and BadPix(0.07) "
                           "'all_wo_runtime': all applicable metrics without runtime "
                           "'all': all applicable metrics including runtime")

    # flags
    parser.add_option("--visualize", action="store_true", dest="visualize", default=False,
                      help="set flag to create visualization figures during evaluation")

    options, args = parser.parse_args()

    # delay imports to speed up usage response
    from utils import misc

    # check algorithms
    available_algo_names = misc.get_available_algo_names()
    if options.algo_names is None:
        algo_names = available_algo_names
    else:
        algo_names = []
        for algo_name in options.algo_names:
            if not algo_name in available_algo_names:
                parser.error("Could not find algorithm for: %s'. "
                             "Available options are: %s" % (algo_name, ", ".join(available_algo_names)))
            else:
                algo_names.append(algo_name)

    # check scenes
    scene_dict = misc.get_scene_dict()
    if options.scene_names is None:
        scenes = scene_dict.values()
    else:
        valid_scene_names = scene_dict.keys()
        scenes = []
        for scene_name in options.scene_names:
            if not scene_name in valid_scene_names:
                parser.error("Could not find scene for: %s'. "
                             "Available options are: %s" % (scene_name, ", ".join(valid_scene_names)))
            else:
                scenes.append(scene_dict[scene_name])

    # check metrics
    metric_dict = get_metric_dict()
    if options.metric_set is not None:
        metrics = metric_dict.get(options.metric_set, None)
        if metrics is None:
            parser.error("Could not find metrics for: '%s'. "
                         "Available options are: %s" % (options.metric_set, ", ".join(metric_dict.keys())))
    else:
        metrics = metric_dict.get(options.metric_set, metric_dict["all_wo_runtime"])

    return algo_names, scenes, metrics, options.visualize


if __name__ == "__main__":
    algo_names, scenes, metrics, visualize = parse_evaluation_options()

    # delay import to speed up usage response
    from evaluations import submission_evaluation
    import settings
    from utils import misc

    for algo_name in algo_names:
        submission_evaluation.evaluate(selected_scenes=scenes,
                                       metrics=metrics,
                                       visualize=visualize,
                                       ground_truth_path=settings.DATA_PATH,
                                       evaluation_output_path=op.join(settings.EVAL_PATH, algo_name),
                                       algorithm_input_path=misc.get_path_to_algo_data(algo_name))