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

import argparse
import re

import settings
from utils.logger import log


class OptionParser(argparse.ArgumentParser):

    def __init__(self, options=[], *args, **kwargs):
        super(OptionParser, self).__init__(formatter_class=argparse.RawTextHelpFormatter, *args, **kwargs)
        self.actions = []

        for option in options:
            self.actions += option.add_arguments(self)

    def parse_args(self, args=None, namespace=None):
        # try to parse all provided arguments
        namespace = super(OptionParser, self).parse_args(args, namespace)

        # call action with default action if option string was not provided
        # e.g. collect default scenes if no "-s" or "--scenes" was provided
        # allows for lazy initialization, avoids file parsing.. when specifying default directly
        for action in self.actions:
            if getattr(namespace, action.dest) is None:
                log.info('No values provided for option: "%s", initializing with default.' % action.option_strings[0])
                action.__call__(self, namespace, values=None)

        # return values in order of parser options
        values = [getattr(namespace, action.dest) for action in self.actions]
        return values


class SceneOps(object):

    def add_arguments(self, parser):
        action = parser.add_argument("-s",
                                     dest="scenes", action=self.SceneAction,
                                     type=str, nargs="+",
                                     help='list of scene names\n'
                                          'example: "-s cotton dino"\n'
                                          'default: all scenes in scene category directories in DATA_PATH \n  %s \n'
                                          'further options: %s' % (settings.DATA_PATH, ", ".join(settings.SCENE_PACKAGES)))
        return [action]

    class SceneAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            from utils import misc
            from scenes import PhotorealisticScene

            available_scene_names = misc.get_available_scene_names()

            # default: use all available scenes
            if not values:
                scene_names = available_scene_names
            else:
                scene_names = []
                for scene_option in values:
                    if scene_option not in available_scene_names:
                        if scene_option in settings.SCENE_PACKAGES:
                            scene_names += self.get_scene_names_from_package(scene_option)
                        else:
                            parser.error("Could not find scene for: %s. "
                                         "Available scenes are: %s." % (scene_option, ", ".join(available_scene_names)))
                    else:
                        scene_names.append(scene_option)

            scene_dict = misc.get_scene_dict()
            scenes = []
            for scene_option in scene_names:
                if scene_option in scene_dict:
                    scene = scene_dict[scene_option]
                elif scene_option in settings.get_scene_names_additional():
                    scene = PhotorealisticScene(name=scene_option, category="additional")
                else:
                    scene = PhotorealisticScene(name=scene_option, category="other")
                scenes.append(scene)

            setattr(namespace, self.dest, scenes)

        @staticmethod
        def get_scene_names_from_package(scene_option):
            package_dict = {settings.TRAINING_SCENE: settings.get_scene_names_training(),
                            settings.TEST_SCENE: settings.get_scene_names_test(),
                            settings.STRATIFIED_SCENE: settings.get_scene_names_stratified(),
                            settings.ADDITIONAL_SCENE: settings.get_scene_names_additional(),
                            settings.BENCHMARK_SCENE: settings.get_scene_names_training() + \
                                                      settings.get_scene_names_test() + \
                                                      settings.get_scene_names_stratified()
                            }

            return package_dict[scene_option]


class AlgorithmOps(object):

    def __init__(self, with_gt=False, additional_help_text="", ignore_meta_algorithms=True):
        self.with_gt = with_gt
        self.additional_help_text = additional_help_text
        self.ignore_meta_algorithms = ignore_meta_algorithms

    def add_arguments(self, parser):
        help = 'list of algorithm names\n' \
               'example: "-a epi1 lf mv"\n' \
               'default: all directories in ALGO_PATH\n  %s\n' \
               '  (per pixel meta algorithms are ignored)' % settings.ALGO_PATH

        if self.with_gt:
            help += '\nfurther options: gt'

        if self.additional_help_text:
            help += '\n' + self.additional_help_text

        if self.ignore_meta_algorithms:
            algorithms_to_ignore = [algorithm.get_name() for algorithm in MetaAlgorithmOps().meta_algorithms.values()]
        else:
            algorithms_to_ignore = []

        action = parser.add_argument("-a",
                                     dest="algorithms", action=AlgorithmAction,
                                     algorithms_to_ignore=algorithms_to_ignore,
                                     type=str, nargs="+",
                                     help=help)
        return [action]


class AlgorithmAction(argparse.Action):

    def __init__(self, option_strings, algorithms_to_ignore=[], *args, **kwargs):
        self.algorithms_to_ignore = algorithms_to_ignore
        super(AlgorithmAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from utils import misc
        from algorithms import Algorithm

        available_algo_names = [a for a in misc.get_available_algo_names() if a not in self.algorithms_to_ignore]

        # default: use all available algorithms
        if not values:
            algo_names = available_algo_names
        # otherwise: check if selected algorithms exist
        else:
            algo_names = []
            for algo_name in values:
                if algo_name not in available_algo_names and algo_name != "gt":
                    parser.error("Could not find algorithm for: %s. "
                                 "Available options are: %s." % (algo_name, ", ".join(available_algo_names)))
                else:
                    algo_names.append(algo_name)

        # create algorithm objects
        algorithms = [Algorithm(file_name=algo_file_name) for algo_file_name in algo_names]
        algorithms = Algorithm.set_colors(algorithms)

        # save result in action destination
        setattr(namespace, self.dest, algorithms)


class MetaAlgorithmOps(object):

    def __init__(self):
        from algorithms import PerPixBest, PerPixMean, PerPixMedianDiff, PerPixMedianDisp
        self.meta_algorithms = {"best": PerPixBest(),
                                "mean": PerPixMean(),
                                "mediandisp": PerPixMedianDisp(),
                                "mediandiff": PerPixMedianDiff()}

    def add_arguments(self, parser):
        action = parser.add_argument("-p",
                                     dest="meta_algorithms", action=MetaAlgorithmAction,
                                     meta_algorithms=self.meta_algorithms,
                                     type=str, nargs="+",
                                     help='list of meta algorithm names\n'
                                          'example: "-a best mean"\n'
                                          'default: all options\n'
                                          'options: %s' % ", ".join(sorted(self.meta_algorithms.keys())))
        return [action]


class MetaAlgorithmAction(argparse.Action):

    def __init__(self, option_strings, meta_algorithms, *args, **kwargs):
        self.meta_algorithms = meta_algorithms
        super(MetaAlgorithmAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        algorithms = []

        if not values:
            algorithms = self.meta_algorithms.values()
        else:
            for value in values:
                try:
                    algorithms.append(self.meta_algorithms[value])
                except KeyError:
                    parser.error("Could not find algorithm for: %s. "
                                 "Available options are: %s." % (value, ", ".join(self.meta_algorithms.keys())))

        # save result in action destination
        setattr(namespace, self.dest, algorithms)


class MetricOps(object):

    @staticmethod
    def get_metric_groups():
        from utils import misc
        metric_dict = {
            "general": misc.get_general_metrics(),
            "stratified": misc.get_stratified_metrics(),
            "regions": misc.get_region_metrics(),
            "all_wo_runtime": misc.get_all_metrics_wo_runtime(),
            "all": misc.get_all_metrics()
        }
        return metric_dict

    @staticmethod
    def get_individual_metrics():
        from metrics import Runtime
        from utils import misc

        metrics = misc.get_all_metrics_wo_runtime() + [Runtime(log=True), Runtime(log=False)]
        metric_dict = {}

        for metric in metrics:
            to_be_removed = ["(", ")", ".", ":"]
            metric_key = metric.get_display_name().lower().replace(" ", "_").translate(None, "".join(to_be_removed))
            metric_dict[metric_key] = [metric]

        return metric_dict

    @staticmethod
    def get_all_options():
        metric_dict = MetricOps.get_individual_metrics()
        metric_dict.update(MetricOps.get_metric_groups())
        return metric_dict

    def add_arguments(self, parser):
        invididual_metric_str = ""
        metric_keys = sorted(self.get_individual_metrics().keys())
        lines = [metric_keys[n:n+3] for n in range(0, len(metric_keys), 3)]
        for line in lines:
            invididual_metric_str += "  " + ", ".join(line) + ",\n"
        invididual_metric_str = invididual_metric_str[:-2]

        general_metrics_str = ", ".join(m.get_display_name() for m in self.get_metric_groups()["general"])

        action = parser.add_argument("-m",
                                     dest="metrics", action=self.MetricAction,
                                     type=str, nargs="+",
                                     help='list of metric names\n'
                                          'example: "-m badpix007 mse"\n'
                                          'default: all\n'
                                          'individual metrics:\n%s\n'
                                          'metric sets:\n'
                                          '  stratified: all metrics of the stratified scenes\n'
                                          '  regions: all region metrics of the photorealistic scenes\n'
                                          '  general: %s\n'
                                          '  all_wo_runtime: all applicable metrics without runtime\n'
                                          '  all: all applicable metrics including runtime\n' % (invididual_metric_str,
                                                                                                 general_metrics_str))
        return [action]

    class MetricAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            from metrics import BadPix

            metric_dict = MetricOps.get_all_options()
            metrics = []

            if not values:
                metrics = metric_dict["all"]
            else:
                for value in values:
                    try:
                        metrics += metric_dict[value]
                    except KeyError:
                        if re.match("^badpix\d{3}", value):
                            threshold = float((value[6] + "." + value[7:]))
                            metrics += [BadPix(threshold)]
                        else:
                            parser.error("Could not find metrics for: %s. "
                                         "Available options are: %s." % (value, ", ".join(sorted(metric_dict.keys()))))

            # save result in action destination
            setattr(namespace, self.dest, metrics)


class VisualizationOps(object):

    def add_arguments(self, parser):
        action = parser.add_argument("-v", "--visualize",
                                     dest="visualize", action="store_true",
                                     help="set flag to create visualization figures during evaluation")
        return [action]


class FigureOpsACCV16(object):
    
    def add_arguments(self, parser):
        actions = []

        actions.append(parser.add_argument("--heatmaps",
                                           dest="heatmaps", action="store_true",
                                           help="create figure with error heatmap per scene over all algorithms"))

        actions.append(parser.add_argument("--radar",
                                           dest="radar_charts", action="store_true",
                                           help="create radar charts for stratified and training scenes"))

        actions.append(parser.add_argument("--stratified",
                                           dest="stratified", action="store_true",
                                           help="create metric visualization figure for each "
                                                "stratified scene with all algorithms"))

        actions.append(parser.add_argument("--training",
                                           dest="training", action="store_true",
                                           help="create metric visualization figure for each "
                                                "training scene with all algorithms"))

        actions.append(parser.add_argument("--charts",
                                           dest="stratified_charts", action="store_true",
                                           help="create charts along image dimensions of stratified scenes"))

        return actions


class FigureOpsCVPR17(object):

    def add_arguments(self, parser):
        actions = []

        actions.append(parser.add_argument("--scenes",
                                           dest="scene_overview", action="store_true",
                                           help="create figure with center view and ground truth per scene"))

        actions.append(parser.add_argument("--normalsdemo",
                                           dest="normals_demo", action="store_true",
                                           help="create figure with ground truth normals, algorithm normals, "
                                                "and angular error for Sideboard scene"))

        actions.append(parser.add_argument("--radar",
                                           dest="radar_charts", action="store_true",
                                           help="create radar charts for stratified and photorealistic scenes"))

        actions.append(parser.add_argument("--badpix",
                                           dest="bad_pix_series", action="store_true",
                                           help="create figures with BadPix series "
                                                "for stratified and photorealistic scenes"))

        actions.append(parser.add_argument("--median",
                                           dest="median_comparisons", action="store_true",
                                           help="create figures with MedianDiff comparisons "
                                                "for stratified and training scenes"))

        actions.append(parser.add_argument("--normals",
                                           dest="normals_overview", action="store_true",
                                           help="create figure with disparity map, normal map, "
                                                "and angular error per algorithm"))

        actions.append(parser.add_argument("--accuracy",
                                           dest="high_accuracy", action="store_true",
                                           help="create figure with BadPix and Q25 visualizations"))
        return actions




