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
from utils import log


class OptionParser(argparse.ArgumentParser):

    def __init__(self, options=[], *args, **kwargs):
        super(OptionParser, self).__init__(formatter_class=argparse.RawTextHelpFormatter,
                                           *args, **kwargs)
        self.actions = []
        for option in options:
            self.actions += option.add_arguments(self)

    def parse_args(self, args=None, namespace=None):
        # try to parse all provided arguments
        namespace = super(OptionParser, self).parse_args(args, namespace)

        # call action with default action if option string was not provided
        # e.g. collect default scenes if no "-s" or "--scenes" was provided
        # (values are not precomputed and set as default argument
        #  to avoid costly and mostly unused initializations)
        for action in self.actions:
            if getattr(namespace, action.dest) is None:
                action.__call__(self, namespace, values=None)

        [log.info("%s: %s" % (a.dest, getattr(namespace, a.dest))) for a in self.actions]

        # return values in order of parser options
        values = [getattr(namespace, action.dest) for action in self.actions]

        if len(values) == 1:
            return values[0]
        return values


class ConverterOps(object):

    def __init__(self, input="path to input file", output="path to output file",
                 config="path to parameters.cfg of the scene"):
        self.input_help = input
        self.output_help = output
        self.config_help = config

    def add_arguments(self, parser):
        actions = []
        actions.append(parser.add_argument(dest='input_file', type=str, help=self.input_help))
        actions.append(parser.add_argument(dest='config_file', type=str, help=self.config_help))
        actions.append(parser.add_argument(dest='output_file', type=str, help=self.output_help))
        return actions


class ConverterOpsExt(ConverterOps):

    def __init__(self, optional_input, *args, **kwargs):
        self.optional_input = optional_input
        super(ConverterOpsExt, self).__init__(*args, **kwargs)

    def add_arguments(self, parser):
        actions = super(ConverterOpsExt, self).add_arguments(parser)
        for flag, name, help in self.optional_input:
            actions.append(parser.add_argument(flag, dest=name, type=str, help=help))
        return actions


class SceneOps(object):

    def add_arguments(self, parser):
        action = parser.add_argument("-s",
                                     dest="scenes", action=self.SceneAction,
                                     type=str, nargs="+",
                                     help='list of scenes or scene category names\n'
                                          'example 1: "-s cotton dino"\n'
                                          'example 2: "-s training" '
                                          '(will add all locally available training scenes)\n'
                                          'default: all scenes in category directories '
                                          'in DATA_PATH\n  %s.' % settings.DATA_PATH)
        return [action]

    class SceneAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            from utils import misc

            # collect available scenes and scene categories
            available_scenes_with_categories = misc.get_available_scenes_with_categories()
            available_scenes = available_scenes_with_categories.keys()
            available_categories = list(set(list(available_scenes_with_categories.values())))

            # parse given scene and category names
            if not values:
                # default: use all available scenes
                scenes_with_categories = available_scenes_with_categories
            else:
                scenes_with_categories = dict()
                for value in values:
                    # add regular scene
                    if value in available_scenes:
                        scenes_with_categories[value] = available_scenes_with_categories[value]
                    else:
                        # add available scenes of a given category
                        if value in available_categories:
                            for scene_name, category in available_scenes_with_categories.items():
                                if category == value:
                                    category = available_scenes_with_categories[scene_name]
                                    scenes_with_categories[scene_name] = category
                        else:
                            parser.error("Could not find scene for: %s.\n"
                                         "Available scenes are: %s.\n"
                                         "Available categories are: %s." %
                                         (value,
                                          ", ".join(available_scenes),
                                          ", ".join(available_categories)))

            # initialize scene objects for all valid scene names
            scenes = [misc.get_scene(n, c) for n, c in sorted(scenes_with_categories.items())]

            setattr(namespace, self.dest, scenes)


class AlgorithmOps(object):

    def __init__(self, with_gt=False, default_algo_names=None):
        self.with_gt = with_gt
        self.default_algo_names = default_algo_names

    def add_arguments(self, parser):
        if self.default_algo_names is not None:
            default = ' '.join(self.default_algo_names)
        else:
            default = 'all algorithm directories in ALGO_PATH\n  %s' % settings.ALGO_PATH
        default += '\n  (per pixel meta algorithms are ignored)'

        further_options = ""
        if self.with_gt:
            further_options += '\nfurther options: gt'

        action = parser.add_argument("-a",
                                     dest="algorithms", action=AlgorithmAction,
                                     default_algo_names=self.default_algo_names,
                                     type=str, nargs="+",
                                     help='list of algorithm names\n'
                                          'example: "-a epi1 lf mv"\n'
                                           'default: %s%s' % (default, further_options))
        return [action]


class AlgorithmAction(argparse.Action):

    def __init__(self, option_strings, default_algo_names=None, *args, **kwargs):
        self.default_algo_names = default_algo_names
        super(AlgorithmAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from utils import misc
        from algorithms import Algorithm, MetaAlgorithm

        ignore = [a.get_name() for a in MetaAlgorithm.get_meta_algorithms()]
        available_algo_names = [a for a in misc.get_available_algo_names() if a not in ignore]

        # default: use all available algorithms
        if not values:
            if self.default_algo_names is not None:
                algo_names = [algo_name for algo_name in self.default_algo_names
                              if algo_name in available_algo_names]
            else:
                algo_names = available_algo_names
        # otherwise: check if selected algorithms exist
        else:
            algo_names = []
            for algo_name in values:
                if algo_name not in available_algo_names and algo_name != "gt":
                    parser.error("Could not find algorithm for: %s. "
                                 "Available options are: %s." %
                                 (algo_name, ", ".join(available_algo_names)))
                else:
                    algo_names.append(algo_name)

        # create algorithm objects
        algorithms = Algorithm.initialize_algorithms(algo_names)

        # save result in action destination
        setattr(namespace, self.dest, algorithms)


class MetaAlgorithmOps(object):

    def __init__(self, default=None, with_load_argument=True):
        self.with_load_argument = with_load_argument

        # prepare algorithm options
        from algorithms import MetaAlgorithm
        self.meta_algorithms = {algo.get_name().replace("per_pix_", ""): algo
                                for algo in MetaAlgorithm.get_meta_algorithms()}

        if default is None:
            default = sorted(self.meta_algorithms.keys())
        self.default_algo_names = default

    def add_arguments(self, parser):
        if self.default_algo_names:
            default = " ".join(self.default_algo_names)
        else:
            default = "no meta algorithm"

        actions = []
        options = ", ".join(sorted(self.meta_algorithms.keys()))
        actions.append(parser.add_argument("-p",
                                           dest="meta_algorithms", action=MetaAlgorithmAction,
                                           meta_algorithms=self.meta_algorithms,
                                           default_algo_names=self.default_algo_names,
                                           type=str, nargs="+",
                                           help='list of meta algorithm names\n'
                                                'example: "-a best mean"\n'
                                                'default: %s\n'
                                                'options: %s' % (default, options)))

        if self.with_load_argument:
            actions.append(parser.add_argument("-u", "--use_existing_meta_files",
                                               dest="compute_meta_algos", action="store_false",
                                               help="use existing meta algorithm files "
                                                    "(per default, meta algorithms\n"
                                                    "are computed from scratch "
                                                    "based on list of 'regular' algorithms)"))

        return actions


class MetaAlgorithmAction(argparse.Action):

    def __init__(self, option_strings, meta_algorithms, default_algo_names, *args, **kwargs):
        self.meta_algorithms = meta_algorithms
        self.default_algo_names = default_algo_names
        super(MetaAlgorithmAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        meta_algorithms = []

        if not values:
            meta_algorithms = [self.meta_algorithms[a] for a in self.default_algo_names]
        else:
            for value in values:
                try:
                    meta_algorithms.append(self.meta_algorithms[value])
                except KeyError:
                    parser.error("Could not find algorithm for: %s. "
                                 "Available options are: %s." %
                                 (value, ", ".join(self.meta_algorithms.keys())))

        # save result in action destination
        setattr(namespace, self.dest, meta_algorithms)


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
            metric_key = metric.get_display_name().lower().replace(" ", "_")
            metric_key = metric_key.translate(None, "".join(to_be_removed))
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

        general_metrics_str = ", ".join(m.get_display_name() for m in
                                        self.get_metric_groups()["general"])

        action = parser.add_argument("-m",
                                     dest="metrics", action=self.MetricAction,
                                     type=str, nargs="+",
                                     help='list of metric names\n'
                                          'example: "-m badpix007 mse"\n'
                                          'default: all\n'
                                          'individual metrics:\n%s\n'
                                          'metric sets:\n'
                                          '  stratified: all metrics of the stratified scenes\n'
                                          '  regions: region metrics of the photorealistic scenes\n'
                                          '  general: %s\n'
                                          '  all_wo_runtime: applicable metrics without runtime\n'
                                          '  all: applicable metrics including runtime\n' %
                                          (invididual_metric_str, general_metrics_str))
        return [action]

    class MetricAction(argparse.Action):

        def __call__(self, parser, namespace, values, option_string=None):
            from metrics import BadPix, Quantile

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
                        elif re.match("^q\d{2}", value):
                            percentage = int(value[1:])
                            metrics += [Quantile(percentage)]
                        else:
                            parser.error("Could not find metrics for: %s. "
                                         "Available options are: %s." %
                                         (value, ", ".join(sorted(metric_dict.keys()))))

            # save result in action destination
            setattr(namespace, self.dest, metrics)


class VisualizationOps(object):

    def add_arguments(self, parser):
        action = parser.add_argument("-v", "--visualize",
                                     dest="visualize", action="store_true",
                                     help="set flag to save figures during evaluation")
        return [action]


class ThresholdOps(object):

    def __init__(self, threshold=0.07):
        self.threshold = threshold

    def add_arguments(self, parser):

        action = parser.add_argument("-t", "--threshold",
                                     dest="threshold", type=float, default=self.threshold,
                                     help="default: %0.3f" % self.threshold)
        return [action]


class FigureOps(object):

    def add_arguments(self, parser):
        options = "".join("\n  %s: %s" % (k, v) for k, v in sorted(self.figure_options.items()))
        action = parser.add_argument("-f",
                                     dest="figure_options", action=FigureOpsAction,
                                     figure_options=self.figure_options,
                                     type=str, nargs="+",
                                     help='list of figure names\n'
                                          'example: "-a heatmaps radar"\n'
                                          'default: all options\n'
                                          'options: %s' % options)
        return [action]


class FigureOpsAction(argparse.Action):
    def __init__(self, option_strings, figure_options, *args, **kwargs):
        self.figure_options = figure_options
        super(FigureOpsAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        figure_options = []
        available_options = self.figure_options.keys()

        if not values:
            figure_options = sorted(available_options)
        else:
            for value in values:
                if value in available_options:
                    figure_options.append(value)
                else:
                    parser.error("Could not find figure option for: %s. "
                                 "Available options are: %s." %
                                 (value, ", ".join(available_options)))

        # save result in action destination
        setattr(namespace, self.dest, figure_options)


class FigureOpsACCV16(FigureOps):

    def __init__(self):
        super(FigureOpsACCV16, self).__init__()
        self.figure_options = {"heatmaps":
                                   "figure with algorithm error heatmap per scene",
                               "radar":
                                   "radar charts for stratified and training scenes",
                               "stratified":
                                   "metric visualization figure for each stratified scene",
                               "training":
                                   "metric visualization figure for each training scene",
                               "backgammon":
                                   "fattening and thinning along vertical image dimension",
                               "dots":
                                   "background error per box with increasing noise levels",
                               "pyramids":
                                   "algorithm disparities vs ground truth disparities on spheres",
                               "stripes":
                                   "visualization of evaluation masks"}


class FigureOpsCVPR17(FigureOps):

    def __init__(self):
        super(FigureOpsCVPR17, self).__init__()
        self.figure_options = {"scenes":
                                   "center view and ground truth per scene",
                               "difficulty":
                                   "error map of per pixel median and best disparity per scene",
                               "normalsdemo":
                                   "ground truth and algorithm normals + angular error (Sideboard)",
                               "radar":
                                   "radar charts for stratified and photorealistic scenes",
                               "badpix":
                                   "BadPix series for stratified and photorealistic scenes",
                               "median":
                                   "MedianDiff comparisons for stratified and training scenes",
                               "normals":
                                   "disparity and normal map + angular error per algorithm (Cotton)",
                               "discont":
                                   "disparity map and MedianDiff per algorithm (Bicycle)",
                               "accuracy":
                                   "BadPix and Q25 visualizations (Cotton and Boxes)"}





