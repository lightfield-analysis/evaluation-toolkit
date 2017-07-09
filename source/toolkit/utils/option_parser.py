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


import abc
import argparse
import re

from toolkit import settings
from toolkit.utils import log


class OptionParser(argparse.ArgumentParser):

    def __init__(self, options, *args, **kwargs):
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
        for action in self.actions:
            if getattr(namespace, action.dest) is None:
                action.__call__(self, namespace, values=None)

        log.info("Command line arguments: ")
        [log.info("%s: %s" % (a.dest.title(), getattr(namespace, a.dest))) for a in self.actions]

        # return values in order of parser options
        values = [getattr(namespace, action.dest) for action in self.actions]

        if len(values) == 1:
            return values[0]
        return values


class Ops(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def add_arguments(self, parser):
        return


class SceneOps(Ops):

    def add_arguments(self, parser):
        action = parser.add_argument("-s",
                                     dest="scenes", action=SceneAction,
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
        from toolkit.utils import misc

        # collect available scenes and scene categories
        available_scenes_by_category = misc.get_available_scenes_by_category()
        available_categories = available_scenes_by_category.keys()
        available_scenes = [s for sublist in available_scenes_by_category.values() for s in sublist]

        # scene names must be unique across categories as they are used as distinct parser arguments
        n_scenes = len(available_scenes)
        n_unique_scenes = len(set(available_scenes))
        if n_scenes > n_unique_scenes:
            raise Exception("Scene names must be unique across all categories. "
                            "Found %d duplicate(s)." % (n_scenes - n_unique_scenes))

        # parse given scene and category names
        scenes_by_category = dict()

        if not values:
            # default: use all available scenes
            scenes_by_category = available_scenes_by_category
        else:
            for value in values:
                if value in available_scenes:
                    # retrieve category and add regular scene
                    category = [category for category, scenes
                                in available_scenes_by_category.items() if value in scenes][0]
                    scenes_by_category.setdefault(category, []).append(value)
                else:
                    # add available scenes of a given category
                    if value in available_categories:
                        scenes = available_scenes_by_category[value]
                        category_scenes = scenes_by_category.get(value, [])
                        scenes_by_category[value] = category_scenes + scenes
                    else:
                        parser.error("Could not find scene for: %s.\n  "
                                     "Available scenes are: %s.\n  "
                                     "Available categories are: %s." %
                                     (value,
                                      ", ".join(available_scenes),
                                      ", ".join(available_categories)))

        # initialize scene objects
        scenes = []
        for category, scene_names in scenes_by_category.items():
            scenes += [misc.get_scene(scene_name, category) for scene_name in scene_names]

        setattr(namespace, self.dest, scenes)


class AlgorithmOps(Ops):

    def __init__(self, default=None, with_gt=False):
        self.with_gt = with_gt
        self.default = default

    def add_arguments(self, parser):

        # prepare help text
        if self.default is not None:
            default = ' '.join(self.default)
        else:
            default = 'all algorithm directories in ALGO_PATH\n  %s' % settings.ALGO_PATH
        default += '\n  (per pixel meta algorithms are ignored)'

        further_options = '\nfurther options: gt' if self.with_gt else ''

        action = parser.add_argument("-a",
                                     dest="algorithms", action=AlgorithmAction,
                                     default_algo_names=self.default,
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
        from toolkit.utils import misc
        from toolkit.algorithms import Algorithm, MetaAlgorithm

        ignore = [a.get_name() for a in MetaAlgorithm.get_meta_algorithms()]
        available_algo_names = [a for a in misc.get_available_algo_names() if a not in ignore]

        if not values:
            # use given default or all available algorithms
            if self.default_algo_names is not None:
                algo_names = [a for a in self.default_algo_names if a in available_algo_names]
            else:
                algo_names = available_algo_names
        else:
            # otherwise: check if selected algorithms exist
            algo_names = []
            for algo_name in values:
                if algo_name not in available_algo_names and algo_name != "gt":
                    parser.error("Could not find algorithm for: %s.\n  "
                                 "Available options are: %s." %
                                 (algo_name, ", ".join(available_algo_names)))
                else:
                    algo_names.append(algo_name)

        # create algorithm objects
        algorithms = Algorithm.initialize_algorithms(algo_names)

        # save result in action destination
        setattr(namespace, self.dest, algorithms)


class MetaAlgorithmOps(Ops):

    def __init__(self, default=None, with_load_argument=True):
        self.default = default
        self.with_load_argument = with_load_argument

    def add_arguments(self, parser):
        from toolkit.algorithms import MetaAlgorithm

        # prepare algorithm options
        algorithms_by_name = {algo.get_name().replace("per_pix_", ""): algo
                              for algo in MetaAlgorithm.get_meta_algorithms()}
        meta_algorithm_keys = sorted(algorithms_by_name.keys())

        if self.default is None:
            self.default = meta_algorithm_keys

        # prepare help text
        option_text = ", ".join(meta_algorithm_keys)
        default_text = " ".join(self.default) if self.default else "no meta algorithm"

        # add arguments
        actions = list()
        actions.append(parser.add_argument("-p",
                                           dest="meta_algorithms", action=MetaAlgorithmAction,
                                           algorithms_by_name=algorithms_by_name,
                                           default_algo_names=self.default,
                                           type=str, nargs="+",
                                           help='list of meta algorithm names\n'
                                                'example: "-a best mean"\n'
                                                'default: %s\n'
                                                'options: %s' % (default_text, option_text)))

        if self.with_load_argument:
            actions.append(parser.add_argument("-u", "--use_existing_meta_files",
                                               dest="compute_meta_algos", action="store_false",
                                               help="use existing meta algorithm files "
                                                    "(per default, meta algorithms\n"
                                                    "are computed from scratch "
                                                    "based on list of 'regular' algorithms)"))
        return actions


class MetaAlgorithmAction(argparse.Action):

    def __init__(self, option_strings, algorithms_by_name, default_algo_names, *args, **kwargs):
        self.algorithms_by_name = algorithms_by_name
        self.default_algo_names = default_algo_names
        super(MetaAlgorithmAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        algorithms = []

        if not values:
            algorithms = [self.algorithms_by_name[a] for a in self.default_algo_names]
        else:
            for value in values:
                try:
                    algorithms.append(self.algorithms_by_name[value])
                except KeyError:
                    parser.error("Could not find algorithm for: %s.\n  "
                                 "Available options are: %s." %
                                 (value, ", ".join(self.algorithms_by_name.keys())))

        # save result in action destination
        setattr(namespace, self.dest, algorithms)


class MetricOps(Ops):

    def add_arguments(self, parser):
        from toolkit.utils import misc

        # prepare metric groups with names: {"stratified": stratified_metrics, ...}
        metric_groups_by_name = misc.get_metric_groups_by_name()

        # prepare individual metrics with short keys: {"fine_thinning": [FineThinning()], ...}
        all_metrics = misc.get_all_metrics()
        ignored_chars = "".join(["(", ")", ".", ":"])

        metrics_by_name = {}
        for metric in all_metrics:
            key = metric.get_display_name().lower().replace(" ", "_").translate(None, ignored_chars)
            metrics_by_name[key] = [metric]

        # prepare help text for general metrics
        general_metrics = ", ".join(m.get_display_name() for m in metric_groups_by_name["general"])

        # prepare help text for all individual metrics
        metric_keys = sorted(metrics_by_name.keys())
        # add line break after each metric triple
        lines = [metric_keys[n:n + 3] for n in range(0, len(metric_keys), 3)]
        all_metrics = ",\n".join(["  " + ", ".join(line) for line in lines])

        # combine all valid options
        all_options = metrics_by_name.copy()
        all_options.update(metric_groups_by_name)

        action = parser.add_argument("-m",
                                     dest="metrics", action=MetricAction,
                                     metric_options=all_options,
                                     type=str, nargs="+",
                                     help='list of metric names\n'
                                          'example: "-m badpix007 mse"\n'
                                          'default: all\n'
                                          'individual metrics:\n%s\n'
                                          'metric sets:\n'
                                          '  stratified: special metrics of the stratified scenes\n'
                                          '  regions: region metrics of the photorealistic scenes\n'
                                          '  general: %s\n'
                                          '  all_wo_runtime: applicable metrics without runtime\n'
                                          '  all: applicable metrics including runtime\n'
                                          % (all_metrics, general_metrics))
        return [action]


class MetricAction(argparse.Action):

    def __init__(self, option_strings, metric_options, *args, **kwargs):
        self.metric_options = metric_options
        super(MetricAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from toolkit.metrics import BadPix, Quantile
        metrics = []

        if not values:
            metrics = self.metric_options["all"]
        else:
            for value in values:
                try:
                    metrics += self.metric_options[value]
                except KeyError:
                    # try to match BadPix metric with threshold
                    if re.match("^badpix\d{3}", value):
                        threshold = float((value[6] + "." + value[7:]))
                        metrics.append(BadPix(threshold))
                    # try to match Quantile metric with percentage
                    elif re.match("^q\d{2}", value):
                        percentage = int(value[1:])
                        metrics.append(Quantile(percentage))
                    else:
                        parser.error("Could not find metrics for: %s.\n  "
                                     "Available options are: %s." %
                                     (value, ", ".join(sorted(self.metric_options.keys()))))

        # save result in action destination
        setattr(namespace, self.dest, metrics)


class VisualizationOps(Ops):

    def add_arguments(self, parser):
        action = parser.add_argument("-v", "--visualize",
                                     dest="visualize", action="store_true",
                                     help="set flag to save figures during evaluation")
        return [action]


class OverwriteOps(Ops):

    def add_arguments(self, parser):
        action = parser.add_argument("-d", "--delete_existing_results",
                                     dest="add_to_existing_results", action="store_false",
                                     help="set flag to create a new results.json, \n"
                                          "deleting all previously computed scores")
        return [action]


class ThresholdOps(Ops):

    def __init__(self, threshold=0.07):
        self.threshold = threshold

    def add_arguments(self, parser):
        action = parser.add_argument("-t", "--threshold",
                                     dest="threshold", type=float, default=self.threshold,
                                     help="default: %0.3f" % self.threshold)
        return [action]


class ConverterOps(Ops):

    def __init__(self,
                 input_help="path to input file",
                 output_help="path to output file",
                 config_help="path to parameters.cfg of the scene"):
        self.input_help = input_help
        self.output_help = output_help
        self.config_help = config_help

    def add_arguments(self, parser):
        actions = list()
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
        for flag, name, help_text in self.optional_input:
            actions.append(parser.add_argument(flag, dest=name, type=str, help=help_text))
        return actions


class FigureOps(Ops):

    def __init__(self, figure_options):
        self.figure_options = figure_options

    def add_arguments(self, parser):
        options = "".join("\n  %s: %s" % (k, v) for k, v in sorted(self.figure_options.items()))
        action = parser.add_argument("-f",
                                     dest="figure_options", action=FigureAction,
                                     figure_options=self.figure_options,
                                     type=str, nargs="+",
                                     help='list of figure names\n'
                                          'example: "-a heatmaps radar"\n'
                                          'default: all options\n'
                                          'options: %s' % options)
        return [action]


class FigureAction(argparse.Action):

    def __init__(self, option_strings, figure_options, *args, **kwargs):
        self.figure_options = figure_options
        super(FigureAction, self).__init__(option_strings=option_strings, *args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        options = []
        available_options = sorted(self.figure_options.keys())

        if not values:
            options = available_options
        else:
            for value in values:
                if value in available_options:
                    options.append(value)
                else:
                    parser.error("Could not find figure option for: %s.\n  "
                                 "Available options are: %s." %
                                 (value, ", ".join(available_options)))

        # save result in action destination
        setattr(namespace, self.dest, options)


class FigureOpsACCV16(FigureOps):

    def __init__(self):
        options = {"heatmaps": "figure with algorithm error heatmap per scene",
                   "radar": "radar charts for stratified and training scenes",
                   "stratified": "metric visualization figure for each stratified scene",
                   "training": "metric visualization figure for each training scene",
                   "backgammon": "fattening and thinning along vertical image dimension",
                   "dots": "background error per box with increasing noise levels",
                   "pyramids": "algorithm disparities vs ground truth disparities on spheres",
                   "stripes": "visualization of evaluation masks"}

        super(FigureOpsACCV16, self).__init__(options)


class FigureOpsCVPR17(FigureOps):

    def __init__(self):
        options = {"scenes": "center view and ground truth per scene",
                   "difficulty": "error map of per pixel median and best disparity per scene",
                   "normalsdemo": "ground truth and algorithm normals + angular error (Sideboard)",
                   "radar": "radar charts for stratified and photorealistic scenes",
                   "badpix": "BadPix series for stratified and photorealistic scenes",
                   "median": "MedianDiff comparisons for stratified and training scenes",
                   "normals": "disparity and normal map + angular error per algorithm (Cotton)",
                   "discont": "disparity map and MedianDiff per algorithm (Bicycle)",
                   "accuracy": "BadPix and Q25 visualizations (Cotton and Boxes)"}

        super(FigureOpsCVPR17, self).__init__(options)
