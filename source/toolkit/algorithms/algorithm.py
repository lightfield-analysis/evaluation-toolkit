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


from toolkit import settings
from toolkit.utils import file_io


class Algorithm(object):

    def __init__(self, file_name, display_name=None,
                 is_baseline=False, is_meta=False,
                 color=None, line_style=None):

        self.file_name = file_name
        self.display_name = file_name.upper() if display_name is None else display_name

        # algorithm category
        self.is_baseline_algorithm = is_baseline
        self.is_meta_algorithm = is_meta

        # plotting properties
        self.color = color
        if line_style is None:
            line_style = (0, (1, 1)) if self.is_meta_algorithm else "-"
        self.line_style = line_style

    def __lt__(self, other):
        return self.get_display_name() < other.get_display_name()

    def __str__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        return self.file_name

    def get_display_name(self):
        return self.display_name

    def is_baseline(self):
        return self.is_baseline_algorithm

    def is_meta(self):
        return self.is_meta_algorithm

    def get_color(self):
        if self.color is None:
            return settings.get_color(0)
        return self.color

    def get_line_style(self):
        return self.line_style

    @staticmethod
    def set_colors(algorithms, offset=0):
        for idx_a, algorithm in enumerate(algorithms):
            algorithm.color = settings.get_color(idx_a+offset)
        return algorithms

    @staticmethod
    def initialize_algorithms(file_names_algorithms,
                              set_colors=True, is_baseline=False, is_meta=False):
        try:
            meta_data = file_io.read_file(settings.PATH_TO_ALGO_META_DATA)
        except IOError:
            meta_data = dict()

        algorithms = []
        for file_name in file_names_algorithms:
            algorithm = Algorithm(file_name=file_name, is_baseline=is_baseline, is_meta=is_meta)

            algo_data = meta_data.get(file_name, dict())
            display_name = algo_data.get("acronym", None)
            if display_name:
                algorithm.display_name = display_name

            algorithms.append(algorithm)

        if set_colors:
            algorithms = Algorithm.set_colors(algorithms)

        return algorithms
