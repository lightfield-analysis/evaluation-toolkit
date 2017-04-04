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

from utils import plotting
from scenes import BasePhotorealistic


class BaseTest(BasePhotorealistic):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_type():
        return "test"

    @staticmethod
    def get_scenes(data_path=None):
        from scenes import Herbs, Bedroom, Bicycle, Origami
        return [Herbs(data_path=data_path), Bedroom(data_path=data_path),
                Bicycle(data_path=data_path), Origami(data_path=data_path)]

    @staticmethod
    def hidden_gt():
        return True

    def is_test(self):
        return True

    @staticmethod
    def plot_radar_chart(algo_names):
        fig_path = plotting.get_path_to_figure("radar_test")
        BasePhotorealistic.plot_radar_chart(algo_names, BaseTest.get_scenes(), fig_path)


class Bedroom(BaseTest):
    def __init__(self, img_name="bedroom", **kwargs):
        super(Bedroom, self).__init__(img_name, **kwargs)


class Bicycle(BaseTest):
    def __init__(self, img_name="bicycle", **kwargs):
        super(Bicycle, self).__init__(img_name, **kwargs)


class Herbs(BaseTest):
    def __init__(self, img_name="herbs", **kwargs):
        super(Herbs, self).__init__(img_name, **kwargs)


class Origami(BaseTest):
    def __init__(self, img_name="origami", **kwargs):
        super(Origami, self).__init__(img_name, **kwargs)
