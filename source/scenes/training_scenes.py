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


class BaseTraining(BasePhotorealistic):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_type():
        return "training"

    @staticmethod
    def get_scenes(data_path=None):
        from scenes import Sideboard, Cotton, Dino, Boxes
        return [Sideboard(data_path=data_path), Cotton(data_path=data_path),
                Dino(data_path=data_path), Boxes(data_path=data_path)]

    @staticmethod
    def plot_radar_chart(algo_names):
        fig_path = plotting.get_path_to_figure("radar_training")
        max_per_metric = [16, 40, 4, 4, 12, 80, 80, 6]
        BasePhotorealistic.plot_radar_chart(algo_names, BaseTraining.get_scenes(), fig_path, max_per_metric)


class Boxes(BaseTraining):
    def __init__(self, img_name="boxes", **kwargs):
        super(Boxes, self).__init__(img_name, **kwargs)


class Cotton(BaseTraining):
    def __init__(self, img_name="cotton", **kwargs):
        super(Cotton, self).__init__(img_name, **kwargs)


class Dino(BaseTraining):
    def __init__(self, img_name="dino", **kwargs):
        super(Dino, self).__init__(img_name, **kwargs)


class Sideboard(BaseTraining):
    def __init__(self, img_name="sideboard", **kwargs):
        super(Sideboard, self).__init__(img_name, **kwargs)





