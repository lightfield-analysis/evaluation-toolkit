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

from scenes import BasePhotorealistic


class BaseAdditional(BasePhotorealistic):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def get_type():
        return "additional"

    @staticmethod
    def get_scenes(data_path=None):
        from scenes import Boardgames, Kitchen, Medieval2, Museum, Pens, Pillows, \
            Platonic, Rosemary, Table, Tomb, Town, Vinyl
        return [Boardgames(data_path=data_path), Kitchen(data_path=data_path), Medieval2(data_path=data_path),
                Museum(data_path=data_path), Pens(data_path=data_path), Pillows(data_path=data_path),
                Platonic(data_path=data_path), Rosemary(data_path=data_path), Table(data_path=data_path),
                Tomb(data_path=data_path), Town(data_path=data_path), Vinyl(data_path=data_path)]


class Boardgames(BaseAdditional):
    def __init__(self, img_name="boardgames", **kwargs):
        super(Boardgames, self).__init__(img_name, **kwargs)


class Kitchen(BaseAdditional):
    def __init__(self, img_name="kitchen", **kwargs):
        super(Kitchen, self).__init__(img_name, **kwargs)


class Medieval2(BaseAdditional):
    def __init__(self, img_name="medieval2", **kwargs):
        super(Medieval2, self).__init__(img_name, **kwargs)


class Museum(BaseAdditional):
    def __init__(self, img_name="museum", **kwargs):
        super(Museum, self).__init__(img_name, **kwargs)


class Pens(BaseAdditional):
    def __init__(self, img_name="pens", **kwargs):
        super(Pens, self).__init__(img_name, **kwargs)


class Pillows(BaseAdditional):
    def __init__(self, img_name="pillows", **kwargs):
        super(Pillows, self).__init__(img_name, **kwargs)


class Platonic(BaseAdditional):
    def __init__(self, img_name="platonic", **kwargs):
        super(Platonic, self).__init__(img_name, **kwargs)


class Rosemary(BaseAdditional):
    def __init__(self, img_name="rosemary", **kwargs):
        super(Rosemary, self).__init__(img_name, **kwargs)


class Table(BaseAdditional):
    def __init__(self, img_name="table", **kwargs):
        super(Table, self).__init__(img_name, **kwargs)


class Tomb(BaseAdditional):
    def __init__(self, img_name="tomb", **kwargs):
        super(Tomb, self).__init__(img_name, **kwargs)


class Town(BaseAdditional):
    def __init__(self, img_name="town", **kwargs):
        super(Town, self).__init__(img_name, **kwargs)


class Vinyl(BaseAdditional):
    def __init__(self, img_name="vinyl", **kwargs):
        super(Vinyl, self).__init__(img_name, **kwargs)