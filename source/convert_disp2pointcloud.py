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


from utils.option_parser import OptionParser, ConverterOpsExt


if __name__ == "__main__":
    parser = OptionParser([ConverterOpsExt(input="path to disparity map",
                                           output="path to point cloud",
                                           optional_input=[("-c", "color_map_file",
                                                            "path to color map, "
                                                            "e.g. to center view of the scene")])])

    disp_map_path, config_path, point_cloud_path, color_map_path = parser.parse_args()

    from scenes import PhotorealisticScene
    from utils import file_io, point_cloud

    scene = PhotorealisticScene("demo", path_to_config=config_path)
    disp_map = file_io.read_file(disp_map_path)

    if color_map_path:
        color_map = file_io.read_file(color_map_path)
    else:
        color_map = None

    pc = point_cloud.convert(scene, disp_map, color_map)
    point_cloud.save(pc, point_cloud_path)