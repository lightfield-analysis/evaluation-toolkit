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


import distutils.dir_util as du
import json
import os
import os.path as op
import sys
import zipfile

import numpy as np

from toolkit.utils import log


def read_file(src_file, **kwargs):
    src_file = op.normpath(src_file)

    if src_file.endswith('.png') or src_file.endswith('.jpg') or src_file.endswith('.bmp'):
        return read_img(src_file)
    elif src_file.endswith('.json'):
        return read_json(src_file)
    elif src_file.endswith('.pfm'):
        return read_pfm(src_file, **kwargs)
    else:
        raise NotImplementedError('No support for file: %s' % src_file)


def write_file(data, tgt_file, **kwargs):
    check_dir_for_fname(tgt_file)

    if tgt_file.endswith('.png') or tgt_file.endswith('.jpg'):
        write_img(data, tgt_file, **kwargs)
    elif tgt_file.endswith('.json'):
        write_json(data, tgt_file)
    elif tgt_file.endswith('.pfm'):
        write_pfm(data, tgt_file, **kwargs)
    else:
        raise NotImplementedError('No support for file: %s' % tgt_file)
    log.info('Saved %s' % tgt_file)


# standard images

def read_img(fpath):
    from scipy import misc
    data = misc.imread(fpath)
    return data


def write_img(img, fpath, cmax=None):
    from scipy import misc

    if cmax is None:
        cmax = 255
        if np.max(img) <= 1.0 and img.dtype == float:
            cmax = 1.0

    img_conv = misc.toimage(img, cmin=0, cmax=cmax)
    img_conv.save(fpath)


# json

def read_json(fpath):
    with open(fpath, 'r') as f:
        data = json.load(f)
    return data


def write_json(data, fpath, indent=4):
    with open(fpath, 'w') as f:
        json.dump(data, f, indent=indent, sort_keys=True)


# pfm

class PFMExeption(Exception):
    pass


def write_pfm(data, fpath, scale=1, file_identifier="Pf", dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as ff:
        ff.write(file_identifier + '\n')
        ff.write('%d %d\n' % (width, height))
        ff.write('%d\n' % scale)
        ff.write(values)


def read_pfm(fpath, expected_identifier="Pf", print_limit=30):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise PFMExeption('Unknown identifier. '
                              'Expected: "%s", got: "%s".' %
                              (expected_identifier, identifier[:print_limit]))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise PFMExeption('Could not parse dimensions: "%s". '
                              'Expected "width height", e.g. "512 512".' %
                              line_dimensions[:print_limit])

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise PFMExeption('Could not parse max value / endianess information: "%s". '
                              'Should be a non-zero number.' % line_scale[:print_limit])

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise PFMExeption('Invalid binary values. '
                              'Could not create %dx%d array from input.' % (height, width))

        return data


def _get_next_line(ff):
    try:
        next_line = ff.readline().rstrip()
        while next_line.startswith('#'):  # ignore comments
            next_line = ff.readline().rstrip()
    except:
        raise PFMExeption('Incomplete file.')
    return next_line


# runtimes


def read_runtime(fname):
    with open(fname, "r") as f:
        try:
            line_runtime = f.readline()
            runtime = float(line_runtime)
        except Exception as e:
            raise IOError('"%s"\n%s' % (line_runtime, e))
    return runtime


def write_runtime(runtime, fname):
    check_dir_for_fname(fname)
    with open(fname, "w") as f:
        f.write("%0.10f" % runtime)


# misc


def unzip(fname_zip, tgt_dir=None):
    if tgt_dir is None:
        tgt_dir = op.abspath(op.join(fname_zip, os.pardir))

    check_dir_for_fname(fname_zip)

    with zipfile.ZipFile(fname_zip, "r") as zf:
        zf.extractall(tgt_dir)


def check_dir_for_fname(tgt_file):
    path, file_name = op.split(tgt_file)
    check_dir(path)


def check_dir(tgt_dir):
    create_dir(tgt_dir)


def create_dir(path):
    if not op.isdir(path):
        du.mkpath(path)
