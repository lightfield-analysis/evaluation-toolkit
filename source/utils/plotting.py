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

import gc
import os.path as op
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid, make_axes_locatable
import numpy as np
import scipy.ndimage.interpolation as sci

import file_io
from logger import log
import misc
import settings


def get_path_to_figure(fig_name):
    return op.join(settings.EVAL_PATH, fig_name + "." + settings.fig_type)


def save_fig(fig, fig_name, dpi=150, bbox_inches='tight', hide_frames=False, remove_ticks=False, **kwargs):
    log.info("Saving figure...")

    if remove_ticks:
        remove_ticks_from_axes(fig.get_axes())
    if hide_frames:
        remove_frames_from_axes(fig.get_axes())

    file_io.check_dir_for_fname(fig_name)
    plt.savefig(fig_name, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    fig.clf()
    plt.close(fig)
    gc.collect()
    log.info('saved: ' + fig_name)


def save_tight_figure(fig, fig_name, remove_ticks=True, wspace=0.0, hspace=0.0,
                      pad_inches=0.1, padding_top=0.92, hide_frames=False):
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=hspace, top=padding_top)
    save_fig(fig, fig_name, pad_inches=pad_inches, hide_frames=hide_frames, remove_ticks=remove_ticks)


def remove_ticks_from_axes(axes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])


def remove_frames_from_axes(axes):
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)


def hide_upper_right():
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


# adds a colorbar to a separate grid field
def add_colorbar(idx, cm, height, width, colorbar_bins=8, fontsize=None, img_width=1, scale=0.9):
    axis = plt.subplot(idx)
    plt.imshow(np.ones((height*scale, img_width)), alpha=0)

    # colorbar width must be given as a percentage of the img width, both together should be equal to w/wscale
    width_factor = 100 * (width - img_width) / float(img_width)

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size=str(width_factor)+"%", pad=0.0)
    create_colorbar(cm, cax, colorbar_bins, fontsize)


def create_colorbar(cm, cax, colorbar_bins=8, fontsize=None, linewidth=0):
    cb = plt.colorbar(mappable=cm, cax=cax)

    if fontsize is not None:
        cb.ax.tick_params(labelsize=fontsize)

    cb.outline.set_linewidth(linewidth)
    cb.locator = ticker.MaxNLocator(nbins=colorbar_bins)
    cb.update_ticks()


def get_grids(fig, rows, cols, axes_pad=0):
    grids = []
    for row in range(rows):
        grid_id = int("%d%d%d" % (rows, 1, row + 1))
        grid = ImageGrid(fig, grid_id,
                         nrows_ncols=(1, cols),
                         axes_pad=(0.05, axes_pad),
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="10%",
                         cbar_pad="5%")
        grids.append(grid)
    return grids


def adjust_binary_vis(vis):
    return 1.0 * vis + ~vis * 0.2


def pixelize(data, factor=0.1, order=0, mode="nearest", add_noise=True, noise_factor=0.3):
    h, w = np.shape(data)[0:2]

    if add_noise:
        noise = noise_factor * np.random.random(np.shape(data)) - 0.5*noise_factor
        data += noise

    factor_h = h / float(int(h)/(int(1/factor)))
    factor_w = w / float(int(w)/(int(1/factor)))
    small = sci.zoom(data, factor, order=order, mode=mode)
    pixelized = sci.zoom(small, [factor_h, factor_w], order=order, mode=mode)

    return pixelized


def plot_img_with_transparent_mask(img, mask, alpha=0.5, color=(1.0, 0.0, 0.0), cmap="gray"):
    if np.size(np.shape(img)) > 2:
        img = misc.rgb2gray(img)

    plt.imshow(img, cmap=cmap)
    mask_vis = misc.make_rgba(mask)
    mask_vis[:, :, 3] = mask*alpha
    mask_vis[:, :, 0:3] *= color
    plt.imshow(mask_vis)