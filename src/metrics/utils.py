import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


def xywh2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2


def draw_fig(dictionary, window_title, plot_title, x_label, plot_color='royalblue'):
    """
    source https://github.com/Cartucho/mAP

    :param plt:
    :param dictionary:
    :param window_title:
    :param plot_title:
    :param x_label:
    :param plot_color:
    :return:
    """

    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)

    n_classes = len(dictionary)

    plt.barh(range(n_classes), sorted_values, color=plot_color)
    """
     Write number on side of bar
    """
    fig = plt.gcf()  # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
        str_val = " " + str(val)  # add a space before
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
        # re-set axes to show number inside the figure
        if i == (len(sorted_values) - 1):  # largest bar
            _adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15  # in percentage of the figure height
    bottom_margin = 0.05  # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()

    return fig


def _adjust_axes(r, t, fig, axes):
    """
    source https://github.com/Cartucho/mAP
    :param r:
    :param t:
    :param fig:
    :param axes:
    :return:
    """

    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def fig_to_numpy(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def plot_average_precision_on_tensorboard(average_precision_for_classes, mAP, summary_writer, global_step=0):
    fig = draw_fig(average_precision_for_classes,
                   window_title="mAP",
                   plot_title="mAP = {0:.2f}%".format(mAP * 100),
                   x_label="Average Precision"
                   )
    data = fig_to_numpy(fig)
    summary_writer.add_image('Average_Precision', data, global_step, dataformats='HWC')
