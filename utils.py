
import os
import cv2
import shutil
import numpy as np
import pandas as pd
from math import floor, ceil


def clear_dir(path):
    """
    Function for recursively clearing a directory (removing all nested files)

    Args:
        param path (str): directory path
    """
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and entry.is_file():
            os.unlink(entry.path)
        elif not entry.name.startswith('.') and entry.is_dir():
            shutil.rmtree(entry.path)


def create_or_clean_dir(path):
    """
    Function to create an empty directory, if the directory exists it is cleared

    Args:
        param path (str): directory path
    """
    if os.path.exists(path):
        clear_dir(path)
    else:
        os.mkdir(path)


def crop_image(old_filename, new_filename):
    """
    Function for cropping images with graphs (white margins are cropped)

    Args:
        param old_filename (str): path to source image
        param new_filename (str): path to new (cropped) image
    """
    img = cv2.imread(old_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    cords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(cords)
    padding = 15
    rect = img[y - padding: y + h + 2 * padding, x - padding: x + w + 2 * padding]
    os.remove(old_filename)
    is_success, im_buf_arr = cv2.imencode(".png", rect)
    im_buf_arr.tofile(new_filename)


def format_xlsx(writer, df, alignments, sheet_name='Sheet1', line_height=20):
    """
    Function for formatting an object of type XlsxWriter

    Args:
        param writer (pandas.io.excel._xlsxwriter._XlsxWriter): object of type XlsxWriter
        param df (pandas.core.frame.DataFrame): pandas dataframe with data
        param alignments (str): string indicating column alignment (r, l, c, j)
        param sheet_name (str): sheet name
        param line_height (int): cell height
    """
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]
    header_list = df.columns.values.tolist()
    # set column width and alignment
    a = {'l': 'left', 'r': 'right', 'c': 'center', 'j': 'justify'}
    for i in range(len(header_list)):
        cw = max([len(str(r)) for r in df[header_list[i]]])
        hw = max(len(header_list[i]), cw) + 5
        cell_format = workbook.add_format()
        cell_format.set_align(a[alignments[i]])
        worksheet.set_column(i, i, hw, cell_format)
    # set cell height
    for i in range(len(df) + 1):
        worksheet.set_row(i, line_height)
    return writer


def get_tick_bounds(max_val, min_val=0):
    """
    Function for generating a list of labels on the chart axis.
    The label step is calculated from the value range, and the number of labels is calculated from the step.

    Args:
        param max_val (float): maximum value of quantity
        param min_val (float): maximum value of quantity

    Returns:
        list: list of values [min, max, number of labels]
    """
    min_val, max_val = floor(min_val), ceil(max_val)
    dif = max_val - min_val
    if dif > 10000000:
        step = 1000000
    elif dif > 5000000:
        step = 500000
    elif dif > 2000000:
        step = 250000
    elif dif > 1000000:
        step = 100000
    elif dif > 500000:
        step = 50000
    elif dif > 200000:
        step = 20000
    elif dif > 100000:
        step = 10000
    elif dif > 50000:
        step = 5000
    elif dif > 20000:
        step = 2000
    elif dif > 10000:
        step = 1000
    elif dif > 5000:
        step = 500
    elif dif > 2000:
        step = 200
    elif dif > 1000:
        step = 100
    elif dif > 500:
        step = 50
    elif dif > 200:
        step = 20
    elif dif > 50:
        step = 10
    elif dif > 10:
        step = 5
    else:
        step = 1
    if min_val % step != 0:
        min_val = (min_val // step) * step
    if max_val % step != 0:
        max_val = ((max_val // step) + 1) * step
    return [min_val, max_val, (max_val - min_val) // step + 1]
