import cv2
import math
import numpy as np
import pandas as pd
from pathlib import Path


def recursive_rmdir(dir_path: str | Path):
    """
    Function for recursively clearing a directory (removing all nested files and dirs)

    Args:
        dir_path: path to the directory that will be cleared
    """
    try:
        path = Path(dir_path)
        if path.is_dir():
            for entry in path.iterdir():
                if entry.is_file():
                    entry.unlink()
                else:
                    recursive_rmdir(entry)
        path.rmdir()
    except PermissionError as e:
        print(f"Insufficient rights to delete. Error message: {e}")
    except FileNotFoundError:
        print(f"File not found: {dir_path}")


def clear_dir(dir_path: str | Path):
    """
    Function for recursively clearing a directory (removing all nested files and dirs)

    Args:
        dir_path: path to the directory that will be cleared
    """
    try:
        path = Path(dir_path)
        if path.is_dir():
            for entry in path.iterdir():
                if entry.is_file():
                    entry.unlink()
                else:
                    recursive_rmdir(entry)
    except PermissionError as e:
        print(f"Insufficient rights to delete. Error message: {e}")
    except FileNotFoundError:
        print(f"File not found: {dir_path}")


def clear_or_create_dir(dir_path: str | Path):
    """
    Function to clear a directory or create a new one if the specified directory does not exist

    Args:
        dir_path: path to the directory that will be cleared or created
    """
    try:
        path = Path(dir_path)
        if path.is_dir():
            clear_dir(path)
        else:
            path.mkdir(parents=False, exist_ok=True)
    except PermissionError as e:
        print(f"Insufficient rights to delete. Error message: {e}")
    except FileNotFoundError:
        print(f"File not found: {dir_path}")


def crop_image_white_margins(old_filename: str | Path, xpadding: int = 15, ypadding: int = 15,
                             new_filename: str | Path | None = None) -> None:
    """
    Function for cropping images with graphs (white margins at the edges are cut off).
    If a new filename is not passed, the original file will be overwritten

    Args:
        old_filename: number of bytes
        xpadding: horizontal padding
        ypadding: vertical padding
        new_filename: path to the new (cropped) image
    """
    img = cv2.imread(old_filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = 255 * (gray < 128).astype(np.uint8)
    cords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(cords)
    rect = img[y - ypadding: y + h + 2 * ypadding, x - xpadding: x + w + 2 * xpadding]
    is_success, im_buf_arr = cv2.imencode(".png", rect)
    if new_filename is None:
        Path(old_filename).unlink()
        im_buf_arr.tofile(old_filename)
    else:
        im_buf_arr.tofile(new_filename)


def format_xlsx(writer: pd.ExcelWriter, df: pd.DataFrame, alignments: str | None = None,
                sheet_name: str = 'Sheet1', cell_height: int = 20) -> pd.ExcelWriter:
    """
    Function for formatting an object of XlsxWriter type.
    Allows to set alignment for each column and adjust cells height.

    Args:
        writer: object of XlsxWriter type
        df: pandas dataframe with data
        alignments: string indicating columns alignments (r, l, c, j), default is left alignment for all columns
        sheet_name: name of the sheet to be formatted
        cell_height: cell height
    """
    if df.shape[0] > 0:
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        if alignments is None:
            alignments = 'l' * df.shape[1]
        # set column width and alignment
        a = {'l': 'left', 'r': 'right', 'c': 'center', 'j': 'justify'}
        for col_index, col_name in enumerate(df.columns):
            col_width = max(len(col_name), max(len(str(r)) for r in df[col_name])) + 1
            cell_format = workbook.add_format()
            cell_format.set_align(a[alignments[col_index]])
            worksheet.set_column(col_index, col_index, col_width, cell_format)
        # set cells height
        for i in range(len(df) + 1):
            worksheet.set_row(i, cell_height)
    return writer


def get_tick_bounds(max_val: float, min_val: float = 0) -> list:
    """
    Dummy function for generating a list of labels on the chart axis.
    The labels step is calculated from the value min and max values on the chart.
    The number of labels is calculated from the step.

    Args:
        max_val: max value on the chart
        min_val: min value on the chart

    Returns:
        list: list of values [min value, max value, number of labels]
    """
    min_val, max_val = math.floor(min_val), math.ceil(max_val)
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
