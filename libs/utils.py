import cv2
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union


def recursive_rmdir(directory: Union[str, Path]) -> None:
    """
    Function for recursively clearing a directory (removing all nested files and dirs)

    Args:
        directory: path to the directory that will be cleared
    """
    try:
        path = Path(directory)
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
        print(f"File not found: {directory}")


def clear_dir(directory: Union[str, Path]) -> None:
    """
    Function for recursively clearing a directory (removing all nested files and dirs)

    Args:
        directory: path to the directory that will be cleared
    """
    try:
        path = Path(directory)
        if path.is_dir():
            for entry in path.iterdir():
                if entry.is_file():
                    entry.unlink()
                else:
                    recursive_rmdir(entry)
    except PermissionError as e:
        print(f"Insufficient rights to delete. Error message: {e}")
    except FileNotFoundError:
        print(f"File not found: {directory}")


def clear_or_create_dir(directory: Union[str, Path]) -> None:
    """
    Function to clear a directory or create a new one if the specified directory does not exist

    Args:
        directory: path to the directory that will be cleared or created
    """
    try:
        path = Path(directory)
        if path.is_dir():
            clear_dir(path)
        else:
            path.mkdir(parents=False, exist_ok=True)
    except PermissionError as e:
        print(f"Insufficient rights to delete. Error message: {e}")
    except FileNotFoundError:
        print(f"File not found: {directory}")


def crop_image_white_margins(old_filename: Union[str, Path], xpadding: int = 15, ypadding: int = 15,
                             new_filename: Optional[Union[str, Path]] = None) -> None:
    """
    Function for cropping images with graphs (white margins at the edges are cut off).
    If a new filename is not passed, the original file will be overwritten

    Args:
        old_filename: number of bytes
        xpadding: horizontal padding
        ypadding: vertical padding
        new_filename: path to the new (cropped) image
    """
    try:
        img = cv2.imread(str(old_filename))
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
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    except ValueError as e:
        print(f"Image reading error: {e}")
    except IOError as e:
        print(f"IO error: {e}")


def format_xlsx(writer: pd.ExcelWriter, df: pd.DataFrame, alignments: str | None = None,
                sheet_name: str = 'Sheet1', font_size: Optional[int] = None, border_width: Optional[int] = None,
                border_color: Optional[str] = None, cell_height: int = 20) -> pd.ExcelWriter:
    """
    Formats an object of XlsxWriter type.
    Allows to set alignment for each column and adjust cells height.

    Args:
        writer: object of XlsxWriter type
        df: pandas dataframe with data
        alignments: string indicating columns alignments (r, l, c, j), default is left alignment for all columns
        sheet_name: name of the sheet to be formatted
        font_size: font size for all cells
        border_width: border width for all cells
        border_color: border color for all cells
        cell_height: cell height for all cells
    """
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
        if font_size:
            cell_format.set_font_size(font_size)
        if border_width:
            cell_format.set_border(border_width)
        if border_color:
            cell_format.set_border_color(border_color)
        worksheet.set_column(col_index, col_index, col_width, cell_format)
    # set cells height
    for i in range(len(df) + 1):
        worksheet.set_row(i, cell_height)
    return writer


def get_tick_bounds(max_val: float, min_val: float = 0) -> list[int]:
    """
    Dummy function for generating a list of labels on the chart axis.
    The labels step is calculated from the value min and max values on the chart.
    The number of labels is calculated from the step.

    Args:
        max_val: max value on the chart
        min_val: min value on the chart

    Returns:
        list of int values: min value, max value, number of labels
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
