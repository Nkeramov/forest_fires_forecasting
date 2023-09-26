import re
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import pandas.core.frame
import matplotlib as mpl
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

from utils import clean_or_create_dir, crop_image, format_xlsx, get_tick_bounds
from logging_utils import get_logger

mpl.rcParams.update({'font.size': 14})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

HTTP_REQUEST_TIMEOUT = 10
HTTP_REQUEST_DELAY = 0.25
HTTP_REQUEST_RETRY_DELAY = 3
HTTP_RETRIES_COUNT = 3
INPUT_PATH = './input'
OUTPUT_PATH = './output'
WINDOW_SIZE = 7
IMG_WIDTH, IMG_HEIGHT, IMG_DPI = 3600, 2000, 150
FLOAT_NUMBER_REGEX = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
# weather site url
WEATHER_URl = "http://pogodaiklimat.ru/monitor.php"

# IDs of cities from weather site
cities = {
    'Khanty-Mansiysk': 23933,
    'October': 23734,
    'Leushi': 28064,
    'Lariak': 23867,
    'Ugut': 23946
}
statistic_cols = {
    'Number (units)': 'int32',
    'Area (ha)': 'float32',
    'Forest area (ha)': 'float32',
    'Year': 'int32'
}

LOG_LEVEL = "INFO"
LOG_MSG_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_FILENAME = "forecast.log"
logger = get_logger(LOG_LEVEL, LOG_MSG_FORMAT, LOG_DATE_FORMAT, LOG_FILENAME)


def get_weather_data(city_name: str, start_year=2000, end_year=2021):
    """
    Function for obtaining weather data for specified city.
    Two files are generated: average monthly temperatures and average monthly precipitations.

    Args:
        city_name (str): city for which data is retrieved (name from cities list).
        start_year (int): year from which data is retrieved.
        end_year (int): year for which data is retrieved.

    Returns:
        int: -1 if an error occurred and 0 if there were no errors.
    """
    writer = pd.ExcelWriter(f"{OUTPUT_PATH}/{city_name}/weather.xlsx", engine='xlsxwriter')
    df_temperatures = pd.DataFrame({'Month': range(1, 13)}, columns=['Month'])
    df_precipitations = pd.DataFrame({'Month': range(1, 13)}, columns=['Month'])
    times = list(((year, month) for year in range(start_year, end_year) for month in range(1, 13)))
    temperatures = []
    precipitations = []
    ua = UserAgent()
    for (year, month) in tqdm(times, total=len(times), colour='green', desc=f"\tRetrieving weather data "
                                                                            f"for {city_name}", position=0,
                              leave=True, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        for k in range(HTTP_RETRIES_COUNT):
            try:
                payload = {'id': cities[city_name], 'month': month, 'year': year}
                header = {'User-Agent': ua.random}
                response = requests.get(WEATHER_URl, headers=header, params=payload, timeout=HTTP_REQUEST_TIMEOUT)
                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'lxml')
                    tags = soup.find_all(['div'], class_='climate-text')
                    text = re.sub(r"\s+", " ", tags[1].text.strip())
                    res = re.findall(FLOAT_NUMBER_REGEX, text)
                    # take the second and fifth value, this is determined by the markup of the site page
                    temperatures.append(float(res[1]) if len(res[1]) > 0 else 0)
                    precipitations.append(float(res[4]) if len(res[4]) > 0 else 0)
                    break
                else:
                    time.sleep(HTTP_REQUEST_RETRY_DELAY)
            except requests.exceptions.HTTPError as err:
                logger.warning(f"\tHTTP error, {city_name} {month}.{year}, {err}. Retrying...")
            except requests.exceptions.ConnectionError as err:
                logger.warning(f"\tConnection error, {city_name} {month}.{year}, {err}. Retrying...")
            except requests.exceptions.Timeout as err:
                logger.warning(f"\tTimeout error, {city_name} {month}.{year}, {err}. Retrying...")
            except requests.exceptions.RequestException | Exception as err:
                logger.warning(f"\tError, {city_name} {month}.{year}, {err}. Retrying...")
            if k < HTTP_RETRIES_COUNT - 1:
                time.sleep(HTTP_REQUEST_RETRY_DELAY)
            else:
                logger.error(f"\tError, {city_name} {month}.{year}. Maximum number of request retries reached")
                return -1
        if month == 12:
            df_temperatures[str(year)] = temperatures
            df_precipitations[str(year)] = precipitations
            temperatures.clear()
            precipitations.clear()
        time.sleep(HTTP_REQUEST_DELAY)
    df_temperatures.to_excel(excel_writer=writer, sheet_name='Temperature', header=True, index=False)
    df_precipitations.to_excel(excel_writer=writer, sheet_name='Precipitations', header=True, index=False)
    writer = format_xlsx(writer, df_temperatures, 'c' * (end_year - start_year + 1), 'Temperature')
    writer = format_xlsx(writer, df_temperatures, 'c' * (end_year - start_year + 1), 'Precipitations')
    writer.close()
    return 0


def get_full_data(city_name: str):
    """
    A function to get the complete dataset for a specified city.
    Combines a dataset from fire statistics and a weather dataset.

    Args:
        city_name (str): city for which data is returned (from cities list).

    Returns:
        pandas.core.frame.DataFrame: dataframe (fires + weather).
    """
    stats_df = pd.read_excel(f"{INPUT_PATH}/statistics.xlsx", names=list(statistic_cols.keys()), dtype=statistic_cols,
                             sheet_name='Sheet1', engine='openpyxl')
    df_temperatures = pd.read_excel(f"{OUTPUT_PATH}/{city_name}/weather.xlsx", sheet_name='Temperature',
                                    engine='openpyxl')

    sum_df_temperature = df_temperatures.loc[(df_temperatures['Month'].between(5, 8, inclusive='both'))] \
        .sum(axis=0, skipna=True).reset_index()

    sum_df_temperature.columns = ['Year', 'Accumulated temperature']
    sum_df_temperature.drop(index=0, axis=0, inplace=True)
    sum_df_temperature.reset_index(inplace=True)

    df_precipitations = pd.read_excel(f"{OUTPUT_PATH}/{city_name}/weather.xlsx", sheet_name='Precipitations',
                                      engine='openpyxl')
    sum_df_precipitations = df_precipitations.loc[(df_precipitations['Month'].between(5, 8, inclusive='both'))] \
        .sum(axis=0, skipna=True).reset_index()
    sum_df_precipitations.columns = ['Year', 'Accumulated precipitations']
    sum_df_precipitations.drop(index=0, axis=0, inplace=True)
    sum_df_precipitations.reset_index(inplace=True)
    stats_df['Accumulated temperature'] = list(sum_df_temperature['Accumulated temperature'])
    stats_df['Accumulated precipitations'] = list(sum_df_precipitations['Accumulated precipitations'])
    return stats_df.copy()


def plot_trends(city_name: str):
    """
    Function for plotting graphs with trends for a specified city. Brings data to a scale from 0 to 100 and plots.
    Needed to visualization and explore the dependence of fires on weather data.

    Args:
        city_name: city for which graphs are plotting (from cities list).
    """
    data = get_full_data(city_name)
    x = data['Year'].tolist()
    data.drop(['Year'], axis=1, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    fig = plt.figure(dpi=IMG_DPI, figsize=(IMG_WIDTH / IMG_DPI, IMG_HEIGHT / IMG_DPI))
    plt.clf()
    plt.title("Statistics for natural fires in Khanty-Mansi Autonomous Okrug-Yugra from 2000 to 2020", fontsize=24)
    plt.xlabel("year", fontsize=18)
    plt.ylabel("scaled value (from 0 to 100)", fontsize=18)
    y1 = df_scaled['Accumulated temperature'].tolist()
    y2 = df_scaled['Accumulated precipitations'].tolist()
    y3 = df_scaled['Area (ha)'].tolist()
    plt.plot(x, y1, color='red', linestyle='solid', lw=2, label='Accumulated temperature from May to August')
    plt.plot(x, y2, color='blue', linestyle='solid', lw=2, label='Accumulated precipitations from May to August')
    plt.plot(x, y3, color='green', linestyle='solid', lw=2, label='Fire area (ha)')
    maxvalue = max(max(y1), max(y2), max(y3))
    b = get_tick_bounds(maxvalue, 0)
    plt.xticks(x, fontsize=14)
    plt.yticks(np.linspace(start=b[0], stop=b[1], num=b[2], dtype=np.int32), fontsize=14)
    plt.grid(axis='both', linestyle='--')
    plt.legend(loc='upper left', fontsize=24)
    fig.savefig('img.png')
    crop_image('img.png', f"{OUTPUT_PATH}/{city_name}/trends.png")


def get_regression_regularity(data: pandas.core.frame.DataFrame, indicator='Area (ha)'):
    """
    Function for correlation analysis. Allows to explore the dependence of the selected indicator
    (fires area, forest fires area, fires number) on the values of the accumulated temperature and precipitations.
    Results are displayed on the screen as polynomial coefficient values. Functions fires_number_extrapolation_func
    and fires_area_extrapolation_func are created based on the results of this function.

    Args:
        data (pandas.core.frame.DataFrame): dataframe (fires + weather).
        indicator (str): indicator for which regression analysis is performed.
    """
    logger.info(indicator)
    x, y = np.array(data['Accumulated temperature']), np.array(data['Accumulated precipitations (2 years)'])
    z = np.array(data[indicator], dtype=np.float64)
    x, y, z = np.meshgrid(x, y, z, copy=False)
    x, y = x.flatten(), y.flatten()
    a = np.array([x * 0 + 1, x, y, x * y, x ** 2, y ** 2, (x ** 2) * y, x * (y ** 2), (x ** 2) * (y ** 2)]).T
    b = z.flatten()
    coeff, r, rank, s = np.linalg.lstsq(a, b, rcond=None)
    logger.info('\t', [round(x, 6) for x in coeff])


def fires_number_extrapolation_func(x: list, a: float, b: float, c: float) -> list:
    """
    Extrapolation function for the number of fires.

    Args:
        x (list): list with temperature and precipitations values (lists).
        a (float): polynomial coefficient.
        b (float): polynomial coefficient.
        c (float): polynomial coefficient.

    Returns:
        list: function values.
    """
    return a + b * x[0] + c * x[1]


def fires_area_extrapolation_func(x: list, a: float, b: float, c: float, d: float, e: float, f: float) -> list:
    """
    Extrapolation function for the area of fires.

    Args:
        x (list): list with temperature and precipitations values (lists).
        a (float): polynomial coefficient.
        b (float): polynomial coefficient.
        c (float): polynomial coefficient.
        d (float): polynomial coefficient.
        e (float): polynomial coefficient.
        f (float): polynomial coefficient.

    Returns:
        list: function values.
    """
    return a + b * x[0] + c * x[1] + d * x[0] * x[1] + e * (x[0] ** 2) + f * (x[1] ** 2)


def get_forecasts(city_name: str, show_last_year=False):
    """
    The function of obtaining forecasts. For each city from the cities list, three forecasts are generated:
    for the total area covered by fire, for the forest area covered by fire, for the number of fires.
    The result is graphs and xlsx-report with initial data and forecast.

    Args:
        city_name (str): city for which forecast is returned (from cities list).
        show_last_year (bool): if the source data contains a value for the forecast year, then it will be included.
    """
    df = get_full_data(city_name)
    precipitations_2years = []
    # calculation of accumulated precipitations for the last two years
    for k in range(len(df) - 1, -1, -1):
        if k > 0:
            precipitations_2years.insert(0, df.iloc[k]['Accumulated precipitations'] +
                                         df.iloc[k - 1]['Accumulated precipitations'])
        else:
            precipitations_2years.insert(0, 2 * df.iloc[k]['Accumulated precipitations'])
    df['Accumulated precipitations (2 years)'] = precipitations_2years
    years = df['Year'].tolist()
    indicators = ['Forest area (ha)', 'Area (ha)', 'Number (units)']
    for indicator in indicators:
        x = [np.array(df['Accumulated temperature'], dtype=np.float64),
             np.array(df['Accumulated precipitations (2 years)'], dtype=np.float64)]
        y = np.array(df[indicator], dtype=np.float64)
        if indicator in ['Forest area (ha)', 'Area (ha)']:
            popt, pcov = curve_fit(fires_area_extrapolation_func, x, y, maxfev=100000)[:2]
        else:
            popt, pcov = curve_fit(fires_number_extrapolation_func, x, y, maxfev=100000)[:2]
        fig = plt.figure(dpi=IMG_DPI, figsize=(IMG_WIDTH / IMG_DPI, IMG_HEIGHT / IMG_DPI))
        plt.clf()
        plt.title("Statistics and forecast for natural fires in Khanty-Mansi Autonomous Okrug-Yugra "
                  "from 2000 to 2020", fontsize=24)
        plt.xlabel("year", fontsize=18)
        plt.ylabel(indicator.lower(), fontsize=18)
        if show_last_year:
            plt.plot(years, y, color='red', linestyle='solid', lw=2, label='Actual area')
        else:
            plt.plot(years[:-1], y[:-1], color='red', linestyle='solid', lw=2, label='Actual area')
        if indicator in ['Forest area (ha)', 'Area (ha)']:
            p = fires_area_extrapolation_func(x, *popt)
        else:
            p = fires_number_extrapolation_func(x, *popt)
        r2 = round(r2_score(y, p), 2)
        plt.plot(years, p, color='green', linestyle='solid', lw=2, label=f"Forecast  R² = {r2}")
        maxvalue = max(max(y), max(p))
        b = get_tick_bounds(maxvalue, 0)
        plt.xticks(years, fontsize=14)
        plt.yticks(np.linspace(start=b[0], stop=b[1], num=b[2], dtype=np.int32), fontsize=14)
        plt.gca().ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.grid(axis='both', linestyle='--')
        plt.legend(loc='upper left', fontsize=24)
        fig.savefig('img.png')
        crop_image('img.png', f"{OUTPUT_PATH}/{city_name}/forecast_{indicator.lower().split(' (')[0]}.png")
        logger.info(f"\t{city_name} {indicator}    Forecast for 2020 - {round(p[-1])}, R²={r2}")
        df[f"Forecast {indicator}"] = [round(x) for x in p]
    writer = pd.ExcelWriter(f"{OUTPUT_PATH}/{city_name}/forecast.xlsx", engine='xlsxwriter')
    df.to_excel(excel_writer=writer, sheet_name='Forecast', header=True, index=False)
    writer = format_xlsx(writer, df, 'c' * len(df.columns), sheet_name='Forecast')
    writer.close()


def test():
    city_name = 'Khanty-Mansiysk'
    test_cols = {
        "Number (units)": "int32",
        "Area (ha)": "float32",
        "Forest area (ha)": "float32",
        "Year": "int32",
        "Accumulated temperature": "float32",
        "Accumulated precipitations": "float32",
        "Accumulated precipitations (2 years)": "float32"
    }
    data = pd.read_excel(f"{OUTPUT_PATH}/{city_name}/weather.xlsx", names=list(test_cols.keys()), dtype=test_cols,
                         sheet_name="Sheet1", engine="openpyxl")
    get_regression_regularity(data, 'Number (units)')
    get_regression_regularity(data, 'Area (ha)')
    get_regression_regularity(data, 'Forest area (ha)')


if __name__ == '__main__':
    start_time = time.time()
    logger.info("Started...")
    clean_or_create_dir(OUTPUT_PATH)
    for city in cities:
        clean_or_create_dir(f"{OUTPUT_PATH}/{city}")
        if get_weather_data(city) == 0:
            plot_trends(city)
            get_forecasts(city, True)
    logger.info(f"Done. Elapsed time {round((time.time() - start_time), 1)} seconds")
