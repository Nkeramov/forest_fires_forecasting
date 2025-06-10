import re
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
from fake_useragent import UserAgent
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from typing import NamedTuple, Literal
from sklearn.preprocessing import MinMaxScaler

import matplotlib as mpl
import matplotlib.pyplot as plt

from libs.utils import clear_or_create_dir, crop_image_white_margins, format_xlsx, get_tick_bounds
from libs.log_utils import LoggerSingleton

mpl.rcParams.update({'font.size': 14})

pd.set_option('display.width', 800)
pd.set_option("display.precision", 2)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

HTTP_RETRIES_COUNT = 3
HTTP_REQUEST_DELAY_SECONDS = 0.25
HTTP_REQUEST_TIMEOUT_SECONDS = 10
HTTP_REQUEST_RETRY_DELAY_SECONDS = 3

WINDOW_SIZE = 7
INPUT_PATH = './input'
OUTPUT_PATH = './output'
IMG_WIDTH, IMG_HEIGHT, IMG_DPI = 3600, 2000, 150
WEATHER_URl = "http://pogodaiklimat.ru/monitor.php"

WeatherRecord = NamedTuple('WeatherRecord', [('temperature', float), ('precipitations', float)])
City = NamedTuple('City', [('name', str), ('id', int)])

# IDs of cities from weather site
cities: list[City] = [
    City('Khanty-Mansiysk', 23933),
    City('October', 23734),
    City('Leushi', 28064),
    City('Lariak', 23867),
    City('Ugut', 23946)
]

IndicatorType = Literal['Forest area (ha)', 'Area (ha)', 'Number (units)']
indicators: tuple[IndicatorType, IndicatorType, IndicatorType] = (
    'Forest area (ha)',
    'Area (ha)',
    'Number (units)'
)

logger = LoggerSingleton(
    log_dir=Path('logs'),
    log_file="forecast.log",
    level="INFO",
    colored=True
).get_logger()


def get_weather_data(city: City, date_from: datetime, date_to: datetime) -> bool:
    """
    Function for obtaining weather data for specified city.
    Two files are generated: average monthly temperatures and average monthly precipitations.

    Args:
        city: city for which data is obtained.
        date_from: date from which data is obtained.
        date_to: date up to which data is obtained.

    Returns:
        bool: True if there were no errors and False otherwise.
    """

    def extract_weather_values(text: str) -> WeatherRecord:
        soup = BeautifulSoup(text, 'lxml')
        tags = soup.find_all(['div'], class_='climate-text')
        text = re.sub(r"\s+", " ", tags[1].text.strip())
        float_number_regex = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
        fvalues = [float(x) for x in re.findall(float_number_regex, text)]
        # take the second and fifth value, this is determined by the markup of the site page
        return WeatherRecord(fvalues[1], fvalues[4])

    data = []
    ua = UserAgent()
    dates = pd.date_range(date_from, date_to, freq='MS', inclusive='left')
    for date in tqdm(dates, total=len(dates), colour='green', desc=f"\tObtaining weather data for {city.name}",
                     position=0, leave=True, bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"):
        for k in range(HTTP_RETRIES_COUNT):
            try:
                payload = {'id': city.id, 'month': date.month, 'year': date.year}
                header = {'User-Agent': ua.random}
                response = requests.get(WEATHER_URl, headers=header, params=payload, timeout=HTTP_REQUEST_TIMEOUT_SECONDS)
                if response.status_code == 200:
                    response.encoding = 'utf-8'
                    weather_data = extract_weather_values(response.text)
                    data.append(
                        {
                            'Year': date.year,
                            'Month': date.month,
                            'Temperature': weather_data.temperature,
                            'Precipitations': weather_data.precipitations
                        }
                    )
                    break
                else:
                    time.sleep(HTTP_REQUEST_RETRY_DELAY_SECONDS)
            except requests.exceptions.HTTPError as err:
                logger.warning(f"\tHTTP error, {city.name} {date.month}.{date.year}, {err}. Retrying...")
            except requests.exceptions.ConnectionError as err:
                logger.warning(f"\tConnection error, {city.name} {date.month}.{date.year}, {err}. Retrying...")
            except requests.exceptions.Timeout as err:
                logger.warning(f"\tTimeout error, {city.name} {date.month}.{date.year}, {err}. Retrying...")
            except (requests.exceptions.RequestException, Exception) as err:
                logger.warning(f"\tError, {city.name} {date.month}.{date.year}, {err}. Retrying...")
            if k < HTTP_RETRIES_COUNT - 1:
                time.sleep(HTTP_REQUEST_RETRY_DELAY_SECONDS)
            else:
                logger.error(f"\tError, {city.name} {date.month}.{date.year}. Maximum number of request retries reached")
                return False
        time.sleep(HTTP_REQUEST_DELAY_SECONDS)
    df = pd.DataFrame(data)
    clear_or_create_dir(f"{OUTPUT_PATH}/{city.name}")
    writer = pd.ExcelWriter(f"{OUTPUT_PATH}/{city.name}/weather.xlsx", engine='xlsxwriter')
    df.to_excel(excel_writer=writer, sheet_name='Data', header=True, index=False)
    writer = format_xlsx(writer, df, 'c' * df.shape[1], 'Data')
    writer.close()
    return True


def collect_data(city: City) -> pd.DataFrame:
    """
    Function to collect the complete dataset from fire statistics and a weather data for specified city.

    Args:
        city: city for which data is returned (from cities list).

    Returns:
        pandas.core.frame.DataFrame: dataframe (fires + weather).
    """
    statistic_cols: dict[str, str] = {
        'Number (units)': 'int32',
        'Area (ha)': 'float32',
        'Forest area (ha)': 'float32',
        'Year': 'int32'
    }
    df_statistics = pd.read_excel(f"{INPUT_PATH}/statistics.xlsx", sheet_name='Sheet1',
                                  usecols=list(statistic_cols.keys()), dtype=statistic_cols)
    weather_cols: dict[str, str] = {
        'Year': 'int32',
        'Month': 'int32',
        'Temperature': 'float64',
        'Precipitations': 'float64'
    }
    df_weather = pd.read_excel(f"{OUTPUT_PATH}/{city.name}/weather.xlsx", sheet_name='Data',
                               usecols=list(weather_cols.keys()), dtype=weather_cols)
    df_weather = (df_weather.loc[(df_weather['Month'].between(5, 8, inclusive='both'))]
                          .groupby(by='Year', as_index=False).agg({'Temperature': 'sum', 'Precipitations': 'sum'}))
    df_weather.rename(columns={
        'Temperature': 'Season temperature sum',
        'Precipitations': 'Season precipitations sum'
    }, inplace=True)
    previous_precipitation_sum = df_weather['Season precipitations sum'].shift(1)
    df_weather['Two seasons precipitations sum'] = df_weather['Season precipitations sum'] + previous_precipitation_sum
    df_weather.loc[0, 'Two seasons precipitations sum'] = 2 * df_weather.loc[0, 'Season precipitations sum']
    df_statistics = df_statistics.merge(df_weather, on='Year', how='left', sort=True)
    return df_statistics.copy()


def plot_trends(city: City) -> None:
    """
    Function for plotting graphs with trends for a specified city. Brings data to a scale from 0 to 100 and plots.
    Needed to visualization and explore the dependence of fires on weather data.

    Args:
        city: city for which graphs are plotting (from cities list).
    """
    data = collect_data(city)
    x = data['Year'].tolist()
    data.drop(['Year'], axis=1, inplace=True)
    scaler = MinMaxScaler(feature_range=(0, 100))
    df_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    fig = plt.figure(dpi=IMG_DPI, figsize=(IMG_WIDTH / IMG_DPI, IMG_HEIGHT / IMG_DPI))
    plt.clf()
    plt.title("Statistics for natural fires in Khanty-Mansi Autonomous Okrug-Yugra from 2000 to 2020 years",
              fontsize=24)
    plt.xlabel("year", fontsize=18)
    plt.ylabel("scaled value (from 0 to 100)", fontsize=18)
    y1 = df_scaled['Season temperature sum'].tolist()
    y2 = df_scaled['Season precipitations sum'].tolist()
    y3 = df_scaled['Area (ha)'].tolist()
    plt.plot(x, y1, color='red', linestyle='solid', lw=2, label='Season temperature sum (from May to August)')
    plt.plot(x, y2, color='blue', linestyle='solid', lw=2, label='Season precipitations sum (from May to August)')
    plt.plot(x, y3, color='green', linestyle='solid', lw=2, label='Fire area (ha)')
    maxvalue = max(max(y1), max(y2), max(y3))
    b = get_tick_bounds(maxvalue, 0)
    plt.xticks(x, fontsize=14)
    plt.yticks(np.linspace(start=b[0], stop=b[1], num=b[2], dtype=np.int32), fontsize=14)
    plt.grid(axis='both', linestyle='--')
    plt.legend(loc='upper left', fontsize=18)
    filename = f"{OUTPUT_PATH}/{city.name}/trends.png"
    fig.savefig(filename)
    crop_image_white_margins(filename)


def get_regression_regularity(df: pd.DataFrame, indicator: IndicatorType = 'Area (ha)') -> None:
    """
    Function for correlation analysis. Allows to explore the dependence of the selected indicator
    (fires area, forest fires area, fires number) on the values of the accumulated temperature and precipitations.
    Results are displayed on the screen as polynomial coefficient values.
    Functions fires_number_extrapolation_func and fires_area_extrapolation_func are created based on the results
    of this function.

    Args:
        df: dataframe (fires + weather).
        indicator: indicator for which regression analysis is performed.
    """
    logger.info(indicator)
    x, y = np.array(df['Accumulated temperature']), np.array(df['Accumulated precipitations (2 years)'])
    z = np.array(df[indicator], dtype=np.float64)
    x, y, z = np.meshgrid(x, y, z, copy=False)
    x, y = x.flatten(), y.flatten()
    a = np.array([x * 0 + 1, x, y, x * y, x ** 2, y ** 2, (x ** 2) * y, x * (y ** 2), (x ** 2) * (y ** 2)]).T
    b = z.flatten()
    coeff, r, rank, s = np.linalg.lstsq(a, b, rcond=None)
    logger.info('\t', [round(x, 6) for x in coeff])


def fires_number_extrapolation_func(x: list[np.typing.NDArray[np.float64]], a: np.float64, b: np.float64,
                                        c: np.float64) -> np.typing.NDArray[np.float64]:
    """
    Extrapolation function for the number of fires.

    Args:
        x: list with temperature and precipitations values (lists).
        a: polynomial coefficient.
        b: polynomial coefficient.
        c: polynomial coefficient.
    Returns:
        extrapolation function values.
    """
    if not (len(x) == 2 and all(isinstance(p, np.ndarray) for p in x)):
        raise ValueError("Input 'x' must be a list containing two NumPy arrays.")
    temperature = x[0]
    precipitations = x[1]
    return a + b * temperature + c * precipitations


def fires_area_extrapolation_func(x: list[np.typing.NDArray[np.float64]], a: np.float64, b: np.float64, c: np.float64,
                                        d: np.float64, e: np.float64, f: np.float64) -> np.typing.NDArray[np.float64]:
    """
    Extrapolation function for the area of fires.

    Args:
        x: list with temperature and precipitations values (lists).
        a: polynomial coefficient.
        b: polynomial coefficient.
        c: polynomial coefficient.
        d: polynomial coefficient.
        e: polynomial coefficient.
        f: polynomial coefficient.
    Returns:
        extrapolation function values.
    """
    if not (len(x) == 2 and all(isinstance(p, np.ndarray) for p in x)):
        raise ValueError("Input 'x' must be a list containing two NumPy arrays.")
    temperature = x[0]
    precipitations = x[1]
    return a + b * temperature + c * precipitations + d * temperature * precipitations + \
                e * (temperature * 2) + f * (precipitations * 2)


def get_forecasts(city: City, show_last_year: bool = False) -> None:
    """
    Function of obtaining forecasts. For each city from the cities list, three forecasts are generated:
    for the total area covered by fire, for the forest area covered by fire, for the number of fires.
    The result is graphs and xlsx-report with initial data and forecast.

    Args:
        city: city for which forecast is returned (from cities list).
        show_last_year: if the source data contains a value for the forecast year, then it will be included.
    """
    df = collect_data(city)
    years = df['Year'].tolist()
    for indicator in indicators:
        x = [np.array(df['Season temperature sum'], dtype=np.float64),
             np.array(df['Two seasons precipitations sum'], dtype=np.float64)]
        y = np.array(df[indicator], dtype=np.float64)
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
            popt, pcov = curve_fit(fires_area_extrapolation_func, x, y, maxfev=100000)[:2]
            p = fires_area_extrapolation_func(x, *popt)
        else:
            popt, pcov = curve_fit(fires_number_extrapolation_func, x, y, maxfev=100000)[:2]
            p = fires_number_extrapolation_func(x, *popt)
        r2 = round(r2_score(y, p), 2)
        plt.plot(years, p, color='green', linestyle='solid', lw=2, label=f"Forecast  R² = {r2}")
        maxvalue = np.maximum(np.max(y), np.max(p))
        b = get_tick_bounds(maxvalue, 0)
        plt.xticks(years, fontsize=14)
        plt.yticks(np.linspace(start=b[0], stop=b[1], num=b[2], dtype=np.int32), fontsize=14)
        plt.gca().ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.grid(axis='both', linestyle='--')
        plt.legend(loc='upper left', fontsize=18)
        filename = f"{OUTPUT_PATH}/{city.name}/forecast_{indicator.lower().split(' (')[0]}.png"
        fig.savefig(filename)
        crop_image_white_margins(filename)
        logger.info(f"\t{city.name} {indicator}    Forecast for 2020 - {round(p[-1])}, R²={r2}")
        df[f"Forecast {indicator}"] = [round(x) for x in p]
    writer = pd.ExcelWriter(f"{OUTPUT_PATH}/{city.name}/forecast.xlsx", engine='xlsxwriter')
    df.to_excel(excel_writer=writer, sheet_name='Forecast', header=True, index=False)
    writer = format_xlsx(writer, df, 'c' * len(df.columns), sheet_name='Forecast')
    writer.close()


def test() -> None:
    city = cities[0]
    test_cols = {
        "Number (units)": "int32",
        "Area (ha)": "float32",
        "Forest area (ha)": "float32",
        "Year": "int32",
        "Season temperature sum": "float32",
        "Season precipitations sum": "float32",
        "Two seasons precipitations sum": "float32"
    }
    data = pd.read_excel(f"{OUTPUT_PATH}/{city.name}/forecast.xlsx", sheet_name="Sheet1",
                         usecols=list(test_cols.keys()), dtype=test_cols)
    for indicator in indicators:
        get_regression_regularity(data, indicator)


if __name__ == '__main__':
    start_time = time.perf_counter()
    logger.info("Started...")
    clear_or_create_dir(OUTPUT_PATH)
    start_date = datetime(year=2000, month=1, day=1)
    end_date = datetime(year=2021, month=1, day=1)
    for city in cities:
        if get_weather_data(city, start_date, end_date):
            plot_trends(city)
            get_forecasts(city, True)
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Done. Elapsed time {elapsed_time:.1f} seconds")
