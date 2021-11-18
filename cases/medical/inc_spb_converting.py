import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def convert_dataset(path: str, save_path: str):
    """ Preprocess data with info about flu 'inc_spb_daily_allyears.xlsx'

    :param path: path to the source file
    :param save_path: path where converted dataframe to save
    """

    # Read time series data from file
    df = pd.read_excel(path, engine='openpyxl')
    df = df.ffill()

    # Change types
    df = df.astype({'Год': 'int32', 'Месяц': 'int32'})

    # Rename
    df = df.rename(columns={"Год": "year", "Месяц": "month", "День": "day"})

    # Create datetime column. For more detail see:
    # https://stackoverflow.com/questions/58072683/combine-year-month-and-day-in-python-to-create-a-date
    df['datetime'] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.sort_values(by=['datetime'])

    plt.plot(df['datetime'], df['Всего'])
    plt.xlabel('Дата')
    plt.ylabel('Заболеваемость')
    plt.grid()
    plt.show()

    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    convert_dataset(path='inc_spb_daily_allyears.xlsx',
                    save_path='inc_spb_daily_allyears_converted.csv')
