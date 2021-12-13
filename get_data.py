import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


def add_date_vars(df_, col_='delivery_date'):
    df_['weekday'] = df_[col_].dt.weekday
    df_['month'] = df_[col_].dt.month
    return df_


def shift(arr, num, timesteps):
    return arr[num:(num+timesteps)]


def reshape_df(df_, timesteps_=15, steps_to_pred_=7):
    windows = [shift(df_, i, timesteps_) for i in range(df_.shape[0] - timesteps_ - steps_to_pred_ + 1)]
    fitting_data = np.reshape(np.vstack(windows), (-1, timesteps_, df_.shape[1]))

    wind_pred = [shift(df_.loc[timesteps_:, 'y'], i, steps_to_pred_) for i in range(df_.shape[0] - timesteps_ - steps_to_pred_ + 1)]
    to_pred = np.reshape(np.vstack(wind_pred), (-1, steps_to_pred_, 1))

    assert fitting_data.shape[0] == to_pred.shape[0], f"X and y must have same number of samples! X has" \
                                                      f" {fitting_data.shape[0]} and y {to_pred.shape[0]}"

    return fitting_data, to_pred


def get_data(filename='banane_091221_last700.csv'):
    """
    Retrieve data from storage (csv)
    :param filename:
    :return: the df must be returned in a (samples, timesteps, features) shape
    """
    df = pd.read_csv(filename)
    df = df.drop(df.columns[0], axis=1)
    df = df.drop(['real_hours', 'entity_promo', 'promo'], axis=1)
    df['delivery_date'] = pd.to_datetime(df['delivery_date'])
    df = add_date_vars(df)
    df = df.drop('delivery_date', axis=1)
    df = df.dropna(how='any')
    x, y = reshape_df(df)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return train_dataset
