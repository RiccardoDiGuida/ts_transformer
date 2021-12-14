import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def add_date_vars(df_, col_='delivery_date'):
    df_['weekday'] = df_[col_].dt.weekday
    df_['month'] = df_[col_].dt.month
    return df_


def scale_3d_df(df_, scaler_=None, scaler_type_=StandardScaler):
    if scaler_:
        df = scaler_.transform(df_.reshape(-1, df_.shape[-1])).reshape(df_.shape)
    else:
        scaler_ = scaler_type_()
        df = scaler_.fit_transform(df_.reshape(-1, df_.shape[-1])).reshape(df_.shape)
    return df, scaler_


def shift(arr, num, timesteps):
    return arr[num:(num+timesteps)]


def reshape_df(df_, timesteps_=7, steps_to_pred_=3):
    windows = [shift(df_, i, timesteps_) for i in range(df_.shape[0] - timesteps_ - steps_to_pred_ + 1)]
    fitting_data = np.reshape(np.vstack(windows), (-1, timesteps_, df_.shape[1]))

    wind_pred = [shift(df_.loc[timesteps_:, 'y'], i, steps_to_pred_) for i in range(df_.shape[0] - timesteps_ - steps_to_pred_ + 1)]
    to_pred = np.reshape(np.vstack(wind_pred), (-1, steps_to_pred_, 1))

    assert fitting_data.shape[0] == to_pred.shape[0], f"X and y must have same number of samples! X has" \
                                                      f" {fitting_data.shape[0]} and y {to_pred.shape[0]}"

    # Here we make sure that pairs samples-response with na are removed.
    # In this way we ensure that all the days making up a sample o response are one after the other (no gaps)

    n_w_na_x = np.unique(np.argwhere(np.isnan(fitting_data))[:, 0])
    n_w_na_y = np.unique(np.argwhere(np.isnan(to_pred))[:, 0])
    to_remove = np.union1d(n_w_na_x, n_w_na_y)

    fitting_data = np.delete(fitting_data, to_remove, 0)
    to_pred = np.delete(to_pred, to_remove, 0)

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
    x, y = reshape_df(df)
    # Only training for now just for testing purposes (also we have very few samples)
    x, _ = scale_3d_df(x)
    y, _ = scale_3d_df(y)
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    return train_dataset
