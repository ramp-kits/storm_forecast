import pandas as pd
import numpy as np


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        X_df_new = pd.concat(
            [X_df.get(['instant_t', 'windspeed', 'latitude', 'longitude',
                       'hemisphere', 'Jday_predictor', 'initial_max_wind',
                       'max_wind_change_12h', 'dist2land']),
             pd.get_dummies(X_df.nature, prefix='nature', drop_first=True)],
            # 'basin' is not used here ..but it can!
            axis=1)

        # get data from the future (to test the error message!)
        rolled_windspeeds = np.roll(X_df['windspeed'].values, -2)
        X_df_new = X_df_new.assign(
            future_windspeed=pd.Series(rolled_windspeeds))

        X_df_new = X_df_new.fillna(-1)
        XX = X_df_new.values
        return XX
